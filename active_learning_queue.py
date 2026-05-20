"""active_learning_queue.py — rank unlabeled frames by labeling value.

Picks the next frames you should hand-label to maximize information gain
for the classifier. Combines four signals into a per-frame score:

  class_scarcity   1 / (1 + hand_count[auto_class])
                   under-represented classes are worth more

  regime_scarcity  1 / (1 + hand_count_in_regime)
                   under-represented regimes are worth more

  uncertainty      0 for high-conf auto, 1 for medium, 2 for low
                   uncertain predictions need human validation most

  novel_class      bonus when auto predicts a class that hand has <10 of —
                   these are exactly the predictions you can't yet trust

A diversity constraint keeps consecutive frames from the same 5-minute
window from dominating the queue (sequential frames from one weather
event are redundant — one labeled example per window is enough).

Output: top-N ranked frames with frame_id, auto-class, confidence,
regime, score, and a short "why this frame" explanation. Paste a
frame_id stem (or just the date prefix) into the labeling tool's
"Frame ID contains" search to jump to it.

Run:
    .venv/bin/python active_learning_queue.py
    .venv/bin/python active_learning_queue.py --top 30 --regime NAUTICAL
    .venv/bin/python active_learning_queue.py --min-spacing-min 15 --out queue.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
HAND_CSV = PROJECT_ROOT / "labels" / "hand_labeled.csv"
AUTO_CSV = PROJECT_ROOT / "labels" / "auto_labels.csv"
WEAK_CSV = PROJECT_ROOT / "labels" / "weak_labels.csv"

REGIMES = ["DAY", "TWILIGHT", "NAUTICAL", "ASTRO", "DARK", "UNKNOWN"]
NOVEL_CLASS_THRESHOLD = 10  # classes with fewer hand labels than this get the bonus


def sun_regime(sun_alt_deg: float) -> str:
    if sun_alt_deg >= 6: return "DAY"
    if sun_alt_deg >= -6: return "TWILIGHT"
    if sun_alt_deg >= -12: return "NAUTICAL"
    if sun_alt_deg >= -18: return "ASTRO"
    return "DARK"


def parse_ts(stem: str) -> dt.datetime | None:
    m = re.search(r"(\d{8}_\d{6})", stem)
    if not m: return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def load_regimes() -> dict[str, str]:
    out: dict[str, str] = {}
    if not WEAK_CSV.exists():
        return out
    with open(WEAK_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] != "ephemeris" or r["attribute"] != "sun_alt_deg":
                continue
            try:
                out[r["frame_id"]] = sun_regime(float(r["value"]))
            except (TypeError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50,
                    help="how many frames to output (default 50)")
    ap.add_argument("--regime", choices=REGIMES, default=None,
                    help="restrict to one regime")
    ap.add_argument("--auto-class", default=None,
                    help="restrict to one auto-predicted class")
    ap.add_argument("--min-spacing-min", type=float, default=5.0,
                    help="minimum minutes between selected frames (default 5)")
    ap.add_argument("--out", default=None,
                    help="optional CSV output (in addition to stdout)")
    args = ap.parse_args()

    if not HAND_CSV.exists():
        print(f"ERROR: {HAND_CSV} not found")
        return
    if not AUTO_CSV.exists():
        print(f"ERROR: {AUTO_CSV} not found — run auto_classify_batch.py first")
        return

    # --- load everything ---
    hand_ids = set()
    hand_class_counts: Counter = Counter()
    with open(HAND_CSV, newline="") as f:
        for r in csv.DictReader(f):
            hand_ids.add(r["frame_id"])
            hand_class_counts[r["class"]] += 1

    regimes = load_regimes()
    hand_regime_counts: Counter = Counter()
    for fid in hand_ids:
        hand_regime_counts[regimes.get(fid, "UNKNOWN")] += 1

    auto_rows = []
    with open(AUTO_CSV, newline="") as f:
        for r in csv.DictReader(f):
            auto_rows.append(r)

    # --- score each unlabeled frame ---
    candidates = []
    for r in auto_rows:
        fid = r["frame_id"]
        if fid in hand_ids:
            continue
        auto_cls = r.get("auto_class", "")
        auto_conf = r.get("auto_confidence", "low")
        regime = regimes.get(fid, "UNKNOWN")

        if args.regime and regime != args.regime:
            continue
        if args.auto_class and auto_cls != args.auto_class:
            continue

        class_scarcity = 1.0 / (1.0 + hand_class_counts.get(auto_cls, 0))
        regime_scarcity = 1.0 / (1.0 + hand_regime_counts.get(regime, 0))
        uncertainty = {"high": 0, "medium": 1, "low": 2}.get(auto_conf, 1)
        novel_bonus = (1.5 if hand_class_counts.get(auto_cls, 0) < NOVEL_CLASS_THRESHOLD
                       else 0.0)

        score = (
            3.0 * class_scarcity
            + 2.0 * regime_scarcity
            + 0.5 * uncertainty
            + novel_bonus
        )

        # Why-line — pick the single dominant reason for readability
        reasons = []
        if novel_bonus > 0:
            reasons.append(f"auto says {auto_cls} (only {hand_class_counts.get(auto_cls, 0)} hand labels — validate)")
        elif class_scarcity > 1.0 / 20:
            reasons.append(f"{auto_cls} under-represented ({hand_class_counts.get(auto_cls, 0)} hand)")
        if uncertainty >= 1:
            reasons.append(f"{auto_conf}-confidence")
        if regime_scarcity > 1.0 / 30:
            reasons.append(f"{regime} thin ({hand_regime_counts.get(regime, 0)} hand)")
        why = "; ".join(reasons) if reasons else "general coverage"

        ts = parse_ts(fid)
        candidates.append({
            "frame_id": fid,
            "auto_class": auto_cls,
            "auto_conf": auto_conf,
            "regime": regime,
            "score": score,
            "ts": ts,
            "why": why,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    # --- diversity: enforce minimum time spacing between selected frames ---
    min_spacing = dt.timedelta(minutes=args.min_spacing_min)
    selected = []
    selected_ts = []
    for c in candidates:
        if len(selected) >= args.top:
            break
        if c["ts"] is None:
            selected.append(c)
            continue
        too_close = any(abs((c["ts"] - t).total_seconds()) < min_spacing.total_seconds()
                        for t in selected_ts)
        if too_close:
            continue
        selected.append(c)
        selected_ts.append(c["ts"])

    # --- print ---
    print(f"\nActive learning queue — top {len(selected)} unlabeled frames")
    print(f"  (from {len(candidates)} candidates, "
          f"hand-labeled {len(hand_ids)} excluded)")
    if args.regime:
        print(f"  filter: regime = {args.regime}")
    if args.auto_class:
        print(f"  filter: auto_class = {args.auto_class}")
    print(f"  min spacing between selections: {args.min_spacing_min:.1f} min")
    print()

    # Summary of what's in the queue
    sel_by_class = Counter(c["auto_class"] for c in selected)
    sel_by_regime = Counter(c["regime"] for c in selected)
    print(f"  By auto-class:  {dict(sel_by_class)}")
    print(f"  By regime:      {dict(sel_by_regime)}")
    print()
    print(f"  Current hand-label distribution (for reference):")
    for cls in sorted(hand_class_counts, key=hand_class_counts.get, reverse=True):
        print(f"    {cls:8s} {hand_class_counts[cls]}")
    print()

    print(f"  {'#':>3}  {'frame_id':32s}  {'auto':8s}  {'conf':6s}  "
          f"{'regime':10s}  {'score':>5s}  why")
    print(f"  {'─' * 3}  {'─' * 32}  {'─' * 8}  {'─' * 6}  "
          f"{'─' * 10}  {'─' * 5}")
    for i, c in enumerate(selected, 1):
        print(f"  {i:>3}  {c['frame_id']:32s}  {c['auto_class']:8s}  "
              f"{c['auto_conf']:6s}  {c['regime']:10s}  "
              f"{c['score']:>5.2f}  {c['why']}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "rank", "frame_id", "auto_class", "auto_conf",
                "regime", "score", "why",
            ])
            w.writeheader()
            for i, c in enumerate(selected, 1):
                w.writerow({
                    "rank": i,
                    "frame_id": c["frame_id"],
                    "auto_class": c["auto_class"],
                    "auto_conf": c["auto_conf"],
                    "regime": c["regime"],
                    "score": f"{c['score']:.3f}",
                    "why": c["why"],
                })
        print(f"\nQueue written to {args.out}")


if __name__ == "__main__":
    main()
