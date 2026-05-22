"""analyze_labels.py — Where is the auto-classifier failing, and what should
we label next?

Reads labels/hand_labeled.csv, labels/auto_labels.csv, labels/weak_labels.csv
and emits a multi-section report:

  1. Overall agreement (hand vs auto).
  2. Per-class precision / recall (treating hand labels as ground truth).
  3. Full confusion matrix.
  4. Calibration — is "high confidence" actually more accurate?
  5. Per-regime breakdown (DAY / TWILIGHT / NAUTICAL / ASTRO / DARK):
     hand-label class distribution + agreement + calibration.
  6. Under-represented classes — what weather conditions to wait for.
  7. Top disagreement patterns + sample frame_ids to review.
  8. Recommendations.

Run:
    .venv/bin/python analyze_labels.py
    .venv/bin/python analyze_labels.py --include-qc        # don't filter QC-flagged frames
    .venv/bin/python analyze_labels.py --out report.md     # also write markdown
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
HAND_CSV = PROJECT_ROOT / "labels" / "hand_labeled.csv"
AUTO_CSV = PROJECT_ROOT / "labels" / "auto_labels.csv"
WEAK_CSV = PROJECT_ROOT / "labels" / "weak_labels.csv"

CLASSES = ["clear", "ci", "cs_cc", "ac_as", "cu", "sc", "st", "ns_cb", "multi"]
CONFIDENCES = ["high", "medium", "low"]
REGIMES = ["DAY", "TWILIGHT", "NAUTICAL", "ASTRO", "DARK"]
QC_FLAGS = ["sun_artifact", "lens_contamination", "rain_on_lens",
            "nighttime_no_moon", "horizon_contamination", "smoke"]

# Minimum frames per class to consider it "trainable" for a CNN baseline.
# Below this, the class is flagged as under-represented in the report.
TRAINABLE_MIN = 50


def _sun_regime(sun_alt_deg: float) -> str:
    if sun_alt_deg >= 6: return "DAY"
    if sun_alt_deg >= -6: return "TWILIGHT"
    if sun_alt_deg >= -12: return "NAUTICAL"
    if sun_alt_deg >= -18: return "ASTRO"
    return "DARK"


def load_regimes() -> dict[str, str]:
    """Build {frame_id: regime} from weak_labels.csv ephemeris.sun_alt_deg rows."""
    if not WEAK_CSV.exists():
        return {}
    out: dict[str, str] = {}
    with open(WEAK_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] != "ephemeris" or r["attribute"] != "sun_alt_deg":
                continue
            try:
                out[r["frame_id"]] = _sun_regime(float(r["value"]))
            except (TypeError, ValueError):
                continue
    return out


def has_qc_flag(row: pd.Series) -> bool:
    for f in QC_FLAGS:
        v = row.get(f, False)
        if str(v).strip().lower() in {"true", "1"}:
            return True
    return False


# ---------- formatting helpers ----------

def _hr(out, char="─", width=78):
    print(char * width, file=out)


def _section(out, title: str):
    print(file=out)
    _hr(out, "═")
    print(f"  {title}", file=out)
    _hr(out, "═")


def _subsection(out, title: str):
    print(file=out)
    print(f"── {title} " + "─" * (74 - len(title)), file=out)


def _bar(n: int, total_max: int, width: int = 30) -> str:
    if total_max <= 0:
        return ""
    return "█" * (n * width // total_max)


# ---------- report sections ----------

def section_overall(out, merged: pd.DataFrame, hand: pd.DataFrame,
                    auto: pd.DataFrame, n_qc_excluded: int) -> None:
    _section(out, "1. OVERALL AGREEMENT")
    print(f"  Hand-labeled frames:    {len(hand)}", file=out)
    print(f"  Auto-classified frames: {len(auto)}", file=out)
    print(f"  Hand frames also in auto: {len(merged)}", file=out)
    if n_qc_excluded:
        print(f"  Excluded (QC-flagged):  {n_qc_excluded}", file=out)
    missing = len(hand) - len(merged) - n_qc_excluded
    if missing > 0:
        print(f"  ⚠ {missing} hand-labeled frames not in auto_labels.csv "
              "— re-run auto_classify_batch.py to refresh.", file=out)
    if len(merged) == 0:
        print("  No overlapping frames; nothing to analyze.", file=out)
        return
    n_match = int((merged["class_hand"] == merged["auto_class"]).sum())
    print(f"\n  Exact-match agreement: {n_match}/{len(merged)} = "
          f"{100 * n_match / len(merged):.1f}%", file=out)
    n_unknown = int((auto["auto_class"] == "unknown").sum())
    if n_unknown:
        n_unknown_in_merged = int((merged["auto_class"] == "unknown").sum())
        print(f"  Classifier punted (auto=unknown): {n_unknown} total "
              f"({n_unknown_in_merged} in matched set) — these are signal-disagreement "
              "frames, prime active-learning targets.", file=out)


def section_per_class(out, merged: pd.DataFrame) -> None:
    """Treat hand labels as ground truth; report precision/recall/F1 per class."""
    _section(out, "2. PER-CLASS PRECISION / RECALL (hand = ground truth)")
    rows = []
    for c in CLASSES:
        hand_c = merged["class_hand"] == c
        auto_c = merged["auto_class"] == c
        tp = int((hand_c & auto_c).sum())
        fp = int((~hand_c & auto_c).sum())
        fn = int((hand_c & ~auto_c).sum())
        support = int(hand_c.sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append((c, support, precision, recall, f1))
    print(f"  {'class':8s}  {'support':>8s}  {'precision':>10s}  "
          f"{'recall':>8s}  {'F1':>6s}", file=out)
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 6}", file=out)
    for c, support, p, r, f in rows:
        if support == 0 and (p + r) == 0:
            continue
        flag = "  ⚠ low support" if 0 < support < 10 else ""
        print(f"  {c:8s}  {support:>8d}  {p:>10.1%}  {r:>8.1%}  {f:>6.2f}{flag}",
              file=out)


def section_confusion(out, merged: pd.DataFrame) -> None:
    _section(out, "3. CONFUSION MATRIX (rows = hand truth, cols = auto pred)")
    cm = pd.crosstab(merged["class_hand"], merged["auto_class"],
                     rownames=[""], colnames=[""], dropna=False)
    # Include "unknown" as a column — those are frames the classifier punted
    # on (signal disagreement, no family resolution). Keep it separate from
    # taxonomy "multi" so the diagnostic doesn't conflate them.
    cm = cm.reindex(index=CLASSES, columns=CLASSES + ["unknown"], fill_value=0)
    cm["TOTAL"] = cm.sum(axis=1)
    cm.loc["TOTAL"] = cm.sum(axis=0)
    # render with hand=row totals so it's obvious where errors live
    buf = StringIO()
    cm.to_string(buf)
    for line in buf.getvalue().splitlines():
        print(f"  {line}", file=out)
    print(file=out)
    print("  Diagonal = agreement. Off-diagonal cell (i,j) = hand-labeled `i` "
          "auto-classified as `j`.", file=out)


def section_calibration(out, merged: pd.DataFrame) -> None:
    _section(out, "4. CONFIDENCE CALIBRATION")
    print("  Does 'high confidence' actually mean higher accuracy?", file=out)
    print(file=out)
    print(f"  {'conf':8s}  {'n':>6s}  {'correct':>8s}  {'agreement':>10s}  bar",
          file=out)
    print(f"  {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 10}", file=out)
    counts = []
    for conf in CONFIDENCES:
        sub = merged[merged["auto_confidence"] == conf]
        n = len(sub)
        correct = int((sub["class_hand"] == sub["auto_class"]).sum())
        counts.append((conf, n, correct))
    max_n = max((n for _, n, _ in counts), default=1)
    for conf, n, correct in counts:
        if n == 0:
            print(f"  {conf:8s}  {n:>6d}  {'—':>8s}  {'—':>10s}", file=out)
            continue
        pct = 100 * correct / n
        print(f"  {conf:8s}  {n:>6d}  {correct:>8d}  {pct:>9.1f}%  "
              f"{_bar(n, max_n)}", file=out)
    print(file=out)
    # Monotonicity check — high should be > medium > low
    pcts = {c: (correct / n) if n else None
            for c, n, correct in counts}
    if all(pcts[c] is not None for c in CONFIDENCES):
        if pcts["high"] > pcts["medium"] > pcts["low"]:
            print("  ✓ Calibration monotone (high > medium > low).", file=out)
        else:
            print("  ⚠ Calibration NOT monotone — confidence scoring may need "
                  "review:", file=out)
            print(f"      high={pcts['high']:.1%}  medium={pcts['medium']:.1%}  "
                  f"low={pcts['low']:.1%}", file=out)


def section_by_regime(out, merged: pd.DataFrame, regimes: dict[str, str]) -> None:
    _section(out, "5. PER-REGIME BREAKDOWN")
    if not regimes:
        print("  No ephemeris weak labels found — re-run fetch_local_sensors.py.",
              file=out)
        return

    merged = merged.copy()
    merged["regime"] = merged["frame_id"].map(regimes).fillna("?")

    print(f"  {'regime':10s}  {'frames':>7s}  {'agree':>7s}  "
          f"{'high_n':>7s}  {'high_acc':>9s}  {'med+low':>8s}",
          file=out)
    print(f"  {'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 7}  "
          f"{'─' * 9}  {'─' * 8}", file=out)
    for regime in REGIMES:
        sub = merged[merged["regime"] == regime]
        n = len(sub)
        if n == 0:
            continue
        agree = int((sub["class_hand"] == sub["auto_class"]).sum())
        high = sub[sub["auto_confidence"] == "high"]
        high_n = len(high)
        high_correct = int((high["class_hand"] == high["auto_class"]).sum())
        high_acc = (100 * high_correct / high_n) if high_n else 0
        med_low = int((sub["auto_confidence"].isin(["medium", "low"])).sum())
        print(f"  {regime:10s}  {n:>7d}  {100 * agree / n:>6.1f}%  "
              f"{high_n:>7d}  {high_acc:>8.1f}%  {med_low:>8d}", file=out)

    _subsection(out, "Hand-label class distribution per regime")
    pivot = pd.crosstab(merged["regime"], merged["class_hand"], dropna=False)
    pivot = pivot.reindex(index=REGIMES, columns=CLASSES, fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1) > 0]
    buf = StringIO()
    pivot.to_string(buf)
    for line in buf.getvalue().splitlines():
        print(f"  {line}", file=out)


def section_underrepresented(out, merged: pd.DataFrame, hand: pd.DataFrame) -> None:
    _section(out, "6. UNDER-REPRESENTED CLASSES (what weather to wait for)")
    counts = hand["class"].value_counts().reindex(CLASSES, fill_value=0)
    max_n = int(counts.max()) if len(counts) else 1
    print(f"  Trainable threshold (per class): {TRAINABLE_MIN} frames", file=out)
    print(file=out)
    print(f"  {'class':8s}  {'hand_n':>7s}  {'auto_n':>7s}  status   bar",
          file=out)
    print(f"  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 8}", file=out)
    auto_counts = merged["auto_class"].value_counts().reindex(CLASSES, fill_value=0)
    todo = []
    for c in CLASSES:
        n = int(counts[c])
        a = int(auto_counts[c])
        if n == 0:
            status = "MISSING "
        elif n < TRAINABLE_MIN // 4:
            status = "URGENT  "
        elif n < TRAINABLE_MIN // 2:
            status = "scarce  "
        elif n < TRAINABLE_MIN:
            status = "thin    "
        else:
            status = "ok      "
        if status.strip() in {"MISSING", "URGENT", "scarce", "thin"}:
            todo.append((c, n, status.strip()))
        print(f"  {c:8s}  {n:>7d}  {a:>7d}  {status}  {_bar(n, max_n)}", file=out)
    if todo:
        print(file=out)
        print("  Priority weather conditions to wait for (and label when they appear):",
              file=out)
        for c, n, status in todo:
            print(f"    • {c:8s} (n={n}, {status}) — {_class_weather_hint(c)}", file=out)


def _class_weather_hint(c: str) -> str:
    hints = {
        "clear":  "blue-sky days with no contrails",
        "ci":     "high thin streaks at sunset, often pink/orange — fast-moving fronts",
        "cs_cc":  "milky-white sheet covering whole sky, sun visible as disc; or ripples",
        "ac_as":  "mid-level grey sheet with sun barely visible; or honeycombed cells",
        "cu":     "summer afternoon convective: fluffy puffs with flat bases and blue gaps",
        "sc":     "low rolls/patches, mostly grey-white, often along chinook arch",
        "st":     "uniform low grey lid; fog mornings",
        "ns_cb":  "rain/snow falling; thunderstorm anvil; whole sky dark grey or black",
        "multi":  "two distinct cloud decks at once (e.g. Cu below, Ci above)",
    }
    return hints.get(c, "")


def section_disagreements(out, merged: pd.DataFrame, k: int = 5) -> None:
    _section(out, "7. TOP DISAGREEMENT PATTERNS")
    bad = merged[merged["class_hand"] != merged["auto_class"]]
    punts = bad[bad["auto_class"] == "unknown"]
    true_bad = bad[bad["auto_class"] != "unknown"]
    total = len(merged)

    pct = lambda n: 100 * n / max(total, 1)
    print(f"  Total disagreements:           {len(bad):>4d} / {total} ({pct(len(bad)):>5.1f}%)", file=out)
    print(f"    True misclassifications:     {len(true_bad):>4d} / {total} "
          f"({pct(len(true_bad)):>5.1f}%) — classifier confidently wrong, fix rules",
          file=out)
    print(f"    Classifier punts (unknown):  {len(punts):>4d} / {total} "
          f"({pct(len(punts)):>5.1f}%) — labeler resolved, no rule fix needed",
          file=out)
    if not len(bad):
        return

    if len(true_bad):
        print(file=out)
        print("  ── TRUE MISCLASSIFICATIONS (rule-tuning targets) ──", file=out)
        grouped = (true_bad.groupby(["class_hand", "auto_class"])
                          .size().reset_index(name="n")
                          .sort_values("n", ascending=False))
        print(f"  {'hand→auto':24s}  {'n':>4s}  example frame_ids", file=out)
        print(f"  {'─' * 24}  {'─' * 4}", file=out)
        for _, row in grouped.head(15).iterrows():
            pair = f"{row['class_hand']:8s} → {row['auto_class']:8s}"
            examples = (true_bad[(true_bad["class_hand"] == row["class_hand"])
                                & (true_bad["auto_class"] == row["auto_class"])]
                        ["frame_id"].head(k).tolist())
            print(f"  {pair:24s}  {row['n']:>4d}  {', '.join(examples)}", file=out)

    if len(punts):
        print(file=out)
        print("  ── PUNTS by hand class (signal-disagreement frames) ──", file=out)
        punt_by_class = (punts.groupby("class_hand").size()
                              .reset_index(name="n").sort_values("n", ascending=False))
        print(f"  {'hand class':24s}  {'n':>4s}  example frame_ids", file=out)
        print(f"  {'─' * 24}  {'─' * 4}", file=out)
        for _, row in punt_by_class.iterrows():
            examples = (punts[punts["class_hand"] == row["class_hand"]]
                        ["frame_id"].head(k).tolist())
            label = f"{row['class_hand']:8s} → unknown"
            print(f"  {label:24s}  {row['n']:>4d}  {', '.join(examples)}", file=out)


def section_recommendations(out, merged: pd.DataFrame, hand: pd.DataFrame,
                            regimes: dict[str, str]) -> None:
    _section(out, "8. RECOMMENDATIONS")
    notes: list[str] = []

    # Coverage of overall hand vs auto
    n_hand = len(hand)
    n_merged = len(merged)
    if n_merged < n_hand * 0.95:
        notes.append(
            f"Auto_labels.csv covers only {n_merged}/{n_hand} hand-labeled frames. "
            "Run `.venv/bin/python auto_classify_batch.py` to refresh."
        )

    # Calibration — record both accuracy and sample size so we can suppress
    # warnings on tiny samples (e.g. n_high=2 reports "50%" which is noise).
    cal = {}
    cal_n = {}
    for c in CONFIDENCES:
        sub = merged[merged["auto_confidence"] == c]
        if len(sub):
            cal[c] = int((sub["class_hand"] == sub["auto_class"]).sum()) / len(sub)
            cal_n[c] = len(sub)
    HIGH_MIN_N = 20
    if cal_n.get("high", 0) < HIGH_MIN_N:
        if cal_n.get("high", 0) > 0:
            notes.append(
                f"High-confidence verdicts: only {cal_n['high']} frames "
                f"({cal.get('high', 0):.0%} agreement) — below the n≥{HIGH_MIN_N} "
                "threshold needed to judge calibration. Headline accuracy is "
                "statistically meaningless at this sample size; ignore until "
                "more `high` verdicts accumulate or the gate is loosened."
            )
    elif cal.get("high", 0) < 0.90:
        notes.append(
            f"High-confidence accuracy is {cal.get('high', 0):.1%} ({cal_n['high']} "
            "frames) — below the 90% threshold needed for the paper's "
            "'auto-accept' claim. Inspect the top disagreement patterns and "
            "consider tightening the rules in auto_classify.py that gate "
            "the `high` verdict."
        )

    # Under-represented
    cnt = hand["class"].value_counts().reindex(CLASSES, fill_value=0)
    missing = [c for c in CLASSES if cnt[c] == 0]
    urgent = [c for c in CLASSES if 0 < cnt[c] < TRAINABLE_MIN // 4]
    if missing:
        notes.append("Classes with ZERO hand labels: "
                     + ", ".join(missing)
                     + ". A CNN baseline will be impossible until these have ≥10–20 examples.")
    if urgent:
        notes.append("Classes urgently under-represented (<13 frames): "
                     + ", ".join(urgent)
                     + ". Prioritize labeling when this weather appears.")

    # Regime coverage
    if regimes:
        merged2 = merged.copy()
        merged2["regime"] = merged2["frame_id"].map(regimes).fillna("?")
        for r in REGIMES:
            sub = merged2[merged2["regime"] == r]
            high = sub[sub["auto_confidence"] == "high"]
            if len(high) >= 10:
                acc = int((high["class_hand"] == high["auto_class"]).sum()) / len(high)
                if acc < 0.85:
                    notes.append(
                        f"In {r} regime, high-confidence accuracy is only {acc:.1%} "
                        f"({len(high)} frames). The confidence-scoring rules may not "
                        f"be regime-aware enough — consider per-regime calibration."
                    )

    # General next-step suggestions
    notes.append(
        "While waiting for more data, use the **regime filter + auto-confidence "
        "filter** in the labeling tool to spot-check the medium-confidence frames "
        "(active learning) — these are where each human label adds the most info."
    )
    notes.append(
        "Hold off on a CNN baseline until at least 1000 frames AND every class "
        "has ≥50. With 380 spread across imbalanced classes, a CNN will memorize."
    )

    for i, n in enumerate(notes, 1):
        # rough word-wrap for readability
        words = n.split()
        line = f"  {i}."
        for w in words:
            if len(line) + len(w) > 78:
                print(line, file=out)
                line = "    "
            line += " " + w
        print(line, file=out)
        print(file=out)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-qc", action="store_true",
                    help="Include QC-flagged frames in the agreement analysis "
                         "(default: exclude — bad lighting/optics aren't classifier failures)")
    ap.add_argument("--out", default=None,
                    help="Also write the report to this file (e.g. report.md)")
    args = ap.parse_args()

    if not HAND_CSV.exists():
        print(f"ERROR: {HAND_CSV} not found.")
        return
    if not AUTO_CSV.exists():
        print(f"ERROR: {AUTO_CSV} not found. Run auto_classify_batch.py first.")
        return

    hand = pd.read_csv(HAND_CSV)
    auto = pd.read_csv(AUTO_CSV)
    regimes = load_regimes()

    n_qc_excluded = 0
    if not args.include_qc:
        before = len(hand)
        hand = hand[~hand.apply(has_qc_flag, axis=1)]
        n_qc_excluded = before - len(hand)

    merged = hand.merge(
        auto[["frame_id", "auto_class", "auto_confidence"]],
        on="frame_id", how="inner", suffixes=("_hand", "_auto"),
    ).rename(columns={"class": "class_hand"})

    # Build the report into a buffer so we can both print and (optionally) save
    out = StringIO()
    print(file=out)
    print("  ALLSKY CLOUD CLASSIFIER — HAND vs AUTO ANALYSIS", file=out)
    print(f"  Hand labels: {HAND_CSV}", file=out)
    print(f"  Auto labels: {AUTO_CSV}", file=out)

    section_overall(out, merged, hand, auto, n_qc_excluded)
    if len(merged) > 0:
        section_per_class(out, merged)
        section_confusion(out, merged)
        section_calibration(out, merged)
        section_by_regime(out, merged, regimes)
        section_underrepresented(out, merged, hand)
        section_disagreements(out, merged)
        section_recommendations(out, merged, hand, regimes)

    report = out.getvalue()
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
