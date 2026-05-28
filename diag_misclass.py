"""
diag_misclass.py — dump every signal feeding the classifier for a set of frames.

Read-only. For each frame_id given (args or stdin), prints:
  - hand label (from labels/hand_labeled.csv)
  - auto verdict + confidence + reasoning + the mask-derived stats
    (from labels/auto_labels.csv — already computed by auto_classify_batch)
  - the raw weak-label signals most relevant to family/genus resolution
    (from labels/weak_labels.csv)

Use it to understand WHY a frame was misclassified before touching auto_classify
rules. Group by a misclassification pattern, e.g.:

  grep -E 'ns_cb|sc' labels/agreement_*.md      # find the frame_ids, or
  python diag_misclass.py ccd1_20260520_114659 ccd1_20260520_112337 ...
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
HAND = ROOT / "labels" / "hand_labeled.csv"
AUTO = ROOT / "labels" / "auto_labels.csv"
WEAK = ROOT / "labels" / "weak_labels.csv"

# (source, attribute) pairs worth showing, in display order.
SIGNALS = [
    ("ephemeris", "sun_alt_deg"),
    ("ephemeris", "moon_alt_deg"),
    ("ephemeris", "moon_phase_pct"),
    ("goes19_acmc", "cloud_present"),
    ("goes19_actpc", "cloud_top_phase"),
    ("goes19_achac", "cloud_top_height_m"),
    ("metar", "coverage_okta"),
    ("metar", "cloud_genus_hint"),
    ("metar", "altitude_bucket"),
    ("metar", "cloud_base_height_m"),
    ("weather_station", "rain_1h_mm"),
    ("weather_station", "humidity_pct"),
    ("derived", "daytime_clear_sky_index"),
    ("derived", "csi_std_10min"),
]

AUTO_STAT_COLS = ["thermal_mean_p", "thermal_std", "rgb_nrbr_p95",
                  "rgb_v_mean", "rgb_v_std", "alignment_mi"]


def load_hand() -> dict[str, str]:
    out: dict[str, str] = {}
    if not HAND.exists():
        return out
    with open(HAND, newline="") as f:
        for r in csv.DictReader(f):
            fid = r.get("frame_id")
            cls = r.get("class") or r.get("label") or ""
            if fid:
                out[fid] = cls
    return out


def load_auto() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not AUTO.exists():
        return out
    with open(AUTO, newline="") as f:
        for r in csv.DictReader(f):
            out[r["frame_id"]] = r
    return out


def load_weak(frame_ids: set[str]) -> dict[str, dict[tuple, str]]:
    out: dict[str, dict[tuple, str]] = {fid: {} for fid in frame_ids}
    if not WEAK.exists():
        return out
    with open(WEAK, newline="") as f:
        for r in csv.DictReader(f):
            fid = r["frame_id"]
            if fid in out:
                out[fid][(r["source"], r["attribute"])] = r["value"]
    return out


def main() -> None:
    frame_ids = sys.argv[1:]
    if not frame_ids:
        frame_ids = [ln.strip() for ln in sys.stdin if ln.strip()]
    if not frame_ids:
        sys.exit("usage: diag_misclass.py <frame_id> [frame_id ...]  (or pipe ids on stdin)")

    hand = load_hand()
    auto = load_auto()
    weak = load_weak(set(frame_ids))

    for fid in frame_ids:
        a = auto.get(fid, {})
        h = hand.get(fid, "—")
        print("═" * 78)
        print(f"{fid}")
        print(f"  hand={h:8s}  auto={a.get('auto_class','?'):8s} "
              f"conf={a.get('auto_confidence','?'):6s}")
        print(f"  reasoning: {a.get('auto_reasoning','—')}")
        stats = "  ".join(f"{c}={a[c]}" for c in AUTO_STAT_COLS if a.get(c))
        if stats:
            print(f"  stats: {stats}")
        w = weak.get(fid, {})
        print("  signals:")
        for key in SIGNALS:
            if key in w:
                print(f"    {key[0]:16s} {key[1]:24s} {w[key]}")
    print("═" * 78)


if __name__ == "__main__":
    main()
