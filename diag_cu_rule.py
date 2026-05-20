"""diag_cu_rule.py — diagnose why the Cu rule never fires.

Looks at the 50 hand-labeled `cu` frames and reports:
  1. What METAR genus_hint values exist anywhere in weak_labels.csv
  2. What METAR okta + CSI values the hand-cu frames actually have
  3. What GOES family (low/mid/high) the hand-cu frames get assigned to
  4. How many would be caught by various looser Cu-rule variants

Run:
    .venv/bin/python diag_cu_rule.py
"""
import csv
import collections
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
HAND = ROOT / "labels" / "hand_labeled.csv"
WEAK = ROOT / "labels" / "weak_labels.csv"


def main():
    cu_frames = {r["frame_id"] for r in csv.DictReader(open(HAND))
                 if r["class"] == "cu"}
    print(f"Hand-labeled cu frames: {len(cu_frames)}")

    # Index weak labels by frame_id → {(source, attribute): value}
    weak = {}
    all_genus_hints = collections.Counter()
    for r in csv.DictReader(open(WEAK)):
        weak.setdefault(r["frame_id"], {})[(r["source"], r["attribute"])] = r["value"]
        if r["source"] == "metar" and r["attribute"] == "cloud_genus_hint":
            all_genus_hints[r["value"]] += 1

    print(f"\nGlobal METAR genus_hint distribution (all frames):")
    if not all_genus_hints:
        print("  (no metar genus_hint rows at all)")
    else:
        for g, n in all_genus_hints.most_common():
            print(f"  {g:8s} {n}")

    # Per-cu-frame signal extraction
    oktas = collections.Counter()
    csis = []
    families = collections.Counter()
    no_okta = no_csi = no_height = 0

    for fid in cu_frames:
        w = weak.get(fid, {})
        okta = w.get(("metar", "coverage_okta"))
        csi  = w.get(("derived", "daytime_clear_sky_index"))
        h    = w.get(("goes19_achac", "cloud_top_height_m"))
        if okta:
            try: oktas[int(float(okta))] += 1
            except (TypeError, ValueError): pass
        else:
            no_okta += 1
        if csi:
            try: csis.append(float(csi))
            except (TypeError, ValueError): no_csi += 1
        else:
            no_csi += 1
        if h:
            try:
                hv = float(h)
                if hv <= 0: families["zero"] += 1
                elif hv < 2000: families["low (<2km)"] += 1
                elif hv < 6000: families["mid (2-6km)"] += 1
                else: families["high (>6km)"] += 1
            except (TypeError, ValueError): pass
        else:
            no_height += 1

    print(f"\nHand-cu METAR okta histogram: {dict(oktas)}")
    print(f"  frames missing okta: {no_okta}")

    if csis:
        print(f"\nHand-cu CSI: min={min(csis):.2f} max={max(csis):.2f} "
              f"mean={sum(csis)/len(csis):.2f} n={len(csis)}")
        print(f"  in current Cu rule range (0.55-1.05): "
              f"{sum(1 for c in csis if 0.55 <= c <= 1.05)}/{len(csis)}")
        print(f"  in widened range (0.30-1.10):         "
              f"{sum(1 for c in csis if 0.30 <= c <= 1.10)}/{len(csis)}")
        print(f"  frames missing CSI: {no_csi}")
    else:
        print(f"\nHand-cu CSI: no values at all (all {len(cu_frames)} frames missing)")

    print(f"\nHand-cu GOES family assignment (from cloud_top_height_m):")
    for fam, n in families.most_common():
        print(f"  {fam:14s} {n}")
    print(f"  frames missing GOES height: {no_height}")

    # Rule-firing counts under various variants
    print(f"\n--- Cu rule firing counts under candidate variants ---")

    def fires(fid, csi_lo, csi_hi, okta_min, okta_max, require_low_family):
        w = weak.get(fid, {})
        okta = w.get(("metar", "coverage_okta"))
        csi  = w.get(("derived", "daytime_clear_sky_index"))
        h    = w.get(("goes19_achac", "cloud_top_height_m"))
        try: okta = int(float(okta)) if okta else None
        except: okta = None
        try: csi = float(csi) if csi else None
        except: csi = None
        try: h = float(h) if h else None
        except: h = None
        if okta is None or csi is None:
            return False
        if not (okta_min <= okta <= okta_max):
            return False
        if not (csi_lo <= csi <= csi_hi):
            return False
        if require_low_family:
            if h is None or h <= 0 or h >= 2000:
                return False
        return True

    variants = [
        ("current  (CSI 0.55-1.05, okta 1-4, family=low)",  0.55, 1.05, 1, 4, True),
        ("widen CSI (0.30-1.10, okta 1-4, family=low)",     0.30, 1.10, 1, 4, True),
        ("widen okta (CSI 0.55-1.05, okta 0-5, family=low)", 0.55, 1.05, 0, 5, True),
        ("drop family req (CSI 0.55-1.05, okta 1-4)",        0.55, 1.05, 1, 4, False),
        ("widest (CSI 0.30-1.10, okta 0-5, any family)",     0.30, 1.10, 0, 5, False),
    ]
    for name, *args in variants:
        n = sum(1 for fid in cu_frames if fires(fid, *args))
        print(f"  {name:55s} → fires on {n}/{len(cu_frames)}")


if __name__ == "__main__":
    main()
