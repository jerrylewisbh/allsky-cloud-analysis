"""
backfill_alignment_mi.py — compute alignment_mi for existing dataset_v2_*
masks + images and patch it into the per-frame meta JSONs.

Use this when mask-gen ran with an older version of make_masks_v2.py that
didn't compute alignment_mi. Cheaper than re-running the full mask-gen
pipeline since this only does the MI computation.

Note: MI is computed on the resized 256x256 outputs (not the pre-resize
crop the live pipeline uses). The absolute MI values are slightly lower
than what the live pipeline would write, but the relative ordering is
preserved — sufficient for picking calibration frames in Phase 2.

Usage:
  python backfill_alignment_mi.py                          # all datasets
  python backfill_alignment_mi.py --datasets dataset_v2_20260521
  python backfill_alignment_mi.py --skip-existing          # don't overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from make_masks_v2 import alignment_mi_score, NO_DATA


PROJECT_ROOT = Path(__file__).parent.resolve()


def backfill_one(mask_path: Path, img_path: Path, meta_path: Path,
                 skip_existing: bool = False) -> tuple[bool, str]:
    """Compute alignment_mi for one frame, patch its meta JSON.
    Returns (success, status) where status is one of:
      'updated', 'no-rgb', 'no-meta', 'night', 'skipped', 'no-mi'.
    """
    if not meta_path.exists():
        return False, "no-meta"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        return False, "no-meta"

    if skip_existing and meta.get("alignment_mi") is not None:
        return True, "skipped"

    # alignment_mi is daytime-only — gate on lux from existing meta
    lux = meta.get("lux", 0.0)
    if lux is None or lux < 1.0:
        # Mirror the daytime/night gating in make_masks_v2 (LUX_NIGHT = 1.0)
        meta["alignment_mi"] = None
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return True, "night"

    if not img_path.exists():
        return False, "no-rgb"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "no-mask"
    rgb = cv2.imread(str(img_path))
    if rgb is None:
        return False, "no-rgb"

    # Resize mask to match RGB if shapes differ (defensive — should be equal)
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    valid = mask != NO_DATA
    if not valid.any():
        return False, "no-valid-pixels"
    # Decode probability: 0..254 → 0..1
    p = mask.astype(np.float32) / 254.0
    np.clip(p, 0.0, 1.0, out=p)

    mi = alignment_mi_score(p, rgb, valid)
    meta["alignment_mi"] = mi  # may be None if MI undefined

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return True, "no-mi" if mi is None else "updated"


def process_dataset(ds_dir: Path, skip_existing: bool = False) -> dict:
    masks_dir = ds_dir / "masks"
    images_dir = ds_dir / "images"
    meta_dir = ds_dir / "meta"

    if not masks_dir.exists() or not meta_dir.exists():
        return {"skipped_dataset": str(ds_dir)}

    counts = {"updated": 0, "skipped": 0, "night": 0,
              "no-mi": 0, "no-rgb": 0, "no-mask": 0,
              "no-meta": 0, "no-valid-pixels": 0, "failed": 0}
    masks = sorted(masks_dir.glob("*.png"))
    print(f"\n{ds_dir.name}: {len(masks)} masks", flush=True)
    t0 = time.time()
    for i, mp in enumerate(masks):
        fid = mp.stem
        ip = images_dir / f"{fid}.jpg"
        mep = meta_dir / f"{fid}.json"
        ok, status = backfill_one(mp, ip, mep, skip_existing=skip_existing)
        if status in counts:
            counts[status] += 1
        else:
            counts["failed"] += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(masks):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"  {i + 1}/{len(masks)}  {rate:.0f} fps  "
                  f"updated={counts['updated']} night={counts['night']} "
                  f"skipped={counts['skipped']} no-mi={counts['no-mi']}",
                  flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dataset_v2_*")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't recompute frames that already have alignment_mi")
    args = ap.parse_args()

    ds_dirs = sorted(PROJECT_ROOT.glob(args.datasets))
    if not ds_dirs:
        print(f"No datasets match {args.datasets!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Backfilling alignment_mi across {len(ds_dirs)} dataset directories")
    if args.skip_existing:
        print("(--skip-existing: leaving already-populated values untouched)")

    grand = {"updated": 0, "skipped": 0, "night": 0, "no-mi": 0,
             "no-rgb": 0, "no-mask": 0, "no-meta": 0,
             "no-valid-pixels": 0, "failed": 0}
    for ds_dir in ds_dirs:
        if not ds_dir.is_dir():
            continue
        counts = process_dataset(ds_dir, skip_existing=args.skip_existing)
        for k, v in counts.items():
            if k in grand:
                grand[k] += v

    print("\n=== Totals ===")
    for k, v in grand.items():
        if v > 0:
            print(f"  {k:18s}: {v}")


if __name__ == "__main__":
    main()
