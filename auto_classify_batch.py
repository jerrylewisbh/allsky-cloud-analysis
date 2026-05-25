"""
auto_classify_batch.py — Run auto_classify across every frame in dataset_v2_*,
write labels/auto_labels.csv, and report distribution + (when hand labels exist)
agreement with humans.

This produces the paper's headline number: "% of frames the multimodal weak
labels can pre-classify without human input, and how often the verdict matches
the human label."

Run:
  python auto_classify_batch.py
  python auto_classify_batch.py --datasets dataset_v2_*
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from auto_classify import classify

PROJECT_ROOT = Path(__file__).parent.resolve()
WEAK_LABELS_CSV = PROJECT_ROOT / "labels" / "weak_labels.csv"
HAND_LABELS_CSV = PROJECT_ROOT / "labels" / "hand_labeled.csv"
AUTO_LABELS_CSV = PROJECT_ROOT / "labels" / "auto_labels.csv"


def thermal_mean(mask_path: Path) -> float | None:
    """Mean cloud probability over valid pixels. Backward-compatible — kept
    as a thin wrapper around thermal_stats() so older callers still work."""
    m, _ = thermal_stats(mask_path)
    return m


def thermal_stats(mask_path: Path) -> tuple[float | None, float | None]:
    """Returns (mean, std) of cloud probability over valid pixels.

    The std measures spatial variance — a textbook discriminator between:
      - uniform sky (clear or overcast Sc/St): low std
      - convective / broken cloud (Cu, broken Ac, fragmented Sc): high std

    This is the single most useful feature for separating Cu from clear,
    which the per-frame vote cascade cannot do (it operates on the scalar
    mean only). Computing std is essentially free since the mask is already
    loaded for the mean.
    """
    raw = np.array(Image.open(mask_path).convert("L"))
    valid = raw != 255
    if not valid.any():
        return None, None
    probs = raw[valid].astype(np.float32) / 254.0
    return float(probs.mean()), float(probs.std())


def _rgb_and_valid(rgb_path: Path, mask_path: Path):
    """Shared helper: load RGB + valid-region mask, aligned to RGB shape."""
    try:
        rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
    except (FileNotFoundError, OSError):
        return None, None
    mask = np.array(Image.open(mask_path).convert("L"))
    valid = mask != 255
    if not valid.any():
        return rgb, None
    if mask.shape != rgb.shape[:2]:
        from PIL import Image as _PILImage
        valid = np.array(
            _PILImage.fromarray(valid.astype(np.uint8) * 255)
            .resize((rgb.shape[1], rgb.shape[0]), _PILImage.NEAREST)
        ) > 127
    return rgb, valid


def rgb_nrbr_p95_in_valid_region(rgb_path: Path, mask_path: Path) -> float | None:
    """95th percentile of (R-B)/(R+B) over the thermal-valid pixels of the RGB crop."""
    rgb, valid = _rgb_and_valid(rgb_path, mask_path)
    if rgb is None or valid is None:
        return None
    r = rgb[..., 0][valid]
    b = rgb[..., 2][valid]
    nrbr = (r - b) / (r + b + 1e-6)
    return float(np.percentile(nrbr, 95))


def rgb_v_stats_in_valid_region(rgb_path: Path, mask_path: Path) -> tuple[float | None, float | None]:
    """Mean and STD of HSV V (brightness, 0–255) over the thermal-valid pixels.
    Used as a nighttime cloud-presence vote (skyglow reflection + texture)."""
    rgb, valid = _rgb_and_valid(rgb_path, mask_path)
    if rgb is None or valid is None:
        return None, None
    v_channel = rgb.max(axis=-1)[valid]
    return float(v_channel.mean()), float(v_channel.std())


def discover_frames(datasets_glob: str) -> list[tuple[str, str, str, str]]:
    """Returns list of (frame_id, mask_path, rgb_path, meta_path)."""
    out = []
    for ds in sorted(PROJECT_ROOT.glob(datasets_glob)):
        img_dir = ds / "images"
        meta_dir = ds / "meta"
        for p in sorted((ds / "masks").glob("*.png")):
            rgb_path = img_dir / f"{p.stem}.jpg"
            meta_path = meta_dir / f"{p.stem}.json"
            out.append((p.stem, str(p), str(rgb_path), str(meta_path)))
    return out


def read_alignment_mi(meta_path: Path) -> float | None:
    """Pull alignment_mi from the per-frame sidecar JSON written by make_masks_v2."""
    try:
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        v = meta.get("alignment_mi")
        return float(v) if v is not None else None
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def load_weak_labels() -> dict[str, dict[tuple, dict]]:
    if not WEAK_LABELS_CSV.exists():
        return {}
    by_frame: dict[str, dict[tuple, dict]] = {}
    with open(WEAK_LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            by_frame.setdefault(row["frame_id"], {})[(row["source"], row["attribute"])] = row
    return by_frame


def load_hand_labels() -> dict[str, dict]:
    if not HAND_LABELS_CSV.exists():
        return {}
    with open(HAND_LABELS_CSV, newline="") as f:
        return {r["frame_id"]: r for r in csv.DictReader(f)}


def reclassify_frame(frame_id: str, mask_path: str, rgb_path: str,
                     meta_path: str | None = None) -> dict | None:
    """Recompute the auto-label row for a single frame and update auto_labels.csv
    in place. Returns the new row dict, or None if AUTO_LABELS_CSV doesn't
    exist yet.

    Used by the labeling UI's mask-cleanup panel: after a labeler saves a
    cleaned mask, this is called so the next visit to the frame sees the
    corrected verdict.
    """
    if not AUTO_LABELS_CSV.exists():
        return None

    weak = load_weak_labels()
    mp, mstd = thermal_stats(Path(mask_path))
    nrbr_p95 = rgb_nrbr_p95_in_valid_region(Path(rgb_path), Path(mask_path))
    v_mean, v_std = rgb_v_stats_in_valid_region(Path(rgb_path), Path(mask_path))
    align_mi = read_alignment_mi(Path(meta_path)) if meta_path else None
    wf = weak.get(frame_id, {})
    cls, conf, reasoning = classify(
        wf, thermal_mean_p=mp,
        rgb_nrbr_p95=nrbr_p95,
        rgb_v_mean=v_mean,
        rgb_v_std=v_std,
        thermal_std=mstd,
    )
    new_row = {
        "frame_id": frame_id,
        "auto_class": cls,
        "auto_confidence": conf,
        "auto_reasoning": reasoning,
        "thermal_mean_p": f"{mp:.3f}" if mp is not None else "",
        "thermal_std": f"{mstd:.3f}" if mstd is not None else "",
        "rgb_nrbr_p95": f"{nrbr_p95:+.3f}" if nrbr_p95 is not None else "",
        "rgb_v_mean": f"{v_mean:.1f}" if v_mean is not None else "",
        "rgb_v_std": f"{v_std:.1f}" if v_std is not None else "",
        "alignment_mi": f"{align_mi:.3f}" if align_mi is not None else "",
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    # Read full CSV, replace (or append) the target row, write back.
    # Safe even if the file is large — auto_labels.csv is <100k rows.
    with open(AUTO_LABELS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or list(new_row.keys())
        rows = list(reader)
    found = False
    for i, r in enumerate(rows):
        if r["frame_id"] == frame_id:
            rows[i] = {k: new_row.get(k, "") for k in fieldnames}
            found = True
            break
    if not found:
        rows.append({k: new_row.get(k, "") for k in fieldnames})
    with open(AUTO_LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return new_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dataset_v2_*")
    args = ap.parse_args()

    weak = load_weak_labels()
    hand = load_hand_labels()
    frames = discover_frames(args.datasets)
    print(f"Frames: {len(frames)}  ·  weak-label coverage: {len(weak)}  ·  hand labels: {len(hand)}")

    rows = []
    dist = Counter()
    conf_dist = Counter()
    for i, (fid, mask_path, rgb_path, meta_path) in enumerate(frames):
        mp, mstd = thermal_stats(Path(mask_path))
        nrbr_p95 = rgb_nrbr_p95_in_valid_region(Path(rgb_path), Path(mask_path))
        v_mean, v_std = rgb_v_stats_in_valid_region(Path(rgb_path), Path(mask_path))
        align_mi = read_alignment_mi(Path(meta_path))
        wf = weak.get(fid, {})
        cls, conf, reasoning = classify(wf, thermal_mean_p=mp,
                                         rgb_nrbr_p95=nrbr_p95,
                                         rgb_v_mean=v_mean,
                                         rgb_v_std=v_std,
                                         thermal_std=mstd)
        rows.append({
            "frame_id": fid,
            "auto_class": cls,
            "auto_confidence": conf,
            "auto_reasoning": reasoning,
            "thermal_mean_p": f"{mp:.3f}" if mp is not None else "",
            "thermal_std": f"{mstd:.3f}" if mstd is not None else "",
            "rgb_nrbr_p95": f"{nrbr_p95:+.3f}" if nrbr_p95 is not None else "",
            "rgb_v_mean": f"{v_mean:.1f}" if v_mean is not None else "",
            "rgb_v_std": f"{v_std:.1f}" if v_std is not None else "",
            "alignment_mi": f"{align_mi:.3f}" if align_mi is not None else "",
            "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        })
        dist[cls] += 1
        conf_dist[conf] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(frames)} classified")

    AUTO_LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTO_LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"Wrote {len(rows)} rows to {AUTO_LABELS_CSV}")

    print("\nAuto-label class distribution:")
    for c, n in dist.most_common():
        bar = "█" * (n * 50 // max(dist.values()))
        print(f"  {c:8s} {n:5d}  {bar}")

    print("\nAuto-label confidence distribution:")
    for c in ["high", "medium", "low"]:
        n = conf_dist.get(c, 0)
        pct = 100 * n / max(len(rows), 1)
        bar = "█" * (n * 50 // max(conf_dist.values()))
        print(f"  {c:8s} {n:5d} ({pct:5.1f}%)  {bar}")

    # Manual-review budget estimate
    needs_review = conf_dist.get("low", 0) + conf_dist.get("medium", 0)
    print(f"\nManual-review budget (low + medium): {needs_review} frames "
          f"({100 * needs_review / len(rows):.1f}%)")
    print(f"High-confidence auto-accepted: {conf_dist.get('high', 0)} frames "
          f"({100 * conf_dist.get('high', 0) / len(rows):.1f}%)")

    # Agreement vs hand labels if we have any
    if hand:
        print("\n--- Agreement vs hand labels ---")
        matched_rows = [(r, hand[r["frame_id"]]) for r in rows if r["frame_id"] in hand]
        if not matched_rows:
            print("  No overlap between auto_labels and hand_labeled.")
            return
        n = len(matched_rows)
        n_exact = sum(1 for ar, hr in matched_rows if ar["auto_class"] == hr["class"])
        print(f"  Frames with both auto + hand label: {n}")
        print(f"  Exact-match agreement: {n_exact}/{n} ({100 * n_exact / n:.1f}%)")

        # Per-confidence agreement (this is the key paper number)
        by_conf = defaultdict(lambda: [0, 0])
        for ar, hr in matched_rows:
            by_conf[ar["auto_confidence"]][1] += 1
            if ar["auto_class"] == hr["class"]:
                by_conf[ar["auto_confidence"]][0] += 1
        print()
        print("  Per-confidence agreement:")
        for c in ["high", "medium", "low"]:
            hit, total = by_conf.get(c, [0, 0])
            if total:
                print(f"    {c:6s}: {hit}/{total} = {100 * hit / total:.1f}%")

        # Per-class confusion (auto → hand)
        print()
        print("  Confusion (auto-label → hand-label, top 10 cells):")
        confusion = Counter((ar["auto_class"], hr["class"]) for ar, hr in matched_rows)
        for (a, h), n in confusion.most_common(10):
            mark = "✓" if a == h else " "
            print(f"    {mark} auto={a:8s} hand={h:8s}  {n}")


if __name__ == "__main__":
    main()
