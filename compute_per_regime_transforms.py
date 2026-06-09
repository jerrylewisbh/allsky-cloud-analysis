"""
compute_per_regime_transforms.py — for each day's dataset_v2_<day>/, find
the best (rot, x_off, y_off) per sun regime by maximizing MI between
thermal-derived and RGB-derived cloud probability.

Pipeline:
  1. Read per-frame metadata JSON sidecars from dataset_v2_<day>/meta/
  2. For each frame, get sun_alt (from weak_labels.csv) and alignment_mi (from meta)
  3. Group frames by sun regime (DAY / TWILIGHT / NAUTICAL / ASTRO+DARK)
  4. For each regime, pick up to N "medium-MI cloud scene" frames as optimizer input
  5. Run MI maximization on each picked frame's (rot, x_off, y_off)
  6. Take the median of the optimized transforms as the regime's calibration
  7. Fall back to the closest available regime when a regime has no usable frames
  8. Write dataset_v2_<day>/transforms_by_regime.json

Usage:
  python compute_per_regime_transforms.py --datasets 'dataset_v2_*'
  python compute_per_regime_transforms.py --datasets dataset_v2_20260521
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from align_optimize import optimize_alignment
from make_masks_v2 import load_thermal
from thermal_utils import ambient_from_sensors


PROJECT_ROOT = Path(__file__).parent.resolve()
WEAK_LABELS_CSV = PROJECT_ROOT / "labels" / "weak_labels.csv"

# Picking criteria for "medium-MI cloud scene" — frames where the default
# transform has measurable but imperfect MI (genuine alignment-detectable scene)
PICK_MI_MIN = 0.10
PICK_MI_MAX = 0.50
PICK_N_PER_REGIME = 5

# Skip near-uniform thermal frames (clear / flat overcast): MI alignment needs
# cloud structure or the fit drifts. Kept-region std in degC.
MIN_STRUCT_STD = 2.0

# Regime gating (matches analyze_labels / weak_labels_reference)
REGIME_ORDER = ["DAY", "TWILIGHT", "NAUTICAL", "ASTRO_DARK"]


def _sun_regime(sun_alt_deg: float) -> str:
    """Bin sun_alt to a regime. ASTRO + DARK collapsed since both are 'no light'."""
    if sun_alt_deg >= 6:
        return "DAY"
    if sun_alt_deg >= -6:
        return "TWILIGHT"
    if sun_alt_deg >= -12:
        return "NAUTICAL"
    return "ASTRO_DARK"


def load_sun_alt_by_frame() -> dict[str, float]:
    """frame_id → sun_alt_deg from weak_labels.csv ephemeris rows."""
    out: dict[str, float] = {}
    if not WEAK_LABELS_CSV.exists():
        return out
    with open(WEAK_LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["source"] == "ephemeris" and row["attribute"] == "sun_alt_deg":
                try:
                    out[row["frame_id"]] = float(row["value"])
                except (TypeError, ValueError):
                    pass
    return out


def load_meta(meta_path: Path) -> dict | None:
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def pick_calibration_frames(ds_dir: Path, sun_alt_by_frame: dict[str, float],
                            n_per_regime: int = PICK_N_PER_REGIME) -> dict[str, list[Path]]:
    """Returns {regime: [meta_path, ...]} with up to n_per_regime frames each.

    Pick criteria:
      - alignment_mi in [PICK_MI_MIN, PICK_MI_MAX] (genuine scene with detectable features)
      - sun_alt available
      - cloud actually present (mean_cloud_prob > 0.05, or alignment_mi > 0.10)
    """
    meta_dir = ds_dir / "meta"
    if not meta_dir.exists():
        return {}

    # Bucket all usable frames by regime, keeping their MI for sorting
    candidates: dict[str, list[tuple[float, Path]]] = defaultdict(list)
    for meta_path in sorted(meta_dir.glob("*.json")):
        meta = load_meta(meta_path)
        if meta is None:
            continue
        fid = meta.get("frame_id") or meta_path.stem
        sun_alt = sun_alt_by_frame.get(fid)
        if sun_alt is None:
            continue
        mi = meta.get("alignment_mi")
        if mi is None:
            continue
        try:
            mi = float(mi)
        except (TypeError, ValueError):
            continue
        if not (PICK_MI_MIN <= mi <= PICK_MI_MAX):
            continue
        regime = _sun_regime(sun_alt)
        # Sort key: how close to mid-band (we want medium-MI for "scene has features but not yet perfectly aligned")
        sort_key = abs(mi - (PICK_MI_MIN + PICK_MI_MAX) / 2)
        candidates[regime].append((sort_key, meta_path))

    # Take up to n_per_regime closest-to-mid-MI per regime
    out: dict[str, list[Path]] = {}
    for regime, items in candidates.items():
        items.sort()
        out[regime] = [p for _, p in items[:n_per_regime]]
    return out


def _load_frame_for_optimizer(ds_dir: Path, frame_id: str) -> tuple | None:
    """Returns (thermal_raw, ambient_c, img_full) or None if anything fails."""
    # Need the ORIGINAL allsky JPG + thermal JSON — not the cropped versions
    # in dataset_v2. Read them from the source paths via the meta JSON.
    # NOTE: we don't store source paths in meta; we have to reconstruct.
    # For now, use the cropped image from dataset_v2_*/images/ as the "RGB"
    # — this is fine because the optimizer operates relative to whatever
    # the build_remap_matrices output produced, and the cropped image IS the
    # output of the same projection. The thermal JSON we need to load fresh.
    #
    # Better long-term: store the raw paths in meta. For now, this works
    # because we have the dataset_v2 outputs already.
    img_path = ds_dir / "images" / f"{frame_id}.jpg"
    if not img_path.exists():
        return None
    img_full = cv2.imread(str(img_path))
    if img_full is None:
        return None

    # Find the source thermal JSON via the dataset's day directory naming convention.
    # The thermal_root is config-dependent — we'd need it from elsewhere.
    # For Phase 2 v1, skip optimization for frames where we can't find raw thermal.
    # An alternative is to add raw_thermal_path to meta when mask-gen runs.
    raw_path = _find_raw_thermal(frame_id, ds_dir)
    if raw_path is None:
        return None
    thermal_raw, sensors = load_thermal(raw_path)
    if thermal_raw is None:
        return None
    # MI-based alignment needs cloud structure; skip near-uniform (clear / flat
    # overcast) frames where the fit is unconstrained and drifts. (corners are
    # NaN here, so use nanstd.)
    if float(np.nanstd(thermal_raw)) < MIN_STRUCT_STD:
        return None
    ambient_c = ambient_from_sensors(sensors)
    return thermal_raw, ambient_c, img_full


def _find_raw_thermal(frame_id: str, ds_dir: Path) -> Path | None:
    """Locate the raw thermal JSON for a frame. Returns None if not found.

    Tries several common patterns based on indi-allsky storage conventions.
    Override THERMAL_ROOT env var if your thermal data lives elsewhere.
    """
    import os
    thermal_root = os.environ.get(
        "THERMAL_ROOT",
        "/mnt/astro_image_thermal/ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212"
    )
    thermal_root = Path(thermal_root)
    if not thermal_root.exists():
        return None
    # frame_id format: ccd1_YYYYMMDD_HHMMSS — extract day
    parts = frame_id.split("_")
    if len(parts) < 3:
        return None
    day = parts[1]  # YYYYMMDD
    # Try patterns like ccd_*/YYYYMMDD/*.json containing the frame timestamp
    for candidate in thermal_root.rglob(f"*{frame_id.replace('ccd1_','')}*.json"):
        return candidate
    return None


def optimize_regime(ds_dir: Path, regime: str, meta_paths: list[Path],
                    static_config: dict, verbose: bool = False) -> dict | None:
    """Optimize alignment for each candidate frame, return median transform."""
    optimized: list[dict] = []
    for mp in meta_paths:
        frame_id = mp.stem
        loaded = _load_frame_for_optimizer(ds_dir, frame_id)
        if loaded is None:
            if verbose:
                print(f"    skip {frame_id}: can't load raw inputs")
            continue
        thermal_raw, ambient_c, img_full = loaded
        if verbose:
            print(f"    optimizing {frame_id} ...", end=" ", flush=True)
        result = optimize_alignment(thermal_raw, ambient_c, img_full, static_config,
                                    verbose=False)
        if result is None:
            if verbose:
                print("no improvement")
            continue
        if verbose:
            print(f"MI {result['mi_score_default']:.3f} → {result['mi_score']:.3f} "
                  f"(rot={result['rot']:+.2f}, x={result['x_off']:+.3f}, y={result['y_off']:+.3f})")
        optimized.append(result)

    if not optimized:
        return None

    # Median across optimized transforms
    return {
        "regime": regime,
        "rot": statistics.median(o["rot"] for o in optimized),
        "x_off": statistics.median(o["x_off"] for o in optimized),
        "y_off": statistics.median(o["y_off"] for o in optimized),
        "mi_score_median": statistics.median(o["mi_score"] for o in optimized),
        "mi_score_default_median": statistics.median(o["mi_score_default"] for o in optimized),
        "n_frames_optimized": len(optimized),
    }


def apply_fallbacks(transforms: dict[str, dict], static_config: dict) -> dict[str, dict]:
    """Fill missing regimes from closest-available regime (or static if none)."""
    static_entry = {
        "rot": static_config["rot"],
        "x_off": static_config["x_off"],
        "y_off": static_config["y_off"],
        "mi_score_median": None,
        "mi_score_default_median": None,
        "n_frames_optimized": 0,
        "from_fallback": "static",
    }
    out: dict[str, dict] = {}
    for regime in REGIME_ORDER:
        if regime in transforms:
            out[regime] = transforms[regime]
            continue
        # Find closest available regime in REGIME_ORDER
        idx = REGIME_ORDER.index(regime)
        chosen = None
        for offset in range(1, len(REGIME_ORDER)):
            for direction in (-1, +1):
                j = idx + direction * offset
                if 0 <= j < len(REGIME_ORDER) and REGIME_ORDER[j] in transforms:
                    chosen = REGIME_ORDER[j]
                    break
            if chosen:
                break
        if chosen:
            src = transforms[chosen]
            out[regime] = {**src, "from_fallback": chosen}
        else:
            out[regime] = {**static_entry, "regime": regime}
    return out


def load_static_config(config_path: Path | None = None) -> dict:
    """Load the static calibration from alignment_config.json — the SAME file
    mask-gen reads. Must match the calibration used to project the masks we're
    optimizing relative to, otherwise the optimizer starts from a wrong baseline
    and may not find the local optimum within its bounds.

    Path resolution:
      1. --config CLI arg if provided
      2. PROJECT_ROOT / 'alignment_config.json'
      3. fall back to firmware defaults (likely wrong)
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(PROJECT_ROOT / "alignment_config.json")
    for p in candidates:
        if p.exists():
            with open(p) as f:
                cfg = json.load(f)
            print(f"Loaded static config from {p}: {cfg}")
            # Fill any missing keys with safe defaults
            cfg.setdefault("fov", 60.0)
            cfg.setdefault("rot", 0.0)
            cfg.setdefault("x_off", 0.0)
            cfg.setdefault("y_off", 0.0)
            cfg.setdefault("dist", 0.0)
            cfg.setdefault("proj_on", 1)
            cfg.setdefault("flip_h", 0)
            cfg.setdefault("flip_v", 1)
            return cfg
    print(f"WARNING: no alignment_config.json found — using firmware defaults. "
          "Optimizer will start from a wrong baseline.")
    return {"fov": 60.0, "rot": 0.0, "x_off": 0.0, "y_off": 0.0,
            "dist": 0.0, "proj_on": 1, "flip_h": 0, "flip_v": 1}


def process_day(ds_dir: Path, static_config: dict, verbose: bool = False) -> dict | None:
    """Process one day's dataset directory. Returns transforms_by_regime dict
    (or None if nothing usable). Writes transforms_by_regime.json to disk."""
    print(f"\n=== {ds_dir.name} ===")
    sun_alt_by_frame = load_sun_alt_by_frame()
    if not sun_alt_by_frame:
        print("  no sun_alt data in weak_labels.csv — skipping")
        return None

    picked = pick_calibration_frames(ds_dir, sun_alt_by_frame)
    if not picked:
        print("  no medium-MI candidate frames available — skipping")
        return None
    for regime, frames in picked.items():
        print(f"  {regime}: {len(frames)} candidate frames")

    raw: dict[str, dict] = {}
    for regime in REGIME_ORDER:
        if regime not in picked:
            continue
        print(f"  Optimizing {regime}...")
        result = optimize_regime(ds_dir, regime, picked[regime], static_config,
                                 verbose=verbose)
        if result is None:
            print(f"    {regime}: optimization failed on all picked frames")
            continue
        raw[regime] = result
        print(f"    {regime}: median rot={result['rot']:+.2f}°, "
              f"x={result['x_off']:+.3f}, y={result['y_off']:+.3f}, "
              f"MI {result['mi_score_default_median']:.3f} → {result['mi_score_median']:.3f} "
              f"(n={result['n_frames_optimized']})")

    final = apply_fallbacks(raw, static_config)
    out_path = ds_dir / "transforms_by_regime.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"  → {out_path.name}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dataset_v2_*",
                    help="Glob pattern for dataset directories")
    ap.add_argument("--config", default=None,
                    help="Path to alignment_config.json (default: project-root file)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print per-frame optimizer progress")
    args = ap.parse_args()

    static_config = load_static_config(args.config)
    ds_dirs = sorted(PROJECT_ROOT.glob(args.datasets))
    if not ds_dirs:
        print(f"No dataset directories match {args.datasets!r}", file=sys.stderr)
        sys.exit(1)
    print(f"Processing {len(ds_dirs)} day(s) with static config {static_config}")

    for ds_dir in ds_dirs:
        if not ds_dir.is_dir():
            continue
        process_day(ds_dir, static_config, verbose=args.verbose)


if __name__ == "__main__":
    main()
