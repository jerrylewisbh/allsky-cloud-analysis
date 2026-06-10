"""
make_masks_v2.py — physics-grounded soft cloud-probability masks.

Replaces the binary/degenerate masks produced by matched_crop.py. Output is a
continuous per-pixel cloud probability (0..254) with 255 reserved for no-data
(outside the thermal sensor FOV, outside the fisheye disc, or sun-disc artifact).

Pipeline per frame:
  1. Load raw 32x24 MLX90640 thermal frame from JSON sidecar; load ambient + lux.
  2. Build fisheye->thermal remap (reuses matched_crop alignment math + config).
  3. Warp thermal to allsky coords using BORDER_CONSTANT (not REPLICATE).
  4. Per-pixel thermal cloud probability via two sigmoids (mirrors firmware):
        p_abs = sigmoid((T - ABS_THRESHOLD) / sigma)        # absolute warm-air rule
        p_rel = sigmoid((T - (ambient + REL_DELTA)) / sigma)# ambient-relative rule
        p_thermal = max(p_abs, p_rel)
  5. Daytime only (lux > LUX_DAY): compute HYTA-style NRBR cloud confidence on
     the RGB crop, refine the thermal prior via guided filter (RGB as guide).
  6. Twilight (LUX_NIGHT < lux < LUX_DAY): blend thermal-only and fused outputs.
  7. Mark no-data: outside thermal valid mask, outside fisheye disc.
  8. Save 256x256 cropped RGB + 256x256 mask (0..254 prob, 255 no-data) + meta JSON.

Usage:
  python make_masks_v2.py --day 20260518
  python make_masks_v2.py --day 20260518 --max-pairs 10 --output-root dataset_v2_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from thermal_utils import reshape_thermal, apply_corner_mask, ambient_from_sensors

# ---- physics thresholds (mirror docs/sky-condition.md and firmware) ----
# Overridable via env vars so the threshold choice can be A/B-tested without a
# code edit, e.g. to reproduce the pre-2026-05-24 values:
#   THERMAL_ABS_C=-18 THERMAL_REL_C=-20 THERMAL_SIGMA_C=6 python make_masks_v2.py ...
#
# RECALIBRATED 2026-06-09 for the ZnSe window (see calibrate_thermal.py). The
# warm window compresses clear sky to ~-9..-3.5C, so the previous ABS=-3 / SIGMA=3
# soft sigmoid sat *inside* the clear-sky band and scored clear sky ~0.18 mean
# (saturating the whole 0..1 range). ABS=-1 / SIGMA=1 drives clear sky to ~0.01
# mean while overcast stays ~1.0 (verified: firmware-clear frames median 0.006,
# the 013646 clear frame 0.184 -> 0.005).
#   NB: this offline ABS (-1) is INTENTIONALLY warmer than the firmware's hard
#   cutoff (-3 in sky_thermal.h). The firmware COUNTS pixels above the cutoff;
#   we take a soft-sigmoid MEAN over the pixel spread, so our center must sit
#   ~2C higher to push the same clear distribution to ~0. The divergence is
#   expected — do not "fix" it to match the firmware.
ABS_THRESHOLD_C = float(os.environ.get("THERMAL_ABS_C", -1.0))   # warmer than this absolute => cloud  (ZnSe soft-sigmoid; firmware hard cutoff is -3)
REL_DELTA_C = float(os.environ.get("THERMAL_REL_C", -12.0))      # warmer than (ambient + REL_DELTA) => cloud  (was -20, was -10 orig)
SIGMOID_SIGMA_C = float(os.environ.get("THERMAL_SIGMA_C", 1.0))  # softness of the cloud transition (°C)  (was 3, mis-centred for ZnSe)

# ---- day / night handling (lux from sensors block) ----
# lux now comes from the weather station's solar irradiance (W/m^2 * 126 lm/W),
# i.e. a real-world daylight scale. Breakpoints in solar-irradiance terms:
#   LUX_NIGHT 1000 lx ~= 8 W/m^2  (sun below ~-4 deg: too dark for RGB -> thermal only)
#   LUX_DAY  10000 lx ~= 80 W/m^2 (solid daylight -> full RGB refinement)
# Between them = twilight blend. Refine empirically from observed dawn/dusk lux.
LUX_NIGHT = 1000.0          # below: thermal-only
LUX_DAY = 10000.0           # above: full RGB refinement
HYTA_NRBR_CLEAR = -0.35      # NRBR <= -0.35 => clear blue sky
HYTA_NRBR_CLOUD = +0.05      # NRBR >= +0.05 => fully white cloud

# ---- spatial / output ----
GUIDED_RADIUS = 16
GUIDED_EPS = 1e-3
OUTPUT_SIZE = 256
NO_DATA = 255


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def alignment_mi_score(thermal_prob: np.ndarray, rgb_crop: np.ndarray,
                       valid: np.ndarray, n_bins: int = 10) -> float | None:
    """Mutual information between thermal-derived cloud probability and an
    RGB-derived cloud probability (NRBR) over thermal-valid pixels.

    High MI = thermal and RGB agree on WHERE cloud is, i.e. alignment is good.
    Low MI = thermal cloud peaks fall on RGB-clear pixels (and vice versa),
    i.e. the projection is misaligned (typically by 1-2 px due to wind drift,
    thermal expansion of mount, etc.).

    Daytime-only — at night the NRBR signal is meaningless (long-exposure
    near-monochrome). Returns None when not enough valid pixels or RGB
    range is too narrow to compute MI meaningfully.

    Range: ~0 (random) to ~ln(n_bins) ≈ 2.3 with 10 bins (perfect agreement).
    Typical well-aligned daytime cloud scene: 0.3-0.6.
    """
    if not valid.any():
        return None
    valid_count = int(valid.sum())
    if valid_count < 200:
        return None  # not enough samples for meaningful MI

    # RGB cloud proxy: NRBR mapped to [0, 1]. Clouds tend to have low NRBR
    # (whiter); clear sky has high NRBR (blue-dominant when negated to match
    # the cloud=high convention used by thermal_prob).
    b, g, r = cv2.split(rgb_crop.astype(np.float32))
    nrbr = (r - b) / (r + b + 1e-6)  # [-1, 1]; negative = blue/clear, near 0 = whitish/cloud
    rgb_cloud = np.clip(1.0 - (nrbr + 1.0) / 2.0, 0.0, 1.0)  # invert + normalize → cloud=high

    t_vals = thermal_prob[valid]
    r_vals = rgb_cloud[valid]

    # If either signal has near-zero variance, MI is undefined / meaningless
    if t_vals.std() < 0.02 or r_vals.std() < 0.02:
        return None

    # Bin both signals to discrete categories for MI computation
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    t_bin = np.digitize(t_vals, bins)
    r_bin = np.digitize(r_vals, bins)

    # Joint and marginal histograms
    joint = np.zeros((n_bins + 2, n_bins + 2), dtype=np.float64)
    np.add.at(joint, (t_bin, r_bin), 1.0)
    joint /= joint.sum()
    p_t = joint.sum(axis=1, keepdims=True)
    p_r = joint.sum(axis=0, keepdims=True)
    pp = p_t @ p_r
    nz = (joint > 0) & (pp > 0)
    return float((joint[nz] * np.log(joint[nz] / pp[nz])).sum())


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. 2010 guided filter, single-channel guide. Pure numpy."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    mean_Ip = cv2.boxFilter(guide * src, -1, k)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, -1, k)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, k)
    mean_b = cv2.boxFilter(b, -1, k)
    return mean_a * guide + mean_b


def build_remap_matrices(allsky_w, allsky_h, thermal_w, thermal_h,
                         allsky_fov_deg, thermal_fov_deg, rotation_deg,
                         offset_x_pct, offset_y_pct, distortion, proj_on):
    """Same equations as matched_crop.build_remap_matrices — kept here so this
    file is standalone. See ALGORITHMS.md."""
    cx_a = (allsky_w / 2.0) + (offset_x_pct * (allsky_w / 2.0))
    cy_a = (allsky_h / 2.0) + (offset_y_pct * (allsky_h / 2.0))
    R_a = allsky_w / 2.0
    max_theta_a = np.radians(allsky_fov_deg / 2.0)
    thermal_fov_rad = np.radians(thermal_fov_deg)
    f_t = (thermal_w / 2.0) / np.tan(thermal_fov_rad / 2.0)
    cx_t, cy_t = thermal_w / 2.0, thermal_h / 2.0
    X, Y = np.meshgrid(np.arange(allsky_w), np.arange(allsky_h))
    dx, dy = X - cx_a, Y - cy_a
    r = np.sqrt(dx * dx + dy * dy)
    phi = np.arctan2(dy, dx) + np.radians(rotation_deg)
    if not proj_on:
        scale = (thermal_fov_deg / allsky_fov_deg) * allsky_w
        s_factor = thermal_w / scale
        dx_rot, dy_rot = r * np.cos(phi), r * np.sin(phi)
        map_x, map_y = cx_t + dx_rot * s_factor, cy_t + dy_rot * s_factor
        invalid = (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
        map_x[invalid] = -1; map_y[invalid] = -1
        return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid
    r_norm = r / R_a
    theta = (r_norm ** (2.0 ** distortion)) * max_theta_a
    valid_theta = theta < (np.pi / 2 - 0.01)
    d_t = np.zeros_like(theta)
    d_t[valid_theta] = f_t * np.tan(theta[valid_theta])
    map_x = cx_t + d_t * np.cos(phi)
    map_y = cy_t + d_t * np.sin(phi)
    invalid = ~valid_theta | (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
    map_x[invalid] = -1; map_y[invalid] = -1
    return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid


def load_thermal(json_path: Path) -> tuple[np.ndarray, dict] | tuple[None, None]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None
    frame = data.get("frame")
    if not frame:
        return None, None
    try:
        arr, _ = reshape_thermal(frame)
    except ValueError:
        return None, None
    # NaN the clipped corners; the warp + np.isnan() handling below excludes them.
    arr = apply_corner_mask(arr)
    return arr, data.get("sensors", {}) or {}


@dataclass
class FramePaths:
    frame_id: str
    rgb_path: Path
    json_path: Path


def discover_pairs(allsky_root: Path, thermal_root: Path, day: str) -> Iterator[FramePaths]:
    """Walk allsky/images/<day>/{day,night}/HH_MM/ tree, yield pairs with matching thermal."""
    day_root = allsky_root / "images" / day
    if not day_root.is_dir():
        return
    for jpg in sorted(day_root.rglob("*.jpg")):
        if "thumbnails" in jpg.parts:
            continue
        rel_from_images = jpg.relative_to(allsky_root / "images")
        json_path = thermal_root / "exposures" / rel_from_images.with_suffix(".json")
        if json_path.exists():
            yield FramePaths(frame_id=jpg.stem, rgb_path=jpg, json_path=json_path)


def process_frame(fp: FramePaths, config: dict) -> dict | None:
    img_full = cv2.imread(str(fp.rgb_path))
    if img_full is None:
        return None
    thermal_raw, sensors = load_thermal(fp.json_path)
    if thermal_raw is None:
        return None

    # Match firmware flips
    flip_h = bool(config.get("flip_h", 0))
    flip_v = bool(config.get("flip_v", 1))
    if flip_h and flip_v:
        thermal_raw = np.flip(thermal_raw, (0, 1))
    elif flip_h:
        thermal_raw = np.flip(thermal_raw, 1)

    # Sensor corner contamination mask — the MLX's bottom-right FOV edge
    # picks up a permanent warm gradient (horizon/housing/dust). Mark those
    # pixels NaN so they're excluded from cloud-probability calculations.
    # Configurable per deployment via config["corner_mask_rows"] /
    # config["corner_mask_cols"]. Set to 0 to disable.
    corner_rows = int(config.get("corner_mask_rows", 0))
    corner_cols = int(config.get("corner_mask_cols", 0))
    if corner_rows > 0 and corner_cols > 0:
        # The flips above mean the "physical bottom-right" of the sensor maps
        # to a specific corner of the post-flip array. With flip_v=1 (firmware
        # default), the physical bottom is now the top. The contaminated
        # physical-bottom-right shows up as TOP-right in the array.
        if flip_v and not flip_h:
            thermal_raw[:corner_rows, -corner_cols:] = np.nan
        elif flip_h and not flip_v:
            thermal_raw[-corner_rows:, :corner_cols] = np.nan
        elif flip_h and flip_v:
            thermal_raw[-corner_rows:, -corner_cols:] = np.nan
        else:
            thermal_raw[-corner_rows:, -corner_cols:] = np.nan
    elif flip_v:
        thermal_raw = np.flip(thermal_raw, 0)

    a_h, a_w = img_full.shape[:2]
    t_h, t_w = thermal_raw.shape
    map_x, map_y, valid_full = build_remap_matrices(
        a_w, a_h, t_w, t_h,
        180.0, config["fov"], config["rot"],
        config["x_off"], config["y_off"], config["dist"],
        config.get("proj_on", 1),
    )

    coords = np.argwhere(valid_full)
    if len(coords) == 0:
        return None
    pad = 30
    y0 = max(0, int(coords[:, 0].min()) - pad)
    x0 = max(0, int(coords[:, 1].min()) - pad)
    y1 = min(a_h, int(coords[:, 0].max()) + pad + 1)
    x1 = min(a_w, int(coords[:, 1].max()) + pad + 1)
    img_crop = img_full[y0:y1, x0:x1]
    map_x_c = map_x[y0:y1, x0:x1]
    map_y_c = map_y[y0:y1, x0:x1]
    valid_c = valid_full[y0:y1, x0:x1]

    # Bilinear remap of the thermal frame plus a co-warped validity field.
    # Bilinear interpolation along the FOV edge would otherwise mix real
    # thermal values with the sentinel and produce a "black ring" of fake
    # ultra-cold pixels. Co-warping a unit field catches every pixel that
    # the interpolator touched the boundary on.
    warped_thermal = cv2.remap(
        thermal_raw, map_x_c, map_y_c,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    valid_field = np.ones_like(thermal_raw, dtype=np.float32)
    warped_valid = cv2.remap(
        valid_field, map_x_c, map_y_c,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    # 0.999 threshold = "only pixels whose bilinear footprint was fully
    # inside the sensor". Erodes the valid region by ~1 px — exactly the
    # contamination width.
    invalid_thermal = (~valid_c) | (warped_valid < 0.999)
    # Corner mask: if sensor corner pixels were NaN'd above, any warped pixel
    # whose bilinear sample included those NaNs will itself be NaN. Mark them
    # invalid too so they're excluded from mean/std stats.
    invalid_thermal = invalid_thermal | np.isnan(warped_thermal)

    # ---- thermal cloud probability (per-pixel firmware physics) ----
    ambient = ambient_from_sensors(sensors)
    p_abs = sigmoid((warped_thermal - ABS_THRESHOLD_C) / SIGMOID_SIGMA_C)
    p_rel = sigmoid((warped_thermal - (ambient + REL_DELTA_C)) / SIGMOID_SIGMA_C)
    p_thermal = np.maximum(p_abs, p_rel)

    # ---- daytime RGB refinement ----
    lux = float(sensors.get("lux", 0.0) or 0.0)
    if lux >= LUX_NIGHT:
        b, g, r = cv2.split(img_crop.astype(np.float32))
        nrbr = (r - b) / (r + b + 1e-6)
        p_rgb = np.clip(
            (nrbr - HYTA_NRBR_CLEAR) / (HYTA_NRBR_CLOUD - HYTA_NRBR_CLEAR), 0.0, 1.0
        )

        # Guided-filter refinement: use grayscale RGB as the guide so cloud
        # edges in RGB transfer onto the low-res thermal probability.
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # The guided filter is box-convolution based, so ANY NaN in p_thermal
        # (corner mask + eroded FOV edge) spreads across the full filter radius
        # and — after two box passes — floods the entire crop, poisoning the
        # refined map to all-NaN. That blanked every daytime frame to an empty
        # mask. Fill invalid pixels with the valid-region mean before filtering;
        # the no-data mask below (invalid_thermal) restores them afterwards.
        finite = p_thermal[np.isfinite(p_thermal)]
        fill_val = float(finite.mean()) if finite.size else 0.0
        p_thermal_fill = np.where(np.isnan(p_thermal), fill_val, p_thermal)
        p_thermal_refined = guided_filter(gray, p_thermal_fill, GUIDED_RADIUS, GUIDED_EPS)
        p_thermal_refined = np.clip(p_thermal_refined, 0.0, 1.0)

        # Day blend: thermal as anchor (0.6), RGB (0.4) — RGB only refines, never overrides
        day_strength = float(np.clip((lux - LUX_NIGHT) / (LUX_DAY - LUX_NIGHT), 0.0, 1.0))
        p_day = 0.6 * p_thermal_refined + 0.4 * p_rgb
        p_combined = (1.0 - day_strength) * p_thermal + day_strength * p_day
    else:
        p_combined = p_thermal

    # No-data masking
    p_combined = np.where(invalid_thermal, np.nan, p_combined)

    # Encode mask: 0..254 = probability * 254, 255 = no-data
    mask_uint8 = np.full_like(p_combined, NO_DATA, dtype=np.uint8)
    valid_p = ~np.isnan(p_combined)
    mask_uint8[valid_p] = np.round(np.clip(p_combined[valid_p], 0.0, 1.0) * 254.0).astype(np.uint8)

    # Resize both to OUTPUT_SIZE x OUTPUT_SIZE
    img_out = cv2.resize(img_crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    mask_out = cv2.resize(mask_uint8, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)

    # Stats for sidecar JSON
    valid_pixels = mask_out != NO_DATA
    valid_count = int(valid_pixels.sum())
    if valid_count > 0:
        probs = mask_out[valid_pixels].astype(np.float32) / 254.0
        mean_prob = float(probs.mean())
        cloud_fraction = float((probs >= 0.5).mean())
    else:
        mean_prob = float("nan")
        cloud_fraction = float("nan")

    # Alignment quality (MI between thermal cloud-prob and RGB cloud-prob).
    # Computed on the pre-resize arrays where both are at full crop resolution.
    # Daytime only — at night NRBR is meaningless.
    if lux >= LUX_NIGHT:
        valid_mi_mask = ~invalid_thermal
        mi_score = alignment_mi_score(p_combined, img_crop, valid_mi_mask)
    else:
        mi_score = None

    return {
        "img": img_out,
        "mask": mask_out,
        "meta": {
            "frame_id": fp.frame_id,
            "ambient_c": ambient,
            "lux": lux,
            "thermal_min": float(np.nanmin(np.where(invalid_thermal, np.nan, warped_thermal))),
            "thermal_max": float(np.nanmax(np.where(invalid_thermal, np.nan, warped_thermal))),
            "mean_cloud_prob": mean_prob,
            "cloud_fraction_p50": cloud_fraction,
            "valid_fraction": valid_count / (OUTPUT_SIZE * OUTPUT_SIZE),
            "alignment_mi": mi_score,  # None if night or unmeasurable
            "fw_sky_condition": sensors.get("sky_condition"),
            "fw_sky_cloud_fraction": sensors.get("sky_cloud_fraction"),
            "fw_sky_abs_cloud_fraction": sensors.get("sky_abs_cloud_fraction"),
        },
    }


def _sun_regime(sun_alt_deg: float) -> str:
    """DAY / TWILIGHT / NAUTICAL / ASTRO_DARK — matches compute_per_regime_transforms."""
    if sun_alt_deg >= 6:
        return "DAY"
    if sun_alt_deg >= -6:
        return "TWILIGHT"
    if sun_alt_deg >= -12:
        return "NAUTICAL"
    return "ASTRO_DARK"


def _config_for_frame(base_config: dict, frame_id: str,
                      sun_alt_by_frame: dict[str, float] | None,
                      transforms_by_regime: dict[str, dict] | None) -> tuple[dict, str | None]:
    """Returns (config, regime_used) for one frame. When per-regime data isn't
    available the base config is returned unchanged with regime_used=None.

    Per-regime transforms override only (rot, x_off, y_off). The lens params
    (fov, dist) stay at the static calibration."""
    if not transforms_by_regime or sun_alt_by_frame is None:
        return base_config, None
    sun_alt = sun_alt_by_frame.get(frame_id)
    if sun_alt is None:
        return base_config, None
    regime = _sun_regime(sun_alt)
    override = transforms_by_regime.get(regime)
    if override is None:
        return base_config, None
    cfg = dict(base_config)  # shallow copy is enough — we only overwrite scalars
    cfg["rot"] = override["rot"]
    cfg["x_off"] = override["x_off"]
    cfg["y_off"] = override["y_off"]
    return cfg, regime


def worker(fp: FramePaths, base_config: dict, out_root: Path,
           sun_alt_by_frame: dict[str, float] | None = None,
           transforms_by_regime: dict[str, dict] | None = None):
    try:
        cfg, regime = _config_for_frame(base_config, fp.frame_id,
                                        sun_alt_by_frame, transforms_by_regime)
        result = process_frame(fp, cfg)
        if result is None:
            return fp.frame_id, False, "Skipped"

        # Tag the meta with which regime + transform variant was applied
        if regime is not None:
            result["meta"]["regime"] = regime
            result["meta"]["alignment_variant"] = "per_regime"
        else:
            result["meta"]["alignment_variant"] = "static"

        cv2.imwrite(str(out_root / "images" / f"{fp.frame_id}.jpg"), result["img"])
        cv2.imwrite(str(out_root / "masks" / f"{fp.frame_id}.png"), result["mask"])
        with open(out_root / "meta" / f"{fp.frame_id}.json", "w") as f:
            json.dump(result["meta"], f, indent=2)
        return fp.frame_id, True, None
    except Exception as e:
        return fp.frame_id, False, str(e)


def _load_per_regime_transforms(out_root: Path) -> dict[str, dict] | None:
    """Read transforms_by_regime.json from the output directory if present."""
    path = out_root / "transforms_by_regime.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_sun_alt_for_day(day: str) -> dict[str, float]:
    """frame_id → sun_alt_deg from labels/weak_labels.csv ephemeris rows for the given day."""
    csv_path = Path(__file__).parent / "labels" / "weak_labels.csv"
    if not csv_path.exists():
        return {}
    import csv as csv_mod
    out: dict[str, float] = {}
    prefix = f"ccd1_{day}_"  # frame_id format
    with open(csv_path, newline="") as f:
        for row in csv_mod.DictReader(f):
            fid = row.get("frame_id", "")
            if not fid.startswith(prefix):
                continue
            if row["source"] == "ephemeris" and row["attribute"] == "sun_alt_deg":
                try:
                    out[fid] = float(row["value"])
                except (TypeError, ValueError):
                    pass
    return out


def main():
    # Default config + NAS paths resolve relative to this script's directory,
    # so the script works regardless of which cwd it's invoked from (cron,
    # manual shell, etc.) without needing to cd first.
    SCRIPT_DIR = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True, help="YYYYMMDD")
    parser.add_argument("--allsky-root",
                        default=os.environ.get("NAS_ALLSKY_PATH", "/mnt/allsky_images"))
    parser.add_argument("--thermal-root",
                        default=os.environ.get(
                            "NAS_THERMAL_PATH", "/mnt/astro_image_thermal") + "/" +
                            os.environ.get("NAS_THERMAL_UUID", ""))
    parser.add_argument("--config", default=str(SCRIPT_DIR / "alignment_config.json"))
    parser.add_argument("--output-root", default=None, help="defaults to dataset_v2_<day>")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="Number of parallel jobs")
    parser.add_argument("--per-regime-align", action="store_true",
                        help="Apply per-regime transforms from <output-root>/transforms_by_regime.json "
                             "if present. Falls back to static config when the file is missing or "
                             "a frame's regime isn't covered.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames whose mask PNG already exists in the output. Use to resume "
                             "an interrupted day's regen without redoing the frames already done.")
    args = parser.parse_args()

    print(f"Thermal thresholds: ABS={ABS_THRESHOLD_C}C  REL={REL_DELTA_C}C  "
          f"SIGMA={SIGMOID_SIGMA_C}C  (override via THERMAL_ABS_C/THERMAL_REL_C/THERMAL_SIGMA_C)")

    with open(args.config) as f:
        config = json.load(f)

    out_root = Path(args.output_root or f"dataset_v2_{args.day}")
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)

    # Optional per-regime alignment
    transforms_by_regime: dict[str, dict] | None = None
    sun_alt_by_frame: dict[str, float] | None = None
    if args.per_regime_align:
        transforms_by_regime = _load_per_regime_transforms(out_root)
        if transforms_by_regime is None:
            print(f"WARNING: --per-regime-align set but {out_root}/transforms_by_regime.json missing — "
                  "falling back to static config for all frames")
        else:
            sun_alt_by_frame = _load_sun_alt_for_day(args.day)
            if not sun_alt_by_frame:
                print(f"WARNING: --per-regime-align set but no sun_alt available in "
                      "labels/weak_labels.csv for this day — falling back to static config")
                transforms_by_regime = None
            else:
                regimes_with_transforms = sorted(transforms_by_regime.keys())
                print(f"per-regime alignment enabled: {len(regimes_with_transforms)} regimes "
                      f"({', '.join(regimes_with_transforms)}), {len(sun_alt_by_frame)} frames with sun_alt")

    pairs = list(discover_pairs(Path(args.allsky_root), Path(args.thermal_root), args.day))
    if args.sample_stride > 1:
        pairs = pairs[:: args.sample_stride]
    if args.max_pairs > 0:
        idx = np.linspace(0, len(pairs) - 1, args.max_pairs, dtype=int)
        pairs = [pairs[i] for i in idx]

    total_pairs = len(pairs)
    if args.skip_existing:
        masks_dir = out_root / "masks"
        pairs = [p for p in pairs if not (masks_dir / f"{p.frame_id}.png").exists()]
        skipped = total_pairs - len(pairs)
        print(f"Resume mode: skipping {skipped} frames that already have masks "
              f"({len(pairs)} new + {skipped} existing = {total_pairs} total)")

    print(f"Found {len(pairs)} paired frames for {args.day} — processing with {args.jobs} jobs")
    t0 = time.time()
    ok = fail = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(worker, fp, config, out_root,
                                   sun_alt_by_frame, transforms_by_regime)
                   for fp in pairs]

        for i, future in enumerate(futures):
            fid, success, err = future.result()
            if success:
                ok += 1
            else:
                fail += 1
                if err != "Skipped":
                    print(f"  [{fid}] FAIL: {err}")

            if (i + 1) % 50 == 0 or i + 1 == len(pairs):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(futures) - (i + 1)) / max(rate, 1e-6)
                print(f"  {i + 1}/{len(futures)}  ok={ok} fail={fail}  "
                      f"{rate:.1f} fps  eta={eta:.0f}s")

    dt = time.time() - t0
    print(f"Done: {ok} ok, {fail} fail in {dt:.1f}s ({ok / max(dt, 1e-6):.1f} fps)")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
