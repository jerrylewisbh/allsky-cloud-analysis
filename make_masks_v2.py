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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# ---- physics thresholds (mirror docs/sky-condition.md and firmware) ----
ABS_THRESHOLD_C = -5.0       # warmer than this absolute => cloud
REL_DELTA_C = -10.0          # warmer than (ambient + REL_DELTA) => cloud
SIGMOID_SIGMA_C = 3.0        # softness of the cloud transition (°C)

# ---- day / night handling (lux from sensors block) ----
LUX_NIGHT = 1.0              # below: thermal-only
LUX_DAY = 500.0              # above: full RGB refinement
HYTA_NRBR_CLEAR = -0.35      # NRBR <= -0.35 => clear blue sky
HYTA_NRBR_CLOUD = +0.05      # NRBR >= +0.05 => fully white cloud

# ---- spatial / output ----
GUIDED_RADIUS = 16
GUIDED_EPS = 1e-3
OUTPUT_SIZE = 256
NO_DATA = 255


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
    arr = np.asarray(frame, dtype=np.float32)
    if arr.size == 768:
        arr = arr.reshape((24, 32))
    elif arr.size == 384:
        arr = arr.reshape((16, 24))
    else:
        return None, None
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

    # ---- thermal cloud probability (per-pixel firmware physics) ----
    ambient = float(sensors.get("temp", 20.0))
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
        p_thermal_refined = guided_filter(gray, p_thermal, GUIDED_RADIUS, GUIDED_EPS)
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
            "fw_sky_condition": sensors.get("sky_condition"),
            "fw_sky_cloud_fraction": sensors.get("sky_cloud_fraction"),
            "fw_sky_abs_cloud_fraction": sensors.get("sky_abs_cloud_fraction"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True, help="YYYYMMDD")
    parser.add_argument("--allsky-root", default="/Volumes/allsky_images")
    parser.add_argument("--thermal-root",
                        default="/Volumes/astro_image_thermal/ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212")
    parser.add_argument("--config", default="allsky-cloud-analysis/alignment_config.json")
    parser.add_argument("--output-root", default=None, help="defaults to dataset_v2_<day>")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    parser.add_argument("--sample-stride", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    out_root = Path(args.output_root or f"dataset_v2_{args.day}")
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)

    pairs = list(discover_pairs(Path(args.allsky_root), Path(args.thermal_root), args.day))
    if args.sample_stride > 1:
        pairs = pairs[:: args.sample_stride]
    if args.max_pairs > 0:
        idx = np.linspace(0, len(pairs) - 1, args.max_pairs, dtype=int)
        pairs = [pairs[i] for i in idx]

    print(f"Found {len(pairs)} paired frames for {args.day}")
    t0 = time.time()
    ok = fail = 0
    for i, fp in enumerate(pairs):
        try:
            result = process_frame(fp, config)
            if result is None:
                fail += 1
                continue
            cv2.imwrite(str(out_root / "images" / f"{fp.frame_id}.jpg"), result["img"])
            cv2.imwrite(str(out_root / "masks" / f"{fp.frame_id}.png"), result["mask"])
            with open(out_root / "meta" / f"{fp.frame_id}.json", "w") as f:
                json.dump(result["meta"], f, indent=2)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  [{fp.frame_id}] FAIL: {e}")
        if (i + 1) % 25 == 0 or i + 1 == len(pairs):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pairs) - (i + 1)) / max(rate, 1e-6)
            print(f"  {i + 1}/{len(pairs)}  ok={ok} fail={fail}  "
                  f"{rate:.1f} fps  eta={eta:.0f}s")
    dt = time.time() - t0
    print(f"Done: {ok} ok, {fail} fail in {dt:.1f}s ({ok / max(dt, 1e-6):.1f} fps)")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
