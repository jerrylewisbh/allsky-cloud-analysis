"""
align_optimize.py — MI maximization over fine-alignment parameters
(rotation, x_offset, y_offset) for the fisheye→thermal projection.

Used by compute_per_regime_transforms.py to find per-regime alignment
corrections to the static calibration. The corrections capture mount drift
from wind, thermal expansion, and gradual sensor settling.

The optimizer keeps `fov` and `dist` (lens optical properties) at the static
calibration values — only mount/sensor angular drift is corrected per regime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import minimize

# Reuse the same primitives the mask-gen pipeline uses
from make_masks_v2 import (
    build_remap_matrices, load_thermal, sigmoid, alignment_mi_score,
    ABS_THRESHOLD_C, REL_DELTA_C, SIGMOID_SIGMA_C,
)


# Bounds for the optimizer — wind/thermal drift shouldn't exceed these
ROT_BOUND_DEG = 2.0      # ± from static rotation
OFF_BOUND_PCT = 0.05     # ± from static x_off / y_off (fraction of half-width)


def _warp_to_mask(thermal_raw: np.ndarray, ambient_c: float, img_shape: tuple,
                  fov: float, rot: float, x_off: float, y_off: float,
                  dist: float, proj_on: int) -> Optional[tuple[np.ndarray, np.ndarray, tuple]]:
    """Warp thermal to RGB space with given projection params. Returns
    (p_combined, valid_mask, crop_bounds) where crop_bounds = (y0, x0, y1, x1)
    for cropping the RGB to match the warped thermal extent."""
    a_h, a_w = img_shape[:2]
    t_h, t_w = thermal_raw.shape
    map_x, map_y, valid_full = build_remap_matrices(
        a_w, a_h, t_w, t_h, 180.0, fov, rot, x_off, y_off, dist, proj_on,
    )
    coords = np.argwhere(valid_full)
    if len(coords) == 0:
        return None
    pad = 30
    y0 = max(0, int(coords[:, 0].min()) - pad)
    x0 = max(0, int(coords[:, 1].min()) - pad)
    y1 = min(a_h, int(coords[:, 0].max()) + pad + 1)
    x1 = min(a_w, int(coords[:, 1].max()) + pad + 1)
    map_x_c = map_x[y0:y1, x0:x1]
    map_y_c = map_y[y0:y1, x0:x1]
    valid_c = valid_full[y0:y1, x0:x1]

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
    invalid_thermal = (~valid_c) | (warped_valid < 0.999)

    # Thermal cloud probability (same as make_masks_v2)
    p_abs = sigmoid((warped_thermal - ABS_THRESHOLD_C) / SIGMOID_SIGMA_C)
    p_rel = sigmoid((warped_thermal - (ambient_c + REL_DELTA_C)) / SIGMOID_SIGMA_C)
    p_thermal = np.maximum(p_abs, p_rel)
    p_thermal = np.where(invalid_thermal, np.nan, p_thermal)

    valid_mask = ~invalid_thermal
    return p_thermal, valid_mask, (y0, x0, y1, x1)


def optimize_alignment(thermal_raw: np.ndarray, ambient_c: float,
                       img_full: np.ndarray,
                       static_config: dict,
                       verbose: bool = False) -> Optional[dict]:
    """Find (rot, x_off, y_off) that maximizes MI between thermal-derived and
    RGB-derived cloud probability, starting from the static calibration.

    Returns dict with 'rot', 'x_off', 'y_off', 'mi_score', 'mi_score_default'
    if optimization improves over the default; None if optimization fails or
    doesn't improve.
    """
    fov = static_config["fov"]
    dist = static_config["dist"]
    proj_on = static_config.get("proj_on", 1)
    rot0 = static_config["rot"]
    x0_param = static_config["x_off"]
    y0_param = static_config["y_off"]

    # Baseline MI with the static transform
    base = _warp_to_mask(
        thermal_raw, ambient_c, img_full.shape,
        fov, rot0, x0_param, y0_param, dist, proj_on,
    )
    if base is None:
        return None
    p_default, valid_default, (y0, x0, y1, x1) = base
    img_crop_default = img_full[y0:y1, x0:x1]
    mi_default = alignment_mi_score(np.nan_to_num(p_default, nan=0.0),
                                    img_crop_default, valid_default)
    if mi_default is None:
        return None  # can't even baseline — featureless scene

    def objective(params: np.ndarray) -> float:
        rot, x_off, y_off = params
        out = _warp_to_mask(
            thermal_raw, ambient_c, img_full.shape,
            fov, rot, x_off, y_off, dist, proj_on,
        )
        if out is None:
            return 0.0  # large penalty: this transform produced empty valid region
        p, v, (yy0, xx0, yy1, xx1) = out
        img_crop = img_full[yy0:yy1, xx0:xx1]
        mi = alignment_mi_score(np.nan_to_num(p, nan=0.0), img_crop, v)
        if mi is None:
            return 0.0
        return -mi  # minimize negative MI = maximize MI

    # Nelder-Mead — no gradients needed, robust to noisy objective
    x_init = np.array([rot0, x0_param, y0_param])
    bounds = [(rot0 - ROT_BOUND_DEG, rot0 + ROT_BOUND_DEG),
              (x0_param - OFF_BOUND_PCT, x0_param + OFF_BOUND_PCT),
              (y0_param - OFF_BOUND_PCT, y0_param + OFF_BOUND_PCT)]
    try:
        result = minimize(
            objective, x_init,
            method="Nelder-Mead",
            options={"xatol": 0.05, "fatol": 0.005, "maxiter": 80,
                     "initial_simplex": _build_simplex(x_init, ROT_BOUND_DEG / 4, OFF_BOUND_PCT / 4)},
        )
    except Exception as e:
        if verbose:
            print(f"  optimizer failed: {e}")
        return None

    rot_opt, x_off_opt, y_off_opt = result.x
    mi_opt = -result.fun

    # Reject if outside bounds (optimizer doesn't enforce, just gets there via bad MI)
    if not (bounds[0][0] <= rot_opt <= bounds[0][1] and
            bounds[1][0] <= x_off_opt <= bounds[1][1] and
            bounds[2][0] <= y_off_opt <= bounds[2][1]):
        if verbose:
            print(f"  optimizer wandered out of bounds: rot={rot_opt:.2f}, "
                  f"x_off={x_off_opt:+.3f}, y_off={y_off_opt:+.3f}")
        return None

    if mi_opt <= mi_default + 0.005:
        # Improvement too marginal to be meaningful
        if verbose:
            print(f"  no meaningful improvement: {mi_default:.3f} → {mi_opt:.3f}")
        return None

    return {
        "rot": float(rot_opt),
        "x_off": float(x_off_opt),
        "y_off": float(y_off_opt),
        "mi_score": float(mi_opt),
        "mi_score_default": float(mi_default),
        "improvement": float(mi_opt - mi_default),
        "iterations": int(result.nit),
    }


def _build_simplex(x0: np.ndarray, step_rot: float, step_off: float) -> np.ndarray:
    """Initial simplex for Nelder-Mead, sized to the typical drift magnitude.
    Three points perturbed from x0, one per dimension."""
    simplex = np.array([
        x0,
        x0 + np.array([step_rot, 0, 0]),
        x0 + np.array([0, step_off, 0]),
        x0 + np.array([0, 0, step_off]),
    ])
    return simplex


__all__ = ["optimize_alignment"]
