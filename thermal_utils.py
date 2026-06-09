"""Shared helpers for MLX90640 thermal frames.

The ESP firmware now emits the FULL 32x24 frame (768 values). The four corners
are clipped by the enclosure (they see warm plastic, not sky), so we mask them
out before alignment / mask-gen / stats. Legacy captures are the cropped 24x16
(384) format and pass through unchanged.

The corner mask here MUST match keep_pixel() in the firmware
(external_components_local/sky_thermal/sky_thermal.h).
"""
import numpy as np

# Per-row corner-clip depth (symmetric), measured against clear sky.
# Sum removed = 128 px; 640 kept of 768.
_DEPTH = np.array([7, 6, 4, 4, 3, 3, 2, 2, 1, 0, 0, 0,
                   0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 6, 7])
_X = np.arange(32)
# Shape (24, 32). True = keep (good sky), False = clipped corner.
KEEP = (_X[None, :] >= _DEPTH[:, None]) & (_X[None, :] <= 31 - _DEPTH[:, None])


def reshape_thermal(raw_1d):
    """Reshape a 1-D thermal frame to 2-D.

    Returns (frame2d, is_full):
      768 -> (24, 32), is_full=True  (current full-frame firmware)
      384 -> (16, 24), is_full=False (legacy cropped firmware)
    """
    a = np.asarray(raw_1d, dtype=np.float32)
    if a.size == 768:
        return a.reshape((24, 32)), True
    if a.size == 384:
        return a.reshape((16, 24)), False
    raise ValueError(f"unexpected thermal frame length {a.size}")


def apply_corner_mask(frame2d, fill=np.nan):
    """Set the clipped corners of a full 32x24 frame to `fill` (default NaN).
    Legacy 24x16 frames are returned unchanged (they were already corner-free)."""
    if frame2d.shape == (24, 32):
        out = frame2d.astype(np.float32).copy()
        out[~KEEP] = fill
        return out
    return frame2d


def fill_corners_clear(frame2d):
    """Replace clipped corners with the clear-sky baseline (10th pct of kept
    pixels). Unlike NaN this is safe for cv2.remap / warping and MI, and the
    corners read as clear sky (no false cloud). Full frames only; legacy
    24x16 frames returned unchanged."""
    if frame2d.shape != (24, 32):
        return frame2d
    out = frame2d.astype(np.float32).copy()
    kept = out[KEEP]
    base = float(np.nanpercentile(kept, 10)) if kept.size else 0.0
    out[~KEEP] = base
    return out


def ambient_from_sensors(sensors, default=20.0):
    """Read ambient temperature from a /json sensors dict, tolerating both the
    new firmware key ('ambient') and the legacy BME key ('temp')."""
    if not sensors:
        return default
    v = sensors.get("ambient", sensors.get("temp"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_thermal(raw_1d, mask_corners=True, fill=np.nan):
    """Reshape and (optionally) mask the clipped corners. Returns a 2-D array.

    Use mask_corners=True for alignment / mask-gen / stats (NaN corners, then
    use np.nan* reductions). Use mask_corners=False for raw visualization.
    """
    frame2d, is_full = reshape_thermal(raw_1d)
    if mask_corners and is_full:
        frame2d = apply_corner_mask(frame2d, fill=fill)
    return frame2d
