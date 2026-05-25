"""mask_cleanup.py — Interactive sensor-contamination cleanup for individual masks.

Sensor contamination (warm dust, lens condensation residue) produces isolated
bright pixels in the cloud-probability mask that don't correspond to real cloud.
True cloud has spatial coherence — neighboring pixels are also bright. This
module detects pixels that exceed the local 5x5 median by a threshold and marks
them as no-data (255).

Intended to be invoked from the Streamlit labeling UI on a per-frame basis,
NOT in batch — the threshold-driven approach can clip real cloud edges if
applied indiscriminately, so a human-in-the-loop confirmation is required.

Usage (from labeling_tool.py):
    from mask_cleanup import clean_anomalous_pixels, log_cleanup
    cleaned, n = clean_anomalous_pixels(mask, threshold=0.3)
    if approved:
        cv2.imwrite(mask_path + ".original", mask)  # backup once
        cv2.imwrite(mask_path, cleaned)
        log_cleanup(frame_id, threshold, n, labeler_id)
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

NO_DATA_VALUE = 255
PROJECT_ROOT = Path(__file__).parent.resolve()
CLEANUP_LOG_CSV = PROJECT_ROOT / "labels" / "mask_cleanups.csv"
CLEANUP_LOG_COLUMNS = [
    "frame_id", "timestamp", "threshold", "neighborhood",
    "n_pixels_marked", "n_pixels_valid_before", "labeler_id",
]


def clean_anomalous_pixels(
    mask: np.ndarray,
    threshold: float = 0.3,
    neighborhood: int = 5,
) -> tuple[np.ndarray, int]:
    """Mark pixels that are anomalously brighter than their local neighborhood.

    Args:
        mask: uint8 mask, 0..254 = cloud_prob * 254, 255 = no-data.
        threshold: probability units a pixel must exceed its local median
            by to be flagged as contamination. 0.3 is a good default — high
            enough to spare real cloud edges (which only differ from
            neighbours by ~0.1-0.2), low enough to catch obvious bright
            specks (which often differ by 0.5+).
        neighborhood: median-filter window size (odd integer). 5 captures
            local context without smearing across cloud features.

    Returns:
        (cleaned_mask, n_pixels_marked). The cleaned mask has marked pixels
        set to NO_DATA_VALUE; downstream code treats them as missing rather
        than zero, so they're correctly excluded from the mean.
    """
    valid = mask != NO_DATA_VALUE
    if not valid.any():
        return mask.copy(), 0

    # Convert to float probability for arithmetic
    p = mask.astype(np.float32) / 254.0

    # Fill no-data cells with the global valid median so the local-median
    # filter doesn't pull values toward zero near the patch edge.
    fill = float(np.median(p[valid]))
    p_filled = np.where(valid, p, fill)
    local_median = median_filter(p_filled, size=neighborhood)

    # Outlier: pixel value exceeds its local neighborhood by `threshold`.
    # Only consider currently-valid pixels (no-data stays no-data).
    outliers = (p - local_median > threshold) & valid

    cleaned = mask.copy()
    cleaned[outliers] = NO_DATA_VALUE
    return cleaned, int(outliers.sum())


def log_cleanup(
    frame_id: str,
    threshold: float,
    n_marked: int,
    n_valid_before: int,
    labeler_id: str,
) -> None:
    """Append a row to labels/mask_cleanups.csv for reproducibility audit.

    The paper methodology section can cite this CSV: how many frames received
    cleanup treatment, average pixels modified, threshold distribution.
    """
    CLEANUP_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CLEANUP_LOG_CSV.exists()
    with open(CLEANUP_LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLEANUP_LOG_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({
            "frame_id": frame_id,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "threshold": f"{threshold:.2f}",
            "neighborhood": 5,
            "n_pixels_marked": n_marked,
            "n_pixels_valid_before": n_valid_before,
            "labeler_id": labeler_id,
        })
