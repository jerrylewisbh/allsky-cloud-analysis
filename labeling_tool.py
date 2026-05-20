"""Cloud labeling UI per docs/labeling-protocol.md.

Run:
    streamlit run labeling_tool.py
Optionally point at a different dataset root or glob:
    DATASET_ROOT=. DATASET_GLOB='dataset_*' streamlit run labeling_tool.py
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from auto_classify import classify as auto_classify

CLASSES = ["clear", "ci", "cs_cc", "ac_as", "cu", "sc", "st", "ns_cb", "multi"]
CLASS_DESCRIPTIONS = {
    "clear": "Clear sky (>95% cloud-free)",
    "ci": "Cirrus — thin, fibrous, high, isolated streaks",
    "cs_cc": "Cirrostratus / Cirrocumulus — high sheet or ripples",
    "ac_as": "Altocumulus / Altostratus — mid-level cells or smooth sheet",
    "cu": "Cumulus — discrete fluffy cells, flat bases, blue gaps",
    "sc": "Stratocumulus — low rolls/patches, mostly continuous",
    "st": "Stratus / Fog — uniform low grey",
    "ns_cb": "Nimbostratus / Cumulonimbus — deep, often precipitating",
    "multi": "Multi-cloud — two or more types in distinct regions",
}
CONFIDENCES = ["high", "medium", "low"]
QC_FLAGS = [
    "sun_artifact",
    "lens_contamination",
    "rain_on_lens",
    "nighttime_no_moon",
    "horizon_contamination",
    "smoke",
]

PROJECT_ROOT = Path(__file__).parent.resolve()
LABELS_CSV = PROJECT_ROOT / "labels" / "hand_labeled.csv"
WEAK_LABELS_CSV = PROJECT_ROOT / "labels" / "weak_labels.csv"
AUTO_LABELS_CSV = PROJECT_ROOT / "labels" / "auto_labels.csv"
ALLSKY_ROOT = Path(os.environ.get("ALLSKY_ROOT", "/Volumes/allsky_images"))

OKTA_LABEL = {0: "0/8 SKC", 1: "1/8 FEW", 2: "2/8 FEW", 3: "3/8 SCT",
              4: "4/8 SCT", 5: "5/8 BKN", 6: "6/8 BKN", 7: "7/8 BKN", 8: "8/8 OVC"}
LABEL_COLUMNS = [
    "frame_id", "rgb_path", "mask_path", "timestamp",
    "class", "confidence", "labeler_id", "labeled_at", "labeling_seconds",
    *QC_FLAGS,
    "notes",
]


def discover_pairs(root: Path, glob_pattern: str) -> list[dict]:
    pairs = []
    for ds_dir in sorted(root.glob(glob_pattern)):
        img_dir = ds_dir / "images"
        mask_dir = ds_dir / "masks"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue
        for jpg in sorted(img_dir.glob("*.jpg")):
            png = mask_dir / f"{jpg.stem}.png"
            if not png.exists():
                continue
            pairs.append({
                "frame_id": jpg.stem,
                "rgb_path": str(jpg.resolve()),
                "mask_path": str(png.resolve()),
                "timestamp": parse_timestamp(jpg.stem),
            })
    return pairs


def parse_timestamp(stem: str) -> dt.datetime | None:
    m = re.search(r"(\d{8}_\d{6})", stem)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def load_labels() -> pd.DataFrame:
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        for c in LABEL_COLUMNS:
            if c not in df.columns:
                df[c] = "" if c not in QC_FLAGS else False
        return df[LABEL_COLUMNS]
    return pd.DataFrame(columns=LABEL_COLUMNS)


def save_label(row: dict) -> None:
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = load_labels()
    df = df[df["frame_id"] != row["frame_id"]]
    df = pd.concat([df, pd.DataFrame([row], columns=LABEL_COLUMNS)], ignore_index=True)
    df.to_csv(LABELS_CSV, index=False)


NO_DATA_VALUE = 255  # v2 mask convention: 0..254 = cloud prob * 254, 255 = no-data

# Perceptual colormap: deep blue (clear) → cyan → white → yellow → red (cloud).
# Hand-built so 0 = unmistakably "sky" and 1 = unmistakably "cloud", with a
# clear midpoint at p=0.5. No-data renders separately as a dim grey stripe.
def _build_sky_cloud_colormap() -> np.ndarray:
    stops = [
        (0.00, (10, 20, 90)),     # deep navy — confident clear sky
        (0.25, (40, 110, 200)),   # mid blue
        (0.45, (180, 220, 240)),  # pale blue — thin / uncertain
        (0.55, (250, 240, 200)),  # pale yellow — possible cloud
        (0.75, (245, 160, 60)),   # orange — likely cloud
        (1.00, (200, 30, 30)),    # deep red — confident dense cloud
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        for j in range(len(stops) - 1):
            x0, c0 = stops[j]
            x1, c1 = stops[j + 1]
            if x0 <= x <= x1:
                t = (x - x0) / max(x1 - x0, 1e-9)
                lut[i] = [int(round(c0[k] * (1 - t) + c1[k] * t)) for k in range(3)]
                break
    return lut


SKY_CLOUD_LUT = _build_sky_cloud_colormap()


# Colormap registry. Each value is a 256x3 RGB LUT applied to the cloud
# probability values (0..255). Different palettes surface different features —
# perceptually uniform (viridis/turbo) for quantitative reading, high-contrast
# (inferno/plasma/jet) for spotting cellular texture (Ac/Sc), custom sky_cloud
# for intuitive blue=clear / red=cloud reading.
def _cv2_lut(cv2_cmap: int) -> np.ndarray:
    """Build a 256x3 RGB LUT from a cv2 colormap id (cv2 outputs BGR)."""
    table = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(table, cv2_cmap).reshape(256, 3)
    return bgr[:, ::-1].copy()  # BGR → RGB


def _grayscale_lut() -> np.ndarray:
    return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=-1)


COLORMAPS: dict[str, np.ndarray] = {
    "sky_cloud (custom)": SKY_CLOUD_LUT,
    "magma":              _cv2_lut(cv2.COLORMAP_MAGMA),
    "inferno":            _cv2_lut(cv2.COLORMAP_INFERNO),
    "plasma":             _cv2_lut(cv2.COLORMAP_PLASMA),
    "viridis":            _cv2_lut(cv2.COLORMAP_VIRIDIS),
    "turbo":              _cv2_lut(cv2.COLORMAP_TURBO),
    "twilight":           _cv2_lut(cv2.COLORMAP_TWILIGHT),
    "jet":                _cv2_lut(cv2.COLORMAP_JET),
    "grayscale":          _grayscale_lut(),
}


@st.cache_data(show_spinner=False)
def index_full_allsky(date_yyyymmdd: str, allsky_root: str) -> dict[str, str]:
    """Build {frame_id: absolute_path} for the full-fisheye captures of one day.
    Cached per-day so the NAS walk happens once."""
    root = Path(allsky_root) / "images" / date_yyyymmdd
    if not root.is_dir():
        return {}
    return {p.stem: str(p.resolve()) for p in root.rglob("*.jpg") if "thumbnails" not in p.parts}


def find_full_allsky_path(frame_id: str) -> str | None:
    """Locate the full fisheye for a frame. The NAS organizes by *observing
    session*, so a post-midnight frame stamped 20260519_010006 actually lives
    under 20260518/night/19_01/. Try the frame's calendar date first, then the
    previous day."""
    m = re.search(r"(\d{8})_\d{6}", frame_id)
    if not m:
        return None
    day = m.group(1)
    prev_day = (dt.datetime.strptime(day, "%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")
    for d in (day, prev_day):
        idx = index_full_allsky(d, str(ALLSKY_ROOT))
        if frame_id in idx:
            return idx[frame_id]
    return None


@st.cache_data(show_spinner=False)
def load_auto_labels(csv_path: str, mtime: float) -> dict[str, dict]:
    """Returns {frame_id: {auto_class, auto_confidence, auto_reasoning}}.
    Used by the review-queue filter — much faster than recomputing auto_classify
    on every Next click. Regenerate with `python auto_classify_batch.py`.
    """
    p = Path(csv_path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            out[row["frame_id"]] = row
    return out


@st.cache_data(show_spinner=False)
def load_weak_labels(csv_path: str, mtime: float) -> dict[str, dict[tuple, dict]]:
    """Returns {frame_id: {(source, attribute): row}}.
    `mtime` participates in the cache key so the cache invalidates when the
    file changes on disk (background re-fetches stay visible)."""
    p = Path(csv_path)
    if not p.exists():
        return {}
    by_frame: dict[str, dict[tuple, dict]] = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            fid = row["frame_id"]
            by_frame.setdefault(fid, {})[(row["source"], row["attribute"])] = row
    return by_frame


def _sun_regime(sun_alt_deg: float) -> str:
    if sun_alt_deg >= 6: return "DAY"
    if sun_alt_deg >= -6: return "TWILIGHT"
    if sun_alt_deg >= -12: return "NAUTICAL"
    if sun_alt_deg >= -18: return "ASTRO"
    return "DARK"


def _fmt_offset(seconds: int) -> str:
    sign = "−" if seconds < 0 else "+"
    s = abs(seconds)
    if s >= 60: return f"{sign}{s // 60} min"
    return f"{sign}{s} s"


def render_context_panel(weak: dict[tuple, dict]) -> None:
    """Multi-source context strip: ephemeris + METAR + weather station + derived + GOES."""
    if not weak:
        st.caption("No weak labels for this frame.")
        return

    def get(source: str, attr: str) -> dict | None:
        return weak.get((source, attr))

    def val(source: str, attr: str, default=None, fmt=None):
        r = get(source, attr)
        if not r: return default
        v = r["value"]
        if fmt: return fmt(v)
        try: return float(v)
        except (TypeError, ValueError): return v

    sun_alt = val("ephemeris", "sun_alt_deg")
    moon_alt = val("ephemeris", "moon_alt_deg")
    moon_phase = val("ephemeris", "moon_phase_pct")
    regime = _sun_regime(sun_alt) if sun_alt is not None else "?"

    csi = val("derived", "daytime_clear_sky_index")
    mpsas = val("esp32_sensor", "sky_brightness_mpsas")
    solar = val("weather_station", "solar_irradiance_wm2")
    humidity = val("weather_station", "humidity_pct")
    pressure = val("weather_station", "pressure_hpa")

    okta_cyyc = val("metar", "coverage_okta")
    base_cyyc = val("metar", "cloud_base_height_m")
    genus = val("metar", "cloud_genus_hint")

    goes_mask = val("goes19_acmc", "cloud_present")
    goes_phase = val("goes19_actpc", "cloud_top_phase")
    goes_height = val("goes19_achac", "cloud_top_height_m")
    goes_cod = val("goes19_codc", "cloud_optical_depth")

    # Headline strip: regime + the most actionable signal for that regime
    headline_cols = st.columns([1, 1, 1, 1, 1])
    headline_cols[0].metric("Regime", regime,
                            help=f"sun_alt = {sun_alt:.1f}°" if sun_alt is not None else "")
    if regime == "DAY":
        headline_cols[1].metric("CSI (1=clear)", f"{csi:.2f}" if csi is not None else "—",
                                help="Clear-Sky Index from AWNET solarradiation vs Haurwitz clear-sky model")
        headline_cols[2].metric("Solar W/m²", f"{solar:.0f}" if solar is not None else "—")
    else:
        headline_cols[1].metric("mpsas (lower=brighter)", f"{mpsas:.2f}" if mpsas is not None else "—",
                                help="ESP SQM — clouds over Calgary reflect city skyglow back, lowering mpsas")
        headline_cols[2].metric("Moon", f"{moon_alt:.0f}°  {moon_phase:.0f}%" if moon_alt is not None else "—",
                                help="moon altitude · phase. Below horizon = dark; above = scattered moonlight changes RGB")

    headline_cols[3].metric("METAR okta", OKTA_LABEL.get(int(okta_cyyc), "—") if okta_cyyc is not None else "—",
                            help="From CYYC. Genus hint: " + (str(genus) if genus else "none"))
    if goes_mask is not None or goes_phase is not None or goes_height is not None:
        goes_line = []
        if goes_mask is not None:
            goes_line.append("cloudy" if int(goes_mask) == 1 else "clear")
        if goes_phase is not None:
            goes_line.append(str(goes_phase))
        # Altitude family from cloud-top height (rough WMO bands)
        if goes_height is not None and goes_height > 0:
            family = "high" if goes_height >= 6000 else "mid" if goes_height >= 2000 else "low"
            goes_line.append(f"{goes_height/1000:.1f} km ({family})")
        help_bits = []
        if goes_cod is not None: help_bits.append(f"COD {goes_cod:.1f}")
        if goes_height is not None: help_bits.append(f"top {goes_height:.0f} m")
        headline_cols[4].metric("GOES-19 overhead", " · ".join(goes_line) or "—",
                                help=" · ".join(help_bits) if help_bits else "")
    else:
        headline_cols[4].metric("GOES-19", "pending", help="fetch_goes.py running in background")

    # Honest framing
    if regime == "DAY":
        regime_note = "Daytime: trust RGB primarily; CSI gives a direct local cloud signal."
    elif regime in ("TWILIGHT", "NAUTICAL"):
        regime_note = "Twilight: both RGB and thermal carry info but neither is fully reliable. Cross-check with METAR genus hint."
    else:
        regime_note = "Nighttime: RGB is moonless or marginal — trust thermal + mpsas. Cu/Sc invisible without moonlight."
    st.caption(f"**{regime_note}**  ·  METAR sees the whole hemisphere from the airport; your crop is a ~75° patch — disagreement is expected.")

    # Expandable details
    with st.expander("All weak labels for this frame"):
        det_cols = st.columns(3)
        # Group by source
        by_source: dict[str, list[tuple[str, dict]]] = {}
        for (src, attr), row in weak.items():
            by_source.setdefault(src, []).append((attr, row))
        for i, src in enumerate(sorted(by_source)):
            with det_cols[i % 3]:
                st.markdown(f"**{src}**")
                for attr, row in sorted(by_source[src]):
                    unit = row.get("value_unit", "")
                    offs = row.get("source_distance_s", "")
                    offs_str = f" ({_fmt_offset(int(offs))})" if offs and offs.lstrip('-').isdigit() else ""
                    st.text(f"  {attr}: {row['value']} {unit}{offs_str}")


def colorize_mask(mask_path: str, colormap: str = "sky_cloud (custom)") -> np.ndarray:
    raw = np.array(Image.open(mask_path).convert("L"))
    valid = raw != NO_DATA_VALUE
    legacy_binary = raw.max() <= 1 or len(np.unique(raw)) <= 2
    if legacy_binary:
        valid = np.ones_like(raw, dtype=bool)
        probs_u8 = np.where(raw > 127, 255, 0).astype(np.uint8)
    else:
        probs_u8 = np.where(valid, raw, 0).astype(np.uint8)
    lut = COLORMAPS.get(colormap, SKY_CLOUD_LUT)
    colored = lut[probs_u8]
    # No-data: diagonal stripe in mid-grey so it's obviously "missing", not "clear"
    yy, xx = np.indices(raw.shape)
    stripe = ((yy + xx) // 6) % 2 == 0
    colored[(~valid) & stripe] = (60, 60, 60)
    colored[(~valid) & ~stripe] = (100, 100, 100)
    return colored


def _load_mask_components(mask_path: str, rgb_shape: tuple[int, int]):
    """Returns (probs in [0,1] for valid px, valid_mask bool, raw uint8 resized)."""
    raw = np.array(Image.open(mask_path).convert("L"))
    if raw.shape != rgb_shape:
        raw = np.array(Image.fromarray(raw).resize((rgb_shape[1], rgb_shape[0]), Image.NEAREST))
    if raw.max() <= 1 or len(np.unique(raw)) <= 2:
        # legacy binary mask: 0 or 255 only, no no-data convention
        valid = np.ones_like(raw, dtype=bool)
        probs = (raw > 127).astype(np.float32)
    else:
        valid = raw != NO_DATA_VALUE
        probs = np.where(valid, raw.astype(np.float32) / 254.0, 0.0)
    return probs, valid, raw


def make_overlay(rgb_path: str, mask_path: str,
                 colormap: str = "sky_cloud (custom)",
                 style: str = "soft",
                 threshold: float = 0.5,
                 alpha_max: float = 0.65) -> np.ndarray:
    """Render the RGB with the colorized thermal mask overlaid.

    style:
      "soft"    — alpha-blend that ramps with cloud probability (good for
                  reading intensity gradients and cellular structure).
      "hard"    — colorized thermal at full opacity where probability >=
                  threshold; pure RGB elsewhere (good for reading the binary
                  cloud/clear decision and precise cloud boundaries).
      "contour" — RGB everywhere, with a single isocontour line drawn at
                  the threshold (good for verifying alignment between
                  thermal-defined cloud edges and visual cloud edges).
    """
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    probs, valid, _ = _load_mask_components(mask_path, rgb.shape[:2])
    lut = COLORMAPS.get(colormap, SKY_CLOUD_LUT)
    probs_u8 = np.clip(probs * 255.0, 0, 255).astype(np.uint8)
    thermal_rgb = lut[probs_u8].astype(np.float32)
    overlay = rgb.astype(np.float32)

    if style == "hard":
        is_cloud = (probs >= threshold) & valid
        a = is_cloud.astype(np.float32) * alpha_max
        a3 = a[..., None]
        overlay = overlay * (1 - a3) + thermal_rgb * a3
    elif style == "contour":
        is_cloud = (probs >= threshold) & valid
        # Morphological gradient = pixels on the boundary of the binary mask
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge = cv2.morphologyEx(is_cloud.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        # Dilate to 2-px line for visibility
        edge = cv2.dilate(edge, kernel, iterations=1).astype(bool)
        # Use the colormap's "high-cloud" color (LUT[230]) for the line
        line_color = lut[230].astype(np.float32)
        overlay[edge] = line_color
    else:  # "soft"
        a = np.clip((probs - 0.15) / 0.7, 0, 1) * alpha_max
        a = a * valid.astype(np.float32)
        a3 = a[..., None]
        overlay = overlay * (1 - a3) + thermal_rgb * a3

    # No-data: dim diagonal stripe so labeler sees where there's no thermal data
    if not valid.all():
        yy, xx = np.indices(valid.shape)
        stripe = ((yy + xx) // 6) % 2 == 0
        nd_dim = (~valid) & stripe
        overlay[nd_dim] = overlay[nd_dim] * 0.35
    return overlay.clip(0, 255).astype(np.uint8)


def thermal_cloud_stats(mask_path: str) -> tuple[float, float, float]:
    """Returns (mean_cloud_prob_over_valid, fraction_above_0.5_over_valid, no_data_fraction)."""
    raw = np.array(Image.open(mask_path).convert("L"))
    if raw.max() <= 1 or len(np.unique(raw)) <= 2:
        return float((raw > 127).mean()), float((raw > 127).mean()), 0.0
    valid = raw != NO_DATA_VALUE
    if not valid.any():
        return float("nan"), float("nan"), 1.0
    probs = raw[valid].astype(np.float32) / 254.0
    return float(probs.mean()), float((probs >= 0.5).mean()), float((~valid).mean())


def _rgb_and_valid(rgb_path: str, mask_path: str):
    try:
        rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
    except (FileNotFoundError, OSError):
        return None, None
    mask = np.array(Image.open(mask_path).convert("L"))
    valid = mask != NO_DATA_VALUE
    if not valid.any():
        return rgb, None
    if mask.shape != rgb.shape[:2]:
        valid_img = Image.fromarray(valid.astype(np.uint8) * 255)
        valid = np.array(valid_img.resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
    return rgb, valid


def rgb_nrbr_mean(rgb_path: str, mask_path: str) -> float | None:
    """Mean (R-B)/(R+B) over thermal-valid pixels — daytime cloud cue."""
    rgb, valid = _rgb_and_valid(rgb_path, mask_path)
    if rgb is None or valid is None:
        return None
    r = rgb[..., 0][valid]
    b = rgb[..., 2][valid]
    return float(((r - b) / (r + b + 1e-6)).mean())


def rgb_v_mean(rgb_path: str, mask_path: str) -> float | None:
    """Mean HSV V over thermal-valid pixels — nighttime cloud cue."""
    rgb, valid = _rgb_and_valid(rgb_path, mask_path)
    if rgb is None or valid is None:
        return None
    return float(rgb.max(axis=-1)[valid].mean())


def advance(pairs: list[dict], labeled_ids: set[str], direction: int,
            skip_labeled: bool, review_filter=None) -> int:
    i = st.session_state.idx
    n = len(pairs)
    for _ in range(n):
        i = (i + direction) % n
        if skip_labeled and pairs[i]["frame_id"] in labeled_ids:
            continue
        if review_filter and not review_filter(pairs[i]):
            continue
        return i
    return st.session_state.idx


def main() -> None:
    st.set_page_config(page_title="Cloud Labeler", layout="wide")

    root = Path(os.environ.get("DATASET_ROOT", str(PROJECT_ROOT))).resolve()
    pattern = os.environ.get("DATASET_GLOB", "dataset_*")

    if "pairs" not in st.session_state:
        st.session_state.pairs = discover_pairs(root, pattern)
        st.session_state.idx = 0
        st.session_state.frame_started_at = time.time()
        st.session_state.labeler_id = ""

    pairs = st.session_state.pairs
    if not pairs:
        st.error(f"No (image, mask) pairs found under {root}/{pattern}.")
        st.info("Each dataset_* directory must contain images/ and masks/ with matching stems.")
        return

    labels_df = load_labels()
    labeled_ids = set(labels_df["frame_id"].astype(str))

    with st.sidebar:
        st.title("Cloud labeler")
        st.caption(f"Source: `{root}/{pattern}`")
        st.metric("Pairs found", len(pairs))
        st.metric("Labeled", f"{len(labeled_ids)} / {len(pairs)}")
        st.progress(len(labeled_ids) / len(pairs))

        st.session_state.labeler_id = st.text_input(
            "Labeler ID", value=st.session_state.labeler_id, max_chars=16,
            help="A short identifier — e.g. your initials. Required to save.",
        )

        skip_labeled = st.checkbox("Skip already-labeled", value=True)
        st.markdown("**Filters** — applied with AND")
        confidence_filter = st.selectbox(
            "Auto-label confidence",
            ["any", "high", "medium", "low", "medium+low (active learning)"],
            index=0,
            help=(
                "**any**: walk all frames.  "
                "**high**: validate the classifier's confident calls (Part A — "
                "confirm it's right when it claims to be).  "
                "**medium/low**: hard cases — humans add the most value here.  "
                "**medium+low**: combined for thorough first-pass review."
            ),
        )
        class_filter = st.selectbox(
            "Auto-label class",
            ["any"] + CLASSES,
            index=0,
            help=(
                "Narrow to frames where the auto-classifier predicted a specific "
                "class. Pairs with the confidence filter — e.g. **class=clear + "
                "confidence=high** validates the high-conf clear predictions."
            ),
        )
        colormap_name = st.selectbox(
            "Thermal colormap",
            list(COLORMAPS.keys()),
            index=0,
            help=(
                "Different colormaps reveal different features. **sky_cloud** is "
                "intuitive (blue=clear, red=cloud). **inferno/magma/plasma** maximize "
                "contrast at cell boundaries — best for spotting altocumulus/"
                "stratocumulus cellular texture. **viridis/turbo** are perceptually "
                "uniform — better when reading quantitative values. **twilight** is "
                "cyclic — useful for edge detection. **grayscale** = no color bias."
            ),
        )
        overlay_style = st.radio(
            "Overlay style",
            ["soft", "hard", "contour"],
            index=0,
            horizontal=True,
            help=(
                "**soft**: alpha-blend ramping with cloud probability (best for "
                "intensity gradients + cellular texture). "
                "**hard**: full opacity above threshold, RGB below (best for "
                "binary cloud/clear decisions + boundary precision). "
                "**contour**: line drawn at the threshold over RGB (best for "
                "verifying thermal-vs-RGB cloud-edge alignment)."
            ),
        )
        overlay_threshold = st.slider(
            "Cloud threshold (probability)",
            min_value=0.05, max_value=0.95, value=0.5, step=0.05,
            help="Used by hard + contour styles. 0.5 = balanced; lower catches thin cloud, higher requires confidence.",
        ) if overlay_style in ("hard", "contour") else 0.5

    # Cache key for weak labels — mtime in the key means background re-fetches
    # show up without restarting the tool. Defined here (above both the review
    # filter and the per-frame block) so both can use it.
    weak_mtime = WEAK_LABELS_CSV.stat().st_mtime if WEAK_LABELS_CSV.exists() else 0.0

    with st.sidebar:

        st.divider()
        st.subheader("Jump")
        idx_input = st.number_input(
            "Frame index", min_value=0, max_value=len(pairs) - 1,
            value=int(st.session_state.idx), step=1,
        )
        if int(idx_input) != st.session_state.idx:
            st.session_state.idx = int(idx_input)
            st.session_state.frame_started_at = time.time()
            st.rerun()

        st.divider()
        st.subheader("Label distribution")
        if len(labels_df):
            dist = labels_df["class"].value_counts().reindex(CLASSES, fill_value=0)
            st.bar_chart(dist)
        else:
            st.caption("No labels yet.")

    # Build review filter from the pre-computed auto_labels.csv (fast — no
    # per-frame mask loading). If auto_labels.csv is missing or stale,
    # the filter falls back to a live auto_classify call (slow but correct).
    auto_mtime = AUTO_LABELS_CSV.stat().st_mtime if AUTO_LABELS_CSV.exists() else 0.0
    auto_index = load_auto_labels(str(AUTO_LABELS_CSV), auto_mtime)

    # Translate the two filter dropdowns into sets of acceptable values.
    wanted_confidences: set[str] | None = None
    if confidence_filter == "high":
        wanted_confidences = {"high"}
    elif confidence_filter == "medium":
        wanted_confidences = {"medium"}
    elif confidence_filter == "low":
        wanted_confidences = {"low"}
    elif confidence_filter == "medium+low (active learning)":
        wanted_confidences = {"medium", "low"}

    wanted_classes: set[str] | None = None
    if class_filter != "any":
        wanted_classes = {class_filter}

    review_filter = None
    if wanted_confidences is not None or wanted_classes is not None:
        def review_filter(p):
            row = auto_index.get(p["frame_id"])
            if row is None:
                # Fallback: compute live (slow). Only happens if auto_labels.csv
                # is stale — re-run auto_classify_batch.py to refresh.
                wf = load_weak_labels(str(WEAK_LABELS_CSV), weak_mtime).get(p["frame_id"], {})
                mp, _, _ = thermal_cloud_stats(p["mask_path"])
                nrbr = rgb_nrbr_mean(p["rgb_path"], p["mask_path"])
                v_mean_live = rgb_v_mean(p["rgb_path"], p["mask_path"])
                cls, conf, _ = auto_classify(wf, thermal_mean_p=mp,
                                             rgb_nrbr_mean=nrbr,
                                             rgb_v_mean=v_mean_live)
                row = {"auto_class": cls, "auto_confidence": conf}
            if wanted_confidences is not None and row.get("auto_confidence") not in wanted_confidences:
                return False
            if wanted_classes is not None and row.get("auto_class") not in wanted_classes:
                return False
            return True

        n_match = sum(1 for p in pairs if review_filter(p))
        filter_desc = []
        if wanted_confidences is not None:
            filter_desc.append(f"conf={','.join(sorted(wanted_confidences))}")
        if wanted_classes is not None:
            filter_desc.append(f"class={','.join(sorted(wanted_classes))}")
        st.sidebar.caption(
            f"**Filter ({' · '.join(filter_desc)}): {n_match} of {len(pairs)} frames match** "
            f"({100 * n_match / max(len(pairs), 1):.1f}%)."
            + ("" if AUTO_LABELS_CSV.exists()
               else "  ⚠ `labels/auto_labels.csv` missing — run `python auto_classify_batch.py`.")
        )

    # Auto-advance: if the current frame doesn't match the filter (or is already
    # labeled when skip_labeled is on), jump forward to one that does. This
    # makes the dropdown feel responsive — changing it immediately jumps to a
    # matching frame instead of waiting for the user to click Next.
    def frame_passes(p):
        if skip_labeled and p["frame_id"] in labeled_ids:
            return False
        if review_filter and not review_filter(p):
            return False
        return True

    if not frame_passes(pairs[st.session_state.idx]):
        st.session_state.idx = advance(pairs, labeled_ids, +1,
                                       skip_labeled=skip_labeled,
                                       review_filter=review_filter)
        st.session_state.frame_started_at = time.time()

    pair = pairs[st.session_state.idx]
    ts = pair["timestamp"]
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "(no timestamp)"
    mean_p, frac_50, nodata_frac = thermal_cloud_stats(pair["mask_path"])

    # Compute auto-label using all weak labels + local thermal + RGB NRBR (day) + RGB V (night)
    # (weak_mtime hoisted above the sidebar so the review filter can use it too)
    weak_for_frame = load_weak_labels(str(WEAK_LABELS_CSV), weak_mtime).get(pair["frame_id"], {})
    nrbr = rgb_nrbr_mean(pair["rgb_path"], pair["mask_path"])
    v_mean = rgb_v_mean(pair["rgb_path"], pair["mask_path"])
    auto_label, auto_conf, auto_reason = auto_classify(
        weak_for_frame, thermal_mean_p=mean_p,
        rgb_nrbr_mean=nrbr, rgb_v_mean=v_mean,
    )

    st.subheader(f"{pair['frame_id']}  ·  {ts_str}")
    nav_l, nav_info, nav_r = st.columns([1, 4, 1])
    if nav_l.button("← Prev", use_container_width=True):
        st.session_state.idx = advance(pairs, labeled_ids, -1, skip_labeled, review_filter)
        st.session_state.frame_started_at = time.time()
        st.rerun()
    nav_info.markdown(
        f"**Frame {st.session_state.idx + 1} of {len(pairs)}**  ·  "
        f"thermal: mean p **{mean_p:.2f}**, "
        f"frac>0.5 **{frac_50 * 100:.0f}%**, "
        f"no-data **{nodata_frac * 100:.0f}%** (context only)"
    )
    if nav_r.button("Next →", use_container_width=True):
        st.session_state.idx = advance(pairs, labeled_ids, +1, skip_labeled, review_filter)
        st.session_state.frame_started_at = time.time()
        st.rerun()

    # Auto-classifier verdict — surfaces the rule-based pre-label so labelers
    # mostly verify (one click) rather than annotate from scratch.
    auto_cols = st.columns([1, 1, 6])
    auto_cols[0].metric("Auto-label", auto_label.upper(),
                        help="Rule-based verdict from weak labels (see auto_classify.py)")
    conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(auto_conf, "")
    auto_cols[1].metric("Confidence", f"{conf_color} {auto_conf}")
    auto_cols[2].markdown(
        f"_**Reasoning:** {auto_reason}_  \n"
        f"_The Class radio below is pre-selected to this verdict; verify or override._"
    )

    render_context_panel(weak_for_frame)

    full_allsky_path = find_full_allsky_path(pair["frame_id"])
    full_cols = st.columns([1, 2, 1])
    with full_cols[1]:
        if full_allsky_path:
            st.image(
                full_allsky_path,
                caption="FULL all-sky fisheye (whole hemisphere) — use this for genus + multi-cloud context",
                use_container_width=True,
            )
        else:
            st.caption(f"Full all-sky not found on NAS (looked under {ALLSKY_ROOT}/images/<date>/).")

    st.markdown("**Thermal-aligned patch (the slice your MLX90640 actually observes):**")
    img_cols = st.columns(3)
    with img_cols[0]:
        st.image(pair["rgb_path"], caption="RGB crop", use_container_width=True)
    with img_cols[1]:
        thr_str = f" @ p≥{overlay_threshold:.2f}" if overlay_style != "soft" else ""
        st.image(
            make_overlay(pair["rgb_path"], pair["mask_path"],
                         colormap=colormap_name,
                         style=overlay_style,
                         threshold=overlay_threshold),
            caption=f"Crop + thermal overlay — {overlay_style}{thr_str} ({colormap_name})",
            use_container_width=True,
        )
    with img_cols[2]:
        st.image(
            colorize_mask(pair["mask_path"], colormap=colormap_name),
            caption=f"Cloud probability heatmap ({colormap_name}) — grey diagonal stripe = no-data",
            use_container_width=True,
        )

    existing_row = labels_df[labels_df["frame_id"] == pair["frame_id"]]
    has_existing = len(existing_row) > 0
    if has_existing:
        ex = existing_row.iloc[0]
        st.info(
            f"Already labeled as **{ex['class']}** ({ex['confidence']}) "
            f"by `{ex['labeler_id']}` at {ex['labeled_at']}. Re-saving will overwrite."
        )
    else:
        ex = None

    def existing_index(values: list[str], col: str, default: int = 0) -> int:
        if not has_existing:
            return default
        v = str(ex[col]) if ex is not None else None
        return values.index(v) if v in values else default

    # Default class = existing hand label if present, else auto_label
    default_class_idx = (existing_index(CLASSES, "class") if has_existing
                        else (CLASSES.index(auto_label) if auto_label in CLASSES else 0))
    default_conf_idx = (existing_index(CONFIDENCES, "confidence") if has_existing
                       else (CONFIDENCES.index(auto_conf) if auto_conf in CONFIDENCES else 0))

    st.subheader("Label")
    form_cols = st.columns([3, 2])
    with form_cols[0]:
        cls = st.radio(
            "Class (pre-selected from auto-label)",
            CLASSES,
            index=default_class_idx,
            format_func=lambda c: f"{c} — {CLASS_DESCRIPTIONS[c]}",
        )
        conf = st.radio(
            "Confidence", CONFIDENCES,
            index=default_conf_idx,
            horizontal=True,
        )
        notes = st.text_input(
            "Notes (optional)",
            value=str(ex["notes"]) if has_existing and not pd.isna(ex["notes"]) else "",
        )
    with form_cols[1]:
        st.markdown("**QC flags**")
        qc_state: dict[str, bool] = {}
        for flag in QC_FLAGS:
            default = bool(ex[flag]) if has_existing and str(ex[flag]).lower() in {"true", "1"} else False
            qc_state[flag] = st.checkbox(flag, value=default, key=f"qc_{flag}_{pair['frame_id']}")

    save_cols = st.columns([1, 1, 2])
    save = save_cols[0].button("Save", use_container_width=True)
    save_next = save_cols[1].button("Save & Next", type="primary", use_container_width=True)
    save_cols[2].caption("Tip: enable 'Skip already-labeled' in the sidebar to walk straight through the unlabeled pool.")

    if save or save_next:
        if not st.session_state.labeler_id.strip():
            st.error("Set a Labeler ID in the sidebar before saving.")
            return
        seconds = round(time.time() - st.session_state.frame_started_at, 1)
        row = {
            "frame_id": pair["frame_id"],
            "rgb_path": pair["rgb_path"],
            "mask_path": pair["mask_path"],
            "timestamp": ts.isoformat() if ts else "",
            "class": cls,
            "confidence": conf,
            "labeler_id": st.session_state.labeler_id.strip(),
            "labeled_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "labeling_seconds": seconds,
            **qc_state,
            "notes": notes,
        }
        save_label(row)
        st.toast(f"Saved {pair['frame_id']} as {cls} ({conf}) — {seconds}s", icon="✅")
        if save_next:
            st.session_state.idx = advance(pairs, labeled_ids | {pair["frame_id"]}, +1, skip_labeled)
            st.session_state.frame_started_at = time.time()
            st.rerun()


if __name__ == "__main__":
    main()
