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
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
REGIMES = ["DAY", "TWILIGHT", "NAUTICAL", "ASTRO", "DARK"]
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

# Frame filenames (ccd1_YYYYMMDD_HHMMSS) are in the camera host's local
# wall-clock time. ALLSKY_LOCAL_TZ should match what the fetchers use so the
# UTC timestamps displayed here line up with weak_labels.csv.
def _resolve_local_tz() -> ZoneInfo:
    name = os.environ.get("ALLSKY_LOCAL_TZ", "UTC")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        sys.exit(f"ALLSKY_LOCAL_TZ={name!r} is not a known IANA timezone")

LOCAL_TZ = _resolve_local_tz()

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
    """Return the real UTC datetime for a frame, decoding the filename time as
    LOCAL_TZ. Use parse_local_date() when you need the local capture day for
    UI grouping — converting via UTC would split evening frames into the next
    date and surprise the user."""
    m = re.search(r"(\d{8}_\d{6})", stem)
    if not m:
        return None
    try:
        ts_local = dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=LOCAL_TZ)
        return ts_local.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def parse_local_date(stem: str) -> str | None:
    """Return the filename's local YYYYMMDD prefix without any TZ conversion.
    The right tool for grouping frames by 'capture day' in the UI."""
    m = re.search(r"(\d{8})_\d{6}", stem)
    return m.group(1) if m else None


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
def index_full_allsky(date_yyyymmdd: str, allsky_root: str,
                      dir_mtime: float) -> dict[str, str]:
    """Build {frame_id: absolute_path} for the full-fisheye captures of one day.
    Cached per-day; `dir_mtime` participates in the cache key so the index
    auto-invalidates when new files land on the NAS."""
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
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    for d in (day, prev_day):
        # Cache key: combine date with a freshness signal. For old days, use
        # the day-dir mtime (stable). For today/yesterday, use a per-minute
        # bucket so cache invalidates every minute as new captures land.
        day_root = ALLSKY_ROOT / "images" / d
        if d in (today, prev_day) and d == today:
            # Live-ingest day: cache for ~1 minute then refresh
            freshness = int(dt.datetime.now(dt.timezone.utc).timestamp() / 60)
        else:
            freshness = day_root.stat().st_mtime if day_root.is_dir() else 0.0
        idx = index_full_allsky(d, str(ALLSKY_ROOT), freshness)
        if frame_id in idx:
            return idx[frame_id]
    return None


@st.cache_data(show_spinner=False)
def load_hand_labels_index(csv_path: str, mtime: float) -> dict[str, dict]:
    """Returns {frame_id: row_dict} for fast hand-class lookups in filters.
    Separate from load_labels() (which returns a DataFrame for the distribution
    chart) — this one is a dict so compute_matching_ids stays O(1) per frame.
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
    lux = val("esp32_sensor", "illuminance_lux")
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
        headline_cols[1].metric("Lux (higher=cloudier)", f"{lux:.3f}" if lux is not None else "—",
                                help="ESP Lux — clouds over Calgary reflect city skyglow back, increasing lux")
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
        regime_note = "Nighttime: RGB is moonless or marginal — trust thermal + lux. Cu/Sc invisible without moonlight."
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


def render_decision_tree(frame_id: str, regime: str,
                         auto_label: str, auto_conf: str) -> dict | None:
    """Interactive classification helper. Returns {class, reasoning} or None
    if the user hasn't answered enough questions to reach a leaf.

    The tree is a guided version of docs/labeling-protocol.md §"Genus decision":
    coverage → texture → (cell size | sheet density), with two cross-checks
    against metadata at the end (auto-classifier agreement; regime sanity).

    Widget keys are frame-scoped so navigating to the next frame doesn't
    carry over the previous frame's answers.
    """
    def k(name: str) -> str:
        return f"dt_{name}_{frame_id}"

    suggestion: dict | None = None

    coverage = st.radio(
        "**1.** How much of the visible sky has cloud?",
        ["—",
         "Mostly clear (>90% cloud-free, only isolated wisps)",
         "Partial coverage — clear blue/dark sky visible between clouds",
         "Full coverage — no clear sky gaps anywhere",
         "Two distinct decks visible (different cloud bases at once)"],
        key=k("coverage"), index=0,
    )

    if coverage.startswith("Mostly clear"):
        suggestion = {"class": "clear",
                      "reasoning": "You reported >90% cloud-free sky"}
    elif coverage.startswith("Two distinct"):
        suggestion = {"class": "multi",
                      "reasoning": "You reported two distinct cloud decks"}
    elif coverage in ("Partial coverage — clear blue/dark sky visible between clouds",
                     "Full coverage — no clear sky gaps anywhere"):
        texture = st.radio(
            "**2.** What's the dominant cloud texture?",
            ["—",
             "Wispy / fibrous / streaky lines (like mare's tails)",
             "Cellular / lumpy / rippled (visible repeating units)",
             "Smooth uniform sheet (no internal structure)",
             "Discrete puffs with flat bases (cumuliform)",
             "Dark deep cloud with falling precipitation visible"],
            key=k("texture"), index=0,
        )

        if texture.startswith("Wispy"):
            suggestion = {"class": "ci",
                          "reasoning": "Fibrous streaks → Cirrus"}
        elif texture.startswith("Dark deep"):
            suggestion = {"class": "ns_cb",
                          "reasoning": "Precipitation visible → Nimbostratus / Cumulonimbus"}
        elif texture.startswith("Discrete puffs"):
            suggestion = {"class": "cu",
                          "reasoning": "Convective puffs with flat bases → Cumulus"}
        elif texture.startswith("Cellular"):
            cells = st.radio(
                "**3.** Cell size near the **zenith** (ignore horizon foreshortening):",
                ["—",
                 "Fine mackerel ripples (>50 cells across the sky)",
                 "Medium cells (~20 across, roughly 3–5° each)",
                 "Large lumpy patches (<10 across the sky, ≥5° each)"],
                key=k("cells"), index=0,
            )
            if cells.startswith("Fine"):
                suggestion = {"class": "cs_cc",
                              "reasoning": "Fine ripples → Cirrocumulus (high)"}
            elif cells.startswith("Medium"):
                suggestion = {"class": "ac_as",
                              "reasoning": "Mid-sized cells → Altocumulus (mid)"}
            elif cells.startswith("Large"):
                suggestion = {"class": "sc",
                              "reasoning": "Large lumpy patches → Stratocumulus (low)"}
        elif texture.startswith("Smooth uniform"):
            sheet = st.radio(
                "**3.** Sheet character:",
                ["—",
                 "Thin milky sheet — sun visible as a clear disc through it",
                 "Mid-grey sheet — sun barely visible / appears hazy",
                 "Low uniform grey lid — sun completely blocked or absent"],
                key=k("sheet"), index=0,
            )
            if sheet.startswith("Thin milky"):
                suggestion = {"class": "cs_cc",
                              "reasoning": "High thin uniform sheet → Cirrostratus"}
            elif sheet.startswith("Mid-grey"):
                suggestion = {"class": "ac_as",
                              "reasoning": "Mid grey sheet → Altostratus"}
            elif sheet.startswith("Low uniform"):
                suggestion = {"class": "st",
                              "reasoning": "Uniform low lid → Stratus"}

    # ---- present suggestion + cross-checks ----
    if suggestion is None:
        st.caption("_Answer the questions above to get a suggested class._")
        return None

    cls = suggestion["class"]

    # Regime sanity checks
    warnings: list[str] = []
    if cls == "cu" and regime not in ("DAY", "TWILIGHT"):
        warnings.append(
            f"Cu requires convective heating but regime is **{regime}** (sun "
            "below horizon). Often these turn out to be Sc fragments or "
            "broken Ac. Re-check before saving."
        )
    if cls == "ns_cb" and regime not in ("DAY", "TWILIGHT"):
        warnings.append(
            f"Ns/Cb is hard to confirm at night ({regime}). If you can't see "
            "lightning or hear thunder, consider Sc/St instead."
        )
    if cls == "st" and regime in ("NAUTICAL", "ASTRO", "DARK"):
        warnings.append(
            f"St at {regime} is climatologically unusual (St is typically a "
            "fog/morning-overcast cloud). Make sure it's a uniform grey lid "
            "and not just dim sky."
        )

    cols = st.columns([2, 2, 3])
    cols[0].markdown(f"### 🧭 Decision tree\n**{cls.upper()}**")
    cols[1].markdown(f"### 🤖 Auto-classifier\n**{auto_label.upper()}** ({auto_conf})")
    with cols[2]:
        if cls == auto_label:
            st.success(f"✓ Agreement — both say **{cls}**")
        else:
            st.warning(f"⚠ Disagreement — tree says **{cls}**, auto says "
                       f"**{auto_label}**. You're the tiebreaker.")
        st.caption(f"_Reasoning:_ {suggestion['reasoning']}")
        for w in warnings:
            st.warning(w)

    return suggestion


CLEANUP_LOG_CSV = PROJECT_ROOT / "labels" / "mask_cleanups.csv"
CLEANUP_LOG_COLUMNS = [
    "frame_id", "timestamp", "threshold", "neighborhood",
    "n_pixels_marked", "n_pixels_valid_before", "labeler_id",
]
AUTO_LABEL_FIELDS = [
    "frame_id", "auto_class", "auto_confidence", "auto_reasoning",
    "thermal_mean_p", "thermal_std", "rgb_nrbr_p95",
    "rgb_v_mean", "rgb_v_std", "alignment_mi", "computed_at",
]


def _clean_anomalous_pixels(mask: np.ndarray, threshold: float = 0.3,
                            neighborhood: int = 5) -> tuple[np.ndarray, int]:
    """Find sensor contamination via two stages:
      1. SEEDS — pixels that exceed their local 5×5 median by `threshold`
         probability units. Robust to isolated specks.
      2. REGION GROW — each seed anchors its connected component of "bright
         vs interior" pixels, catching contamination CLUSTERS that the
         median-filter step alone misses (when neighbours are also bright,
         the local median is itself elevated, so cluster bodies hide).
    Growth is anchored: a connected component is only marked if it contains
    a seed, so isolated real-cloud cells with no seed never get expanded.

    Returns (cleaned_mask, n_pixels_marked).
    """
    valid = mask != NO_DATA_VALUE
    if not valid.any():
        return mask.copy(), 0
    p = mask.astype(np.float32) / 254.0
    fill = float(np.median(p[valid]))
    p_filled = np.where(valid, p, fill).astype(np.float32)
    ksize = neighborhood if neighborhood in (3, 5) else 5
    local_median = cv2.medianBlur(p_filled, ksize)
    seeds = (p - local_median > threshold) & valid

    if not seeds.any():
        return mask.copy(), 0

    # Interior baseline: median of valid pixels far from any seed (dilate
    # seeds by 11px first so the baseline isn't contaminated by the cluster
    # itself). Falls back to the global valid median if everything is
    # near a seed.
    seeds_u8 = seeds.astype(np.uint8)
    seed_halo = cv2.dilate(seeds_u8, np.ones((11, 11), np.uint8)).astype(bool)
    interior = valid & ~seed_halo
    interior_med = float(np.median(p[interior])) if interior.any() else fill

    # Grow: any valid pixel meaningfully brighter than the interior baseline
    # is a *candidate* for masking. Growth threshold is softer (half the
    # detection threshold) — once we have a real seed, we trust pixels much
    # closer to seed brightness to belong to the same artifact.
    growth_thr = threshold * 0.5
    candidates = (p > interior_med + growth_thr) & valid

    # Connected components on the candidate mask; keep only components that
    # contain at least one seed. cv2.connectedComponents uses 8-connectivity
    # by default when you pass connectivity=8.
    n_lbl, labels = cv2.connectedComponents(
        candidates.astype(np.uint8), connectivity=8)
    seed_components = np.unique(labels[seeds])
    seed_components = seed_components[seed_components != 0]  # drop background
    if seed_components.size == 0:
        # Shouldn't happen since seeds ⊂ candidates, but be defensive
        anchored = seeds
    else:
        anchored = np.isin(labels, seed_components) & valid

    cleaned = mask.copy()
    cleaned[anchored] = NO_DATA_VALUE
    return cleaned, int(anchored.sum())


def _log_cleanup(frame_id: str, threshold: float, n_marked: int,
                 n_valid_before: int, labeler_id: str) -> None:
    """Append a cleanup-audit row so the methodology section can cite which
    frames received treatment and at what threshold."""
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


def _reclassify_one(pair: dict) -> dict | None:
    """Recompute the auto_label row for a single frame and patch
    auto_labels.csv in place. Mirrors auto_classify_batch.main()'s per-frame
    logic but lives here so the labeling container doesn't need that module.
    Returns the new row dict, or None if auto_labels.csv doesn't exist yet."""
    if not AUTO_LABELS_CSV.exists():
        return None
    fid = pair["frame_id"]
    mask_path = pair["mask_path"]
    rgb_path = pair["rgb_path"]
    mean_p, _, _, std_p = thermal_cloud_stats(mask_path)
    nrbr_p95 = rgb_nrbr_p95(rgb_path, mask_path)
    v_mean, v_std = rgb_v_stats(rgb_path, mask_path)
    weak_mt = WEAK_LABELS_CSV.stat().st_mtime if WEAK_LABELS_CSV.exists() else 0.0
    weak = load_weak_labels(str(WEAK_LABELS_CSV), weak_mt).get(fid, {})
    # Alignment MI sidecar — best-effort, optional
    align_mi: float | None = None
    meta_p = Path(mask_path).parent.parent / "meta" / f"{Path(mask_path).stem}.json"
    if meta_p.exists():
        try:
            import json
            v = json.loads(meta_p.read_text()).get("alignment_mi")
            align_mi = float(v) if v is not None else None
        except (ValueError, TypeError, OSError):
            align_mi = None

    cls, conf, reasoning = auto_classify(
        weak, thermal_mean_p=mean_p, rgb_nrbr_p95=nrbr_p95,
        rgb_v_mean=v_mean, rgb_v_std=v_std, thermal_std=std_p,
    )
    new_row = {
        "frame_id": fid,
        "auto_class": cls,
        "auto_confidence": conf,
        "auto_reasoning": reasoning,
        "thermal_mean_p": f"{mean_p:.3f}" if mean_p == mean_p else "",
        "thermal_std": f"{std_p:.3f}" if std_p == std_p else "",
        "rgb_nrbr_p95": f"{nrbr_p95:+.3f}" if nrbr_p95 is not None else "",
        "rgb_v_mean": f"{v_mean:.1f}" if v_mean is not None else "",
        "rgb_v_std": f"{v_std:.1f}" if v_std is not None else "",
        "alignment_mi": f"{align_mi:.3f}" if align_mi is not None else "",
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    # Read-modify-write the CSV. <100k rows so this is fine.
    with open(AUTO_LABELS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or AUTO_LABEL_FIELDS
        rows = list(reader)
    found = False
    for i, r in enumerate(rows):
        if r["frame_id"] == fid:
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


def render_cleanup_panel(pair: dict, current_mean_p: float, colormap_name: str) -> None:
    """Per-frame sensor-contamination cleanup. Lets the labeler preview a
    median-filter outlier removal, then save the cleaned mask (with backup)
    and trigger a re-classification of just this frame.

    Hidden by default — only worth opening when a frame's heatmap shows
    isolated bright specks that don't look like real cloud.
    """
    with st.expander("Mask cleanup (remove sensor contamination)", expanded=False):
        st.caption(
            "Pixels much brighter than their local 5×5 neighborhood are likely "
            "sensor contamination (warm dust, lens residue), not real cloud. "
            "Real cloud has spatial coherence — neighbouring pixels also bright."
        )
        threshold = st.slider(
            "Outlier threshold (probability units above local median)",
            min_value=0.10, max_value=0.60, value=0.30, step=0.05,
            key=f"cleanup_thr_{pair['frame_id']}",
            help="Higher = only mask obvious outliers (spares cloud edges). "
                 "Lower = aggressive (may clip real broken cloud).",
        )
        preview_clicked = st.button("Preview cleanup", key=f"cleanup_prev_{pair['frame_id']}")
        state_key = f"cleanup_pending_{pair['frame_id']}"
        if preview_clicked:
            raw_mask = np.array(Image.open(pair["mask_path"]).convert("L"))
            cleaned, n_marked = _clean_anomalous_pixels(raw_mask, threshold=threshold)
            st.session_state[state_key] = {
                "cleaned": cleaned,
                "raw": raw_mask,
                "n_marked": n_marked,
                "threshold": threshold,
            }

        pending = st.session_state.get(state_key)
        if pending is None:
            return

        cleaned = pending["cleaned"]
        raw_mask = pending["raw"]
        n_marked = pending["n_marked"]
        n_valid_before = int((raw_mask != NO_DATA_VALUE).sum())

        # Side-by-side heatmaps + new thermal_mean_p
        col_b, col_a = st.columns(2)
        with col_b:
            st.caption(f"Before — thermal_mean_p = {current_mean_p:.3f}")
            st.image(colorize_mask(pair["mask_path"], colormap=colormap_name),
                     use_container_width=True)
        valid_after = cleaned != NO_DATA_VALUE
        if valid_after.any():
            new_mean = float((cleaned[valid_after].astype(np.float32) / 254.0).mean())
            new_mean_str = f"{new_mean:.3f}"
        else:
            new_mean_str = "n/a (no valid pixels left)"
        with col_a:
            st.caption(f"After — thermal_mean_p = {new_mean_str}  ·  {n_marked} px masked")
            # colorize_mask reads from disk, so we render the cleaned array directly
            st.image(_colorize_array(cleaned, colormap_name),
                     use_container_width=True)

        if n_marked == 0:
            st.info("No pixels exceeded the threshold — no cleanup needed at this setting.")
            return

        action_cols = st.columns([1, 1, 2])
        save_clicked = action_cols[0].button(
            "✓ Save cleaned mask",
            key=f"cleanup_save_{pair['frame_id']}",
            type="primary",
            help="Backs up the original to <frame>.original.png, overwrites the "
                 "mask, re-runs auto_classify on this frame.",
        )
        cancel_clicked = action_cols[1].button(
            "Cancel", key=f"cleanup_cancel_{pair['frame_id']}",
        )

        if cancel_clicked:
            del st.session_state[state_key]
            st.rerun()

        if save_clicked:
            labeler_id = st.session_state.get("labeler_id", "").strip()
            if not labeler_id:
                st.error("Set a Labeler ID in the sidebar before saving cleanup.")
                return
            mask_path = pair["mask_path"]
            # cv2.imwrite infers format from the extension, so the backup
            # must end in .png — use <stem>.original.png, not <stem>.png.original.
            backup_path = str(Path(mask_path).with_suffix(".original.png"))
            # Backup only on first cleanup — keep the truly pristine version.
            if not Path(backup_path).exists():
                cv2.imwrite(backup_path, raw_mask)
            cv2.imwrite(mask_path, cleaned)
            _log_cleanup(pair["frame_id"], pending["threshold"], n_marked,
                         n_valid_before, labeler_id)
            new_row = _reclassify_one(pair)
            # Clear caches so the next render reads the new mask + auto_label.
            thermal_cloud_stats.clear() if hasattr(thermal_cloud_stats, "clear") else None
            try:
                st.cache_data.clear()
            except Exception:
                pass
            del st.session_state[state_key]
            verdict = (f" → auto-label now **{new_row['auto_class']}** "
                       f"({new_row['auto_confidence']})") if new_row else ""
            st.toast(f"Cleaned {pair['frame_id']}: {n_marked} px masked{verdict}",
                     icon="🧹")
            st.rerun()


def _colorize_array(raw: np.ndarray, colormap: str) -> np.ndarray:
    """Same as colorize_mask but takes an in-memory array (not a path).
    Lets the cleanup preview render the cleaned mask without touching disk."""
    valid = raw != NO_DATA_VALUE
    probs_u8 = np.where(valid, raw, 0).astype(np.uint8)
    lut = COLORMAPS.get(colormap, SKY_CLOUD_LUT)
    colored = lut[probs_u8]
    yy, xx = np.indices(raw.shape)
    stripe = ((yy + xx) // 6) % 2 == 0
    colored[(~valid) & stripe] = (60, 60, 60)
    colored[(~valid) & ~stripe] = (100, 100, 100)
    return colored


def gradient_view(mask_path: str, colormap: str = "sky_cloud (custom)",
                  saturation: float = 0.3) -> np.ndarray:
    """Sobel gradient magnitude of the cloud probability mask.

    Highlights cloud EDGES rather than cloud INTENSITY. Useful for:
      - Multi-cloud detection — sharp boundaries between distinct cloud regions
        (e.g., Cu cells against background Ac sheet) light up strongly.
      - Genus disambiguation — convective genera (Cu, Cb) have sharp boundaries
        and high gradient; stratiform (St, Ns) have gradual transitions and low
        gradient even at high mean_p.
      - QC — a "speckly" gradient over an otherwise uniform region often
        indicates residual sensor noise or sub-detection contamination.

    Args:
        saturation: gradient magnitude (prob/pixel) at which the colormap
            saturates. 0.3 = moderately strong edges saturate; lower values
            amplify subtle texture, higher values suppress it.
    """
    raw = np.array(Image.open(mask_path).convert("L"))
    valid = raw != NO_DATA_VALUE
    probs = np.where(valid, raw, 0).astype(np.float32) / 254.0

    gx = cv2.Sobel(probs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(probs, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Suppress edges along the no-data boundary itself — those are artifacts
    # of the patch border, not real cloud edges. Erode the valid mask by 1px
    # and only keep gradient values inside that eroded region.
    valid_inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    mag = np.where(valid_inner, mag, 0.0)

    mag_norm = np.clip(mag / max(saturation, 1e-6), 0.0, 1.0)
    mag_u8 = (mag_norm * 254).astype(np.uint8)
    lut = COLORMAPS.get(colormap, SKY_CLOUD_LUT)
    colored = lut[mag_u8]

    # Match the heatmap's no-data stripe pattern for visual consistency
    yy, xx = np.indices(raw.shape)
    stripe = ((yy + xx) // 6) % 2 == 0
    colored[(~valid) & stripe] = (60, 60, 60)
    colored[(~valid) & ~stripe] = (100, 100, 100)
    return colored


def colorize_mask(mask_path: str, colormap: str = "sky_cloud (custom)") -> np.ndarray:
    raw = np.array(Image.open(mask_path).convert("L"))
    valid = raw != NO_DATA_VALUE
    # Legacy (matched_crop-era) masks are true 0/1 binary. Detect ONLY that.
    # The old `len(np.unique(raw)) <= 2` test also fired on a fully-overcast v2
    # mask — whose pixels are just {254 cloud, 255 no-data}, i.e. 2 values —
    # which then erased the no-data corners and flattened the heatmap to solid
    # red. 255 is always no-data under the v2 convention; honor it unconditionally.
    legacy_binary = raw.max() <= 1
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


def render_mask_legend(colormap_name: str,
                       overlay_style: str,
                       overlay_threshold: float,
                       show_gradient: bool,
                       gradient_saturation: float) -> None:
    """Color-scale legend for the mask panels.

    Renders the *currently selected* colormap as a gradient bar with labelled
    endpoints, a threshold marker (for hard/contour overlays), and a no-data
    swatch matching the diagonal grey stripe used in the panels. Adapts its
    wording for the Sobel edge-gradient view, whose values mean "magnitude of
    cloud-probability change" rather than "cloud probability".
    """
    lut = COLORMAPS.get(colormap_name, SKY_CLOUD_LUT)
    # Sample the LUT into CSS gradient stops so the bar matches the panels
    # exactly, whatever colormap is chosen.
    n = 24
    stops = []
    for i in range(n + 1):
        x = i / n
        r, g, b = (int(c) for c in lut[int(round(x * 255))])
        stops.append(f"rgb({r},{g},{b}) {x * 100:.1f}%")
    gradient = "linear-gradient(to right, " + ", ".join(stops) + ")"
    # No-data swatch: same alternating mid-greys as the panel stripe.
    nodata = ("repeating-linear-gradient(45deg, rgb(60,60,60) 0 4px, "
              "rgb(100,100,100) 4px 8px)")

    marker = ""
    note = ""
    if show_gradient:
        title = "Sobel edge gradient — rate of change of cloud probability"
        left_lbl, mid_lbl = "uniform region", ""
        right_lbl = f"sharp edge (≥ {gradient_saturation:.2f}/px)"
    else:
        title = "Cloud probability (thermal mask)"
        left_lbl, mid_lbl, right_lbl = "0.0 · clear sky", "0.5", "1.0 · dense cloud"
        if overlay_style in ("hard", "contour"):
            pct = max(0.0, min(100.0, overlay_threshold * 100))
            marker = (f"<div style='position:absolute;top:-3px;bottom:-3px;"
                      f"left:{pct:.1f}%;width:2px;background:#fff;"
                      f"box-shadow:0 0 2px #000;'></div>")
            note = (f"<div style='margin-top:3px;opacity:0.7;'>"
                    f"▲ white line = cloud threshold p ≥ {overlay_threshold:.2f} "
                    f"(used by the <b>{overlay_style}</b> overlay)</div>")

    html = f"""
    <div style="margin:0.1rem 0 0.6rem 0;font-size:0.8rem;line-height:1.3;">
      <div style="margin-bottom:4px;opacity:0.85;">{title}</div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div style="flex:1;min-width:220px;">
          <div style="position:relative;height:16px;border-radius:3px;
                      background:{gradient};
                      border:1px solid rgba(128,128,128,0.4);">{marker}</div>
          <div style="display:flex;justify-content:space-between;
                      margin-top:2px;opacity:0.8;">
            <span>{left_lbl}</span><span>{mid_lbl}</span><span>{right_lbl}</span>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:6px;white-space:nowrap;">
          <span style="display:inline-block;width:16px;height:16px;border-radius:3px;
                       background:{nodata};
                       border:1px solid rgba(128,128,128,0.4);"></span>
          <span style="opacity:0.8;">no-data (outside sensor FOV)</span>
        </div>
      </div>
      {note}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def thermal_cloud_stats(mask_path: str) -> tuple[float, float, float, float]:
    """Returns (mean_cloud_prob_over_valid, fraction_above_0.5_over_valid,
    no_data_fraction, std_cloud_prob_over_valid).

    The std is the spatial standard deviation — used by the auto-classifier's
    Rule 0 to detect broken/textured cloud regimes (Cu, broken Sc) that the
    scalar mean alone misses.
    """
    raw = np.array(Image.open(mask_path).convert("L"))
    if raw.max() <= 1 or len(np.unique(raw)) <= 2:
        binary = (raw > 127).astype(np.float32)
        return float(binary.mean()), float(binary.mean()), 0.0, float(binary.std())
    valid = raw != NO_DATA_VALUE
    if not valid.any():
        return float("nan"), float("nan"), 1.0, float("nan")
    probs = raw[valid].astype(np.float32) / 254.0
    return (float(probs.mean()), float((probs >= 0.5).mean()),
            float((~valid).mean()), float(probs.std()))


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


def rgb_nrbr_p95(rgb_path: str, mask_path: str) -> float | None:
    """95th percentile of (R-B)/(R+B) over thermal-valid pixels — daytime cloud peak cue."""
    rgb, valid = _rgb_and_valid(rgb_path, mask_path)
    if rgb is None or valid is None:
        return None
    r = rgb[..., 0][valid]
    b = rgb[..., 2][valid]
    nrbr = (r - b) / (r + b + 1e-6)
    return float(np.percentile(nrbr, 95))


def rgb_v_stats(rgb_path: str, mask_path: str) -> tuple[float | None, float | None]:
    """Mean and STD of HSV V over thermal-valid pixels — nighttime cloud cue."""
    rgb, valid = _rgb_and_valid(rgb_path, mask_path)
    if rgb is None or valid is None:
        return None, None
    v_channel = rgb.max(axis=-1)[valid]
    return float(v_channel.mean()), float(v_channel.std())


@st.cache_data(show_spinner=False)
def compute_matching_ids(
    frame_ids: tuple[str, ...],
    auto_csv: str, auto_mtime: float,
    weak_csv: str, weak_mtime: float,
    hand_csv: str, hand_mtime: float,
    wanted_confidences: frozenset[str] | None,
    wanted_classes: frozenset[str] | None,
    wanted_regimes: frozenset[str] | None,
    wanted_hand_classes: frozenset[str] | None,
    id_substring: str,
    date_prefix: str | None = None,
    day_night_mode: str = "any",
) -> frozenset[str]:
    """Pre-compute the set of frame_ids passing all active filters in one sweep.

    Pure dict lookups against the already-cached auto/weak/hand label indices —
    no PIL I/O, no live auto_classify fallback. Cached per (filter combo,
    csv mtimes) so it survives reruns until any indexed file changes on disk.
    """
    auto_needed = wanted_confidences is not None or wanted_classes is not None
    auto_index = load_auto_labels(auto_csv, auto_mtime) if auto_needed else {}
    weak_index = load_weak_labels(weak_csv, weak_mtime) if (wanted_regimes or day_night_mode != "any") else {}
    hand_index = load_hand_labels_index(hand_csv, hand_mtime) if wanted_hand_classes else {}
    sub = id_substring.lower().strip()
    out: set[str] = set()
    for fid in frame_ids:
        if date_prefix and date_prefix not in fid:
            continue
        if sub and sub not in fid.lower():
            continue
        if wanted_hand_classes is not None:
            hand_row = hand_index.get(fid)
            if hand_row is None or hand_row.get("class") not in wanted_hand_classes:
                continue
        if auto_needed:
            row = auto_index.get(fid)
            if row is None:
                continue
            if wanted_confidences is not None and row.get("auto_confidence") not in wanted_confidences:
                continue
            if wanted_classes is not None and row.get("auto_class") not in wanted_classes:
                continue

        # Sun regime filtering (either via explicit regimes or simple day/night toggle)
        if wanted_regimes is not None or day_night_mode != "any":
            wf = weak_index.get(fid, {})
            sun_alt_row = wf.get(("ephemeris", "sun_alt_deg"))
            if sun_alt_row is None:
                continue
            try:
                sun_alt = float(sun_alt_row["value"])
            except (TypeError, ValueError):
                continue
            
            regime = _sun_regime(sun_alt)
            if wanted_regimes is not None and regime not in wanted_regimes:
                continue
            if day_night_mode == "Day only" and regime != "DAY":
                continue
            if day_night_mode == "Night only" and regime == "DAY":
                continue

        out.add(fid)
    return frozenset(out)


@st.cache_data(show_spinner="Mining pre-rain candidates...")
def compute_rain_sessions(
    frame_ids: tuple[str, ...],
    weak_csv: str, weak_mtime: float,
    window_minutes: int = 60,
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Returns ((session_label, candidate_frame_ids), ...) — one entry per
    rain-onset event detected in the data.

    Each session_label is the onset timestamp formatted as "YYYY-MM-DD HH:MM UTC".
    candidate_frame_ids is the set of frames in (onset - window, onset) that
    show GOES cloud signature (cloud_present=1 + phase ice/mixed + top > 5 km).

    These are the high-value "ns_cb cloud overhead, lens still clean" frames
    for breaking the rain_on_lens confound — same physical regime as wet-lens
    ns_cb frames but without the lens artifact, so models trained on them
    learn cloud structure rather than raindrops on glass.

    Returns empty tuple if no rain events in the data.
    """
    import bisect
    weak = load_weak_labels(weak_csv, weak_mtime)

    # (timestamp, frame_id, rain_mm) sorted by time
    records: list[tuple[dt.datetime, str, float]] = []
    for fid in frame_ids:
        ts = parse_timestamp(fid)
        if ts is None:
            continue
        wf = weak.get(fid, {})
        rain_row = wf.get(("weather_station", "rain_1h_mm"))
        if rain_row is None:
            continue
        try:
            rain_mm = float(rain_row["value"])
        except (ValueError, TypeError):
            continue
        records.append((ts, fid, rain_mm))
    records.sort(key=lambda r: r[0])

    # Find rain-onset timestamps: prev ≤ 0.01 mm and current > 0.01 mm.
    # rain_1h_mm is cumulative over the past hour — the leading edge is
    # the first frame where it transitions to non-zero. Subsequent onsets
    # within the same physical storm get separate sessions only if rain
    # truly drops to zero between them (rare for sustained storms).
    THRESHOLD = 0.01
    BUFFER_MINS = 5  # Stop N mins before onset to ensure lens is truly dry
    onset_times = [
        records[i][0]
        for i in range(1, len(records))
        if records[i - 1][2] <= THRESHOLD and records[i][2] > THRESHOLD
    ]
    if not onset_times:
        return tuple()

    record_times = [r[0] for r in records]
    window = dt.timedelta(minutes=window_minutes)
    buffer = dt.timedelta(minutes=BUFFER_MINS)
    sessions: list[tuple[str, frozenset[str]]] = []
    for onset in onset_times:
        lo = bisect.bisect_left(record_times, onset - window)
        hi = bisect.bisect_left(record_times, onset - buffer)
        session_frames: set[str] = set()
        
        # Continuity: if GOES drops out for a few frames, assume the cloud 
        # is still there if it was just there. Reset for each session.
        last_valid_goes_passed = False
        
        for idx in range(lo, hi):
            fid = records[idx][1]
            wf = weak.get(fid, {})

            goes_present = wf.get(("goes19_acmc", "cloud_present"))
            goes_phase = wf.get(("goes19_actpc", "cloud_top_phase"), {}).get("value")
            goes_height_raw = wf.get(("goes19_achac", "cloud_top_height_m"))

            if goes_present is not None:
                # If we have GOES data, evaluate it strictly
                is_cloudy = str(goes_present.get("value")) == "1"
                
                height = -1.0
                try:
                    if goes_height_raw:
                        height = float(goes_height_raw["value"])
                except (ValueError, TypeError):
                    pass

                # Expanded criteria: 5km+ ice/mixed (Cb), or 3km+ mixed/supercooled (Ns)
                passed = is_cloudy and (
                    (height >= 5000 and goes_phase in ("ice", "mixed")) or
                    (height >= 3000 and goes_phase in ("mixed", "supercooled_water"))
                )
                last_valid_goes_passed = passed
            else:
                # GOES dropout: stick with the last known state
                pass

            if last_valid_goes_passed:
                session_frames.add(fid)

        if session_frames:
            label = onset.strftime("%Y-%m-%d %H:%M UTC")
            sessions.append((label, frozenset(session_frames)))
    return tuple(sessions)


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

    # weak_mtime is used by both the sidebar filter (pre-rain candidates) and
    # the per-frame block below. Defined once here so both share the same
    # cache key — background re-fetches invalidate both views together.
    weak_mtime = WEAK_LABELS_CSV.stat().st_mtime if WEAK_LABELS_CSV.exists() else 0.0

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

        # Date filter — group by local capture day (from filename prefix), not
        # by UTC, so evening frames don't split across two dates in the UI.
        # date_prefix is later substring-matched against frame_id, which uses
        # the same local YYYYMMDD prefix, so the comparison stays consistent.
        all_dates = sorted({d for p in pairs if (d := parse_local_date(p["frame_id"]))}, reverse=True)
        date_filter = st.selectbox(
            "Date",
            ["any"] + all_dates,
            index=0,
            help="Filter by observation date (YYYYMMDD)."
        )

        # Day/Night filter
        day_night_filter = st.selectbox(
            "Day / Night",
            ["any", "Day only", "Night only"],
            index=0,
            help="Quick filter for day vs night. 'Day' = sun > 6°, 'Night' = sun < 6° (includes twilight/dark)."
        )

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
            ["any"] + CLASSES + ["unknown"],
            index=0,
            help=(
                "Narrow to frames where the auto-classifier predicted a specific "
                "class. **any** = any of the 9 taxonomy classes (excludes unknown). "
                "**unknown** = classifier punted (signal disagreement). "
                "Pairs with the confidence filter — e.g. **any + low** is your "
                "active-learning queue for low-confidence taxonomy predictions."
            ),
        )
        regime_filter = st.multiselect(
            "Sun regime",
            REGIMES,
            default=[],
            help=(
                "Filter by solar regime (from ephemeris.sun_alt_deg). "
                "**DAY** sun≥6°, **TWILIGHT** −6°..6°, **NAUTICAL** −12°..−6°, "
                "**ASTRO** −18°..−12°, **DARK** sun<−18°. Empty = any. "
                "Multi-select to audit \"any night\" by picking "
                "TWILIGHT+NAUTICAL+ASTRO+DARK."
            ),
        )

        st.markdown("**Hand-label audit** — filter by your own labels")
        hand_class_filter = st.selectbox(
            "Hand-label class",
            ["any"] + CLASSES,
            index=0,
            help=(
                "Show only frames you already hand-labeled with this class. "
                "Combine with **Sun regime** to audit suspicious cases — e.g. "
                "`hand_class=cu` + `regime=ASTRO+DARK+NAUTICAL` finds cumulus "
                "labels at sun-below-horizon times (Cu needs convective heating "
                "and shouldn't form pre-dawn). Remember to also uncheck "
                "'Skip already-labeled' above so the matches actually appear."
            ),
        )
        frame_id_search = st.text_input(
            "Frame ID contains",
            value="",
            help="Substring match on frame_id (case-insensitive). Useful for "
                 "jumping to a specific date prefix like `20260519_10` or to "
                 "the exact frame from a disagreement report.",
        )

        st.markdown("**Special queues** — targeted labeling missions")
        pre_rain_only = st.checkbox(
            "Pre-rain ns_cb candidates (lens-clean)",
            value=False,
            help=(
                "Frames in the 60-minute window BEFORE a rain-onset event AND "
                "with GOES already showing high ice/mixed cloud overhead "
                "(top > 5 km). These are the rare 'ns_cb cloud overhead, lens "
                "still dry' frames — high-value for breaking the 100% "
                "ns_cb↔rain_on_lens correlation that would otherwise teach a "
                "CNN to predict ns_cb from raindrops-on-glass rather than "
                "cloud structure. Label these as `ns_cb` with "
                "`rain_on_lens=False` to give downstream models a clean "
                "training/eval subset."
            ),
        )
        pre_rain_window_min = st.number_input(
            "Pre-rain lookback (min)",
            min_value=10, max_value=240, value=60, step=10,
            disabled=not pre_rain_only,
            help="How far back from each rain-onset to search. Shorter = tighter "
                 "to onset (more chance lens is still dry). Longer = more "
                 "candidates but possibly less ns_cb-like cloud structure.",
        )

        # Session-level navigation: list each rain event so labeler can focus
        # on one storm at a time rather than seeing all sessions interleaved.
        rain_sessions: tuple[tuple[str, frozenset[str]], ...] = tuple()
        if pre_rain_only:
            rain_sessions = compute_rain_sessions(
                tuple(p["frame_id"] for p in pairs),
                str(WEAK_LABELS_CSV), weak_mtime,
                window_minutes=int(pre_rain_window_min),
            )
        session_options = ["all sessions"] + [
            f"{label}  ({len(frames)} frames)"
            for label, frames in rain_sessions
        ]
        if pre_rain_only and rain_sessions:
            st.caption(f"📊 Found {len(rain_sessions)} rain session(s), "
                       f"{sum(len(f) for _,f in rain_sessions)} pre-rain candidate frames total.")
        elif pre_rain_only:
            st.caption("📊 No rain sessions detected in the dataset.")
        rain_session_pick = st.selectbox(
            "Rain session",
            session_options,
            index=0,
            disabled=not pre_rain_only or not rain_sessions,
            help="Pick a specific storm to focus on, or 'all sessions' to see "
                 "candidates from every rain event merged together. Each "
                 "label is the rain-onset timestamp; the lens-clean window "
                 "ends at that timestamp.",
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
        show_gradient = st.checkbox(
            "Edge gradient view (Sobel)",
            value=False,
            help=(
                "Replaces the cloud-probability heatmap with the Sobel gradient "
                "magnitude. Highlights cloud EDGES (sharp transitions) rather "
                "than cloud intensity. Useful for spotting multi-cloud frames "
                "(distinct cloud regions show sharp boundaries) and "
                "distinguishing convective cells (sharp edges) from stratiform "
                "sheets (gradual transitions)."
            ),
        )
        gradient_saturation = st.slider(
            "Gradient saturation",
            min_value=0.10, max_value=0.80, value=0.30, step=0.05,
            help="Gradient value (prob/pixel) at which the colormap saturates. "
                 "Lower = amplify subtle texture; higher = only show strong edges.",
        ) if show_gradient else 0.30

    # (weak_mtime hoisted to top of main() — used by both sidebar filter
    # and per-frame block.)

    with st.sidebar:

        st.divider()
        st.subheader("Jump")
        
        if st.button("🎲 Random Unlabeled (balanced)", use_container_width=True,
                     help="Picks an unlabeled frame, weighted toward auto-classes "
                          "you've hand-confirmed least often — so the queue fills "
                          "the gaps in your label distribution instead of "
                          "oversampling whatever the auto-classifier sees most."):
            import random
            from collections import Counter, defaultdict

            random_auto_mtime = (
                AUTO_LABELS_CSV.stat().st_mtime if AUTO_LABELS_CSV.exists() else 0.0
            )
            random_auto_index = load_auto_labels(
                str(AUTO_LABELS_CSV), random_auto_mtime
            )

            buckets: dict[str, list[int]] = defaultdict(list)
            for i, p in enumerate(pairs):
                if p["frame_id"] in labeled_ids:
                    continue
                ac = random_auto_index.get(p["frame_id"], {}).get("auto_class") \
                    or "_unknown"
                buckets[ac].append(i)

            if not buckets:
                st.sidebar.success("All frames have been labeled!")
            else:
                hand_counts = Counter(labels_df["class"].astype(str))
                # Inverse-frequency weight: a class with 0 hand labels gets
                # weight 1.0; one with 50 gets ~0.02. The +1 keeps it finite.
                weights = {
                    cls: 1.0 / (1 + hand_counts.get(cls, 0))
                    for cls in buckets
                }
                chosen = random.choices(
                    list(weights.keys()),
                    weights=list(weights.values()),
                    k=1,
                )[0]
                st.session_state.idx = random.choice(buckets[chosen])
                st.session_state.frame_started_at = time.time()
                st.rerun()

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

    # Translate the dropdowns into frozensets (hashable for cache key).
    wanted_confidences: frozenset[str] | None = None
    if confidence_filter == "high":
        wanted_confidences = frozenset({"high"})
    elif confidence_filter == "medium":
        wanted_confidences = frozenset({"medium"})
    elif confidence_filter == "low":
        wanted_confidences = frozenset({"low"})
    elif confidence_filter == "medium+low (active learning)":
        wanted_confidences = frozenset({"medium", "low"})

    wanted_classes: frozenset[str] | None = None
    if class_filter == "any":
        # "any" = any of the 9 taxonomy classes; excludes "unknown" so the
        # active-learning queue (e.g. "any class + confidence=low") doesn't
        # get polluted by classifier punts. Pick "unknown" explicitly to see those.
        wanted_classes = frozenset(CLASSES)
    else:
        wanted_classes = frozenset({class_filter})

    wanted_regimes: frozenset[str] | None = frozenset(regime_filter) if regime_filter else None
    wanted_hand_classes: frozenset[str] | None = (
        None if hand_class_filter == "any" else frozenset({hand_class_filter})
    )
    id_substring = frame_id_search.strip()

    hand_mtime = LABELS_CSV.stat().st_mtime if LABELS_CSV.exists() else 0.0

    review_filter = None
    any_filter_active = (wanted_confidences is not None or wanted_classes is not None
                         or wanted_regimes is not None or wanted_hand_classes is not None
                         or bool(id_substring) or date_filter != "any" or day_night_filter != "any"
                         or pre_rain_only)
    if any_filter_active:
        # One sweep over the cached auto/weak/hand indices. No PIL I/O, no
        # live auto_classify fallback — frames missing from auto_labels.csv
        # are simply excluded (the user is told to re-run auto_classify_batch.py).
        matching_ids = compute_matching_ids(
            tuple(p["frame_id"] for p in pairs),
            str(AUTO_LABELS_CSV), auto_mtime,
            str(WEAK_LABELS_CSV), weak_mtime,
            str(LABELS_CSV), hand_mtime,
            wanted_confidences, wanted_classes, wanted_regimes,
            wanted_hand_classes, id_substring,
            date_prefix=(None if date_filter == "any" else date_filter),
            day_night_mode=day_night_filter,
        )

        if pre_rain_only and rain_sessions:
            # rain_sessions was already computed for the sidebar selector
            if rain_session_pick == "all sessions":
                pre_rain_ids = frozenset().union(*(f for _, f in rain_sessions))
            else:
                # Match by the label prefix (strip the "  (N frames)" suffix)
                picked_label = rain_session_pick.split("  (")[0]
                pre_rain_ids = next(
                    (frames for label, frames in rain_sessions if label == picked_label),
                    frozenset(),
                )
            matching_ids = matching_ids & pre_rain_ids
        elif pre_rain_only:
            # Checkbox on but no rain sessions found → empty queue
            matching_ids = frozenset()

        def review_filter(p, _ids=matching_ids):
            return p["frame_id"] in _ids

        n_match = len(matching_ids)
        filter_desc = []
        if date_filter != "any":
            filter_desc.append(f"date={date_filter}")
        if day_night_filter != "any":
            filter_desc.append(f"mode={day_night_filter}")
        if wanted_confidences is not None:
            filter_desc.append(f"auto_conf={','.join(sorted(wanted_confidences))}")
        if wanted_classes is not None:
            filter_desc.append(f"auto_class={','.join(sorted(wanted_classes))}")
        if wanted_regimes is not None:
            filter_desc.append(f"regime={','.join(sorted(wanted_regimes))}")
        if wanted_hand_classes is not None:
            filter_desc.append(f"hand_class={','.join(sorted(wanted_hand_classes))}")
        if id_substring:
            filter_desc.append(f"id~{id_substring!r}")
        if pre_rain_only:
            session_tag = (
                "all" if rain_session_pick == "all sessions"
                else rain_session_pick.split("  (")[0]
            )
            filter_desc.append(f"pre_rain≤{int(pre_rain_window_min)}min/{session_tag}")
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
    mean_p, frac_50, nodata_frac, std_p = thermal_cloud_stats(pair["mask_path"])

    # Compute auto-label using all weak labels + local thermal + RGB peak (day) + RGB V (night)
    # (weak_mtime hoisted above the sidebar so the review filter can use it too)
    weak_for_frame = load_weak_labels(str(WEAK_LABELS_CSV), weak_mtime).get(pair["frame_id"], {})
    nrbr_p95 = rgb_nrbr_p95(pair["rgb_path"], pair["mask_path"])
    v_mean, v_std = rgb_v_stats(pair["rgb_path"], pair["mask_path"])
    auto_label, auto_conf, auto_reason = auto_classify(
        weak_for_frame, thermal_mean_p=mean_p,
        rgb_nrbr_p95=nrbr_p95, rgb_v_mean=v_mean,
        rgb_v_std=v_std,
        thermal_std=std_p,
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
        if show_gradient:
            st.image(
                gradient_view(pair["mask_path"], colormap=colormap_name,
                              saturation=gradient_saturation),
                caption=(f"Sobel edge gradient ({colormap_name}, sat={gradient_saturation:.2f}) — "
                         f"bright = sharp cloud boundary, dark = uniform region"),
                use_container_width=True,
            )
        else:
            st.image(
                colorize_mask(pair["mask_path"], colormap=colormap_name),
                caption=f"Cloud probability heatmap ({colormap_name}) — grey diagonal stripe = no-data",
                use_container_width=True,
            )

    render_mask_legend(colormap_name, overlay_style, overlay_threshold,
                       show_gradient, gradient_saturation)

    render_cleanup_panel(pair, mean_p, colormap_name)

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

    # Default class = existing hand label if present, else auto_label.
    # When auto says "unknown" (classifier punted), pre-select "multi" so the
    # labeler isn't misled into a confident "clear" — they still verify.
    if has_existing:
        default_class_idx = existing_index(CLASSES, "class")
    elif auto_label in CLASSES:
        default_class_idx = CLASSES.index(auto_label)
    elif auto_label == "unknown":
        default_class_idx = CLASSES.index("multi")
    else:
        default_class_idx = 0
    default_conf_idx = (existing_index(CONFIDENCES, "confidence") if has_existing
                       else (CONFIDENCES.index(auto_conf) if auto_conf in CONFIDENCES else 0))

    # Optional decision-tree helper — collapsed by default for experienced
    # labelers; guides newer labelers (or genus-ambiguous frames) through the
    # standard protocol questions and cross-checks against auto + regime.
    with st.expander("🧭 Classification helper (decision tree)", expanded=False):
        sun_alt_row = weak_for_frame.get(("ephemeris", "sun_alt_deg"))
        try:
            current_regime = _sun_regime(float(sun_alt_row["value"])) if sun_alt_row else "UNKNOWN"
        except (TypeError, ValueError):
            current_regime = "UNKNOWN"
        render_decision_tree(pair["frame_id"], current_regime, auto_label, auto_conf)

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
