"""
fetch_goes.py — Pull GOES-19 ABI Level-2 cloud products for our dataset frames.

Four products sampled at the site lat/lon:
  ACMC — binary cloud mask (BCM)             → cloud_present {0|1}
  ACTPC — cloud-top phase                    → cloud_top_phase {clear|water|supercooled|mixed|ice}
  CTTC  — cloud-top temperature              → cloud_top_temp_k
  COTC  — cloud optical depth at 0.65 µm     → cloud_optical_depth

Source: NOAA Open Data, anonymous HTTPS access to s3://noaa-goes19/. Files are
cached in goes_cache/ to avoid re-downloads (each ABI file is 5–20 MB).

Run:
  python fetch_goes.py
  python fetch_goes.py --products ACMC --max-scans 5  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT_ROOT = Path(__file__).parent.resolve()
LABELS_DIR = PROJECT_ROOT / "labels"
WEAK_LABELS_CSV = LABELS_DIR / "weak_labels.csv"
CACHE_DIR = PROJECT_ROOT / "goes_cache"
LABEL_COLS = ["frame_id", "source", "attribute", "value", "value_unit",
              "timestamp", "source_distance_km", "source_distance_s"]

# Site lat/lon must be provided via --site-lat/--site-lon or the SITE_LAT/
# SITE_LON env vars. We intentionally don't ship hardcoded defaults — exact
# coordinates are sensitive (effectively a home address for a backyard rig).
DEFAULT_LAT = None
DEFAULT_LON = None

GOES_BUCKET = "noaa-goes19"
S3_HTTPS_BASE = f"https://{GOES_BUCKET}.s3.amazonaws.com"
SCAN_INTERVAL_MIN = 5  # CONUS scan cadence

# Frame filenames (ccd1_YYYYMMDD_HHMMSS) are local wall-clock time on the
# camera host. ALLSKY_LOCAL_TZ must be an IANA name (e.g. "America/Edmonton")
# so DST is handled. Without it the snap-to-scan logic samples GOES from the
# wrong moment by the local-UTC offset.
def _resolve_local_tz() -> ZoneInfo:
    name = os.environ.get("ALLSKY_LOCAL_TZ", "UTC")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        sys.exit(f"ALLSKY_LOCAL_TZ={name!r} is not a known IANA timezone")

LOCAL_TZ = _resolve_local_tz()

# CONUS sectors have product code suffix C. GOES-19 ABI Level-2 product names.
# NOTE: GOES-19 doesn't publish cloud-top *temperature* for CONUS (only full
# disk/mesoscale). Cloud-top *height* in ACHAC is the better signal anyway —
# it maps directly to altitude family for genus discrimination, and we have
# the surface thermal sensor for ground-level temperature.
PRODUCT_CODES = {
    "ACMC":  ("BCM",   "cloud_present",          "binary"),       # 0=clear 1=cloud
    "ACTPC": ("Phase", "cloud_top_phase",        "category"),     # 0..5
    "ACHAC": ("HT",    "cloud_top_height_m",     "m"),            # m AGL
    "CODC":  ("COD",   "cloud_optical_depth",    "dimensionless"),
}
ACTP_PHASE_LABELS = {
    0: "clear", 1: "water", 2: "supercooled_water",
    3: "mixed", 4: "ice", 5: "unknown",
}


def curl_text(url: str, timeout: int = 60) -> str:
    r = subprocess.run(["curl", "-sS", "--fail", "--max-time", str(timeout), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed: {url}\n{r.stderr.strip()}")
    return r.stdout


def curl_binary(url: str, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = subprocess.run(["curl", "-sS", "--fail", "--max-time", str(timeout), "-o", str(tmp), url])
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"curl failed: {url}")
    tmp.replace(dest)


def s3_list_keys(prefix: str) -> list[str]:
    """Anonymous S3 list-objects via the public XML REST API."""
    url = f"{S3_HTTPS_BASE}/?list-type=2&prefix={prefix}"
    xml_text = curl_text(url)
    root = ET.fromstring(xml_text)
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    return [c.findtext(f"{ns}Key") for c in root.findall(f"{ns}Contents")]


def find_nearest_file(product: str, scan_dt: dt.datetime, cache: dict) -> tuple[str, dt.datetime] | None:
    """Pick the GOES file whose scan start time is nearest scan_dt for that product+hour."""
    doy = scan_dt.timetuple().tm_yday
    prefix = f"ABI-L2-{product}/{scan_dt.year}/{doy:03d}/{scan_dt.hour:02d}/"
    key = (product, scan_dt.year, doy, scan_dt.hour)
    if key not in cache:
        try:
            cache[key] = s3_list_keys(prefix)
        except Exception:
            cache[key] = []
    candidates = cache[key]
    if not candidates:
        return None
    best, best_dt, best_dist = None, None, None
    for k in candidates:
        m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})\d", k)
        if not m:
            continue
        y, d, hh, mm, ss = (int(x) for x in m.groups())
        file_dt = dt.datetime(y, 1, 1, hh, mm, ss, tzinfo=dt.timezone.utc) + dt.timedelta(days=d - 1)
        dist = abs((file_dt - scan_dt).total_seconds())
        if best_dist is None or dist < best_dist:
            best, best_dt, best_dist = k, file_dt, dist
    return (best, best_dt) if best else None


def download_cached(key: str) -> Path:
    local = CACHE_DIR / key
    if local.exists() and local.stat().st_size > 1024:
        return local
    url = f"{S3_HTTPS_BASE}/{key}"
    curl_binary(url, local)
    return local


def latlon_to_goes_xy(lat_deg: float, lon_deg: float, proj_vars: dict) -> tuple[float, float]:
    """Convert geodetic lat/lon to ABI fixed-grid scan/elev angles using
    the standard NOAA equations from the ABI PUG."""
    H = proj_vars["perspective_point_height"] + proj_vars["semi_major_axis"]
    req = proj_vars["semi_major_axis"]
    rpol = proj_vars["semi_minor_axis"]
    lon_0 = math.radians(proj_vars["longitude_of_projection_origin"])
    e2 = 1.0 - (rpol / req) ** 2

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    phi_c = math.atan((rpol ** 2 / req ** 2) * math.tan(lat))
    rc = rpol / math.sqrt(1.0 - e2 * math.cos(phi_c) ** 2)
    sx = H - rc * math.cos(phi_c) * math.cos(lon - lon_0)
    sy = -rc * math.cos(phi_c) * math.sin(lon - lon_0)
    sz = rc * math.sin(phi_c)
    # Visibility test (above-horizon)
    if H * (H - sx) < (sy ** 2 + (req / rpol) ** 2 * sz ** 2):
        raise ValueError(f"Point ({lat_deg},{lon_deg}) is below GOES horizon")
    y = math.atan(sz / sx)
    x = math.asin(-sy / math.sqrt(sx * sx + sy * sy + sz * sz))
    return x, y


def sample_product(local_nc: Path, var_name: str, lat: float, lon: float):
    """Open a GOES ABI L2 NetCDF and sample the data variable at lat/lon."""
    with netCDF4.Dataset(str(local_nc)) as nc:
        gip = nc.variables.get("goes_imager_projection")
        if gip is None:
            return None
        proj = {a: getattr(gip, a) for a in [
            "perspective_point_height", "semi_major_axis", "semi_minor_axis",
            "longitude_of_projection_origin",
        ]}
        try:
            sat_x, sat_y = latlon_to_goes_xy(lat, lon, proj)
        except ValueError:
            return None
        x_arr = nc.variables["x"][:]
        y_arr = nc.variables["y"][:]
        ix = int(np.argmin(np.abs(x_arr - sat_x)))
        iy = int(np.argmin(np.abs(y_arr - sat_y)))
        if var_name not in nc.variables:
            return None
        data = nc.variables[var_name][iy, ix]
        if np.ma.is_masked(data) or (hasattr(data, "mask") and data.mask):
            return None
        return float(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dataset_v2_*")
    ap.add_argument("--site-lat", type=float,
                    default=os.environ.get("SITE_LAT", DEFAULT_LAT))
    ap.add_argument("--site-lon", type=float,
                    default=os.environ.get("SITE_LON", DEFAULT_LON))
    ap.add_argument("--products", nargs="+", default=list(PRODUCT_CODES.keys()),
                    choices=list(PRODUCT_CODES.keys()))
    ap.add_argument("--match-window-s", type=int, default=600)
    ap.add_argument("--max-scans", type=int, default=0,
                    help="0 = all; otherwise limit per-product for smoke testing")
    args = ap.parse_args()

    if args.site_lat is None or args.site_lon is None:
        ap.error("--site-lat and --site-lon are required (or set SITE_LAT/SITE_LON env vars)")

    print(f"GOES-19 fetcher → products {args.products} at ({args.site_lat:.4f}, {args.site_lon:.4f})")

    print(f"Filename timezone: {LOCAL_TZ.key} (set ALLSKY_LOCAL_TZ to change)")
    # Discover frames with timestamps
    frames: list[tuple[str, dt.datetime]] = []
    for ds in PROJECT_ROOT.glob(args.datasets):
        for p in (ds / "masks").glob("*.png"):
            m = re.search(r"(\d{8}_\d{6})", p.stem)
            if m:
                ts_local = dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=LOCAL_TZ)
                frames.append((p.stem, ts_local.astimezone(dt.timezone.utc)))
    print(f"Discovered {len(frames)} dataset frames")

    # Snap each frame's timestamp to the nearest 5-min scan boundary
    def snap(ts: dt.datetime) -> dt.datetime:
        mins = ts.minute + ts.second / 60.0
        snapped = round(mins / SCAN_INTERVAL_MIN) * SCAN_INTERVAL_MIN
        if snapped >= 60:
            return (ts.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1))
        return ts.replace(minute=int(snapped), second=0, microsecond=0)

    scans_to_fetch = sorted({snap(ts) for _, ts in frames})
    if args.max_scans > 0:
        scans_to_fetch = scans_to_fetch[: args.max_scans]
    print(f"Unique 5-min scan boundaries to fetch: {len(scans_to_fetch)}")

    listing_cache: dict = {}
    weak_rows: list[dict] = []

    for product in args.products:
        var, attr, unit = PRODUCT_CODES[product]
        print(f"\n--- product {product} (var={var}) ---")
        # Pre-resolve file keys for each scan
        scan_to_file: dict[dt.datetime, tuple[str, dt.datetime]] = {}
        for i, scan_dt in enumerate(scans_to_fetch):
            result = find_nearest_file(product, scan_dt, listing_cache)
            if result:
                scan_to_file[scan_dt] = result
            if (i + 1) % 50 == 0:
                print(f"  resolved {i+1}/{len(scans_to_fetch)} scans …")

        print(f"  files resolved: {len(scan_to_file)}/{len(scans_to_fetch)}")
        if not scan_to_file:
            continue

        # Build per-scan sampled value (download each file once, then map to frames)
        sampled: dict[dt.datetime, tuple[float, dt.datetime]] = {}
        for i, (scan_dt, (key, file_dt)) in enumerate(scan_to_file.items()):
            try:
                local = download_cached(key)
                value = sample_product(local, var, args.site_lat, args.site_lon)
                if value is not None:
                    sampled[scan_dt] = (value, file_dt)
            except Exception as e:
                print(f"  [{scan_dt.isoformat()}] sample failed: {e}")
            if (i + 1) % 25 == 0:
                print(f"  downloaded/sampled {i+1}/{len(scan_to_file)}")
        print(f"  successful samples: {len(sampled)}")

        # Match frames → snapped scan → sampled value
        matched = 0
        for frame_id, ts in frames:
            scan_dt = snap(ts)
            if scan_dt not in sampled:
                continue
            value, file_dt = sampled[scan_dt]
            offset_s = int((ts - file_dt).total_seconds())
            if abs(offset_s) > args.match_window_s:
                continue
            matched += 1
            if product == "ACTPC":
                value_str = ACTP_PHASE_LABELS.get(int(value), f"code_{int(value)}")
            elif product == "ACMC":
                value_str = str(int(value))
            else:
                value_str = f"{value:.3f}"
            weak_rows.append({
                "frame_id": frame_id, "source": f"goes19_{product.lower()}",
                "attribute": attr, "value": value_str, "value_unit": unit,
                "timestamp": file_dt.isoformat(),
                "source_distance_km": "0.00",  # satellite is overhead — site distance is degenerate
                "source_distance_s": str(offset_s),
            })
        print(f"  matched {matched} frames")

    # Merge into weak_labels.csv (dedup by composite key)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if WEAK_LABELS_CSV.exists():
        with open(WEAK_LABELS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                existing[(r["frame_id"], r["source"], r["attribute"], r["timestamp"])] = r
    new_count = 0
    for row in weak_rows:
        key = (row["frame_id"], row["source"], row["attribute"], row["timestamp"])
        if key not in existing:
            new_count += 1
        existing[key] = row
    with open(WEAK_LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLS)
        w.writeheader()
        for r in sorted(existing.values(), key=lambda x: (x["frame_id"], x["source"], x["attribute"])):
            w.writerow(r)
    print(f"\nWrote {new_count} new rows ({len(existing)} total) to {WEAK_LABELS_CSV}")
    print(f"Cache: {CACHE_DIR} ({sum(p.stat().st_size for p in CACHE_DIR.rglob('*.nc')) // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()
