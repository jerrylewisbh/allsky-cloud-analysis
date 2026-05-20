"""
fetch_metar.py — Pull METAR observations near each frame timestamp and emit
weak labels per docs/labeling-protocol.md §"Output file format".

For each frame in dataset_v2_*/meta/*.json the script finds the closest METAR
observation from a configurable ICAO airport within ±MATCH_WINDOW seconds,
parses cloud groups (coverage + base height + CB/TCU genus hint), and appends
rows to labels/weak_labels.csv.

Run:
  python fetch_metar.py --station SBPA
  python fetch_metar.py --station SBPA --site-lat -30.05 --site-lon -51.17 \
                        --datasets 'dataset_v2_*'

Data source: Iowa Environmental Mesonet ASOS archive
(mesonet.agron.iastate.edu). Public, no auth required. Returns raw METAR text;
this script does its own cloud-group parsing because the structured fields
collapse multi-layer reports.

Finding your nearest ICAO airport:
  - search openstreetmap.org for "ICAO" near your location, OR
  - https://www.airportcodes.io/en/airport/ → filter by country/region.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def http_get(url: str, timeout: int = 60) -> str:
    """Use curl so we inherit the macOS system trust store. The python.org
    installer ships its own CA bundle that won't trust corporate MITM."""
    r = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", str(timeout), url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {r.stderr.strip()}")
    return r.stdout

PROJECT_ROOT = Path(__file__).parent.resolve()
LABELS_DIR = PROJECT_ROOT / "labels"
WEAK_LABELS_CSV = LABELS_DIR / "weak_labels.csv"

ASOS_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    "?station={station}&data=metar&year1={y1}&month1={m1}&day1={d1}"
    "&year2={y2}&month2={m2}&day2={d2}&tz=Etc/UTC&format=onlycomma"
    "&latlon=yes&missing=M&trace=T&direct=yes"
)
STATION_META_URL = "https://mesonet.agron.iastate.edu/json/network.py?network={network}"

MATCH_WINDOW_S = 15 * 60  # ±15 min — standard in the literature

# Cloud-group regex: SKC/CLR/NCD/NSC/CAVOK, or 3-letter coverage + 3-digit base.
# Optional CB/TCU genus suffix.
CLOUD_GROUP_RE = re.compile(
    r"\b(SKC|CLR|NCD|NSC|CAVOK|NOCLO|(FEW|SCT|BKN|OVC|VV)(\d{3})(CB|TCU)?)\b"
)

COVERAGE_TO_OKTA = {
    "SKC": 0, "CLR": 0, "NCD": 0, "NSC": 0, "CAVOK": 0, "NOCLO": 0,
    "FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8,
}


@dataclass
class MetarObs:
    station: str
    timestamp: dt.datetime  # UTC
    raw: str
    coverage_okta: int            # 0..8
    cloud_base_m: float | None    # meters AGL; None if no clouds reported
    genus_hint: str | None        # CB, TCU, or None
    layers: list[tuple[str, int]] # [(coverage_str, base_ft), ...] all layers


def parse_metar_cloud(raw_metar: str) -> tuple[int, float | None, str | None, list[tuple[str, int]]]:
    """Return (max_okta_coverage, lowest_base_m_with_okta>=3, genus_hint, all_layers).

    Highest-coverage layer wins for okta. CB/TCU on any layer flags genus_hint.
    """
    layers: list[tuple[str, int]] = []
    max_okta = 0
    genus_hint: str | None = None
    lowest_base_ft: int | None = None

    for m in CLOUD_GROUP_RE.finditer(raw_metar):
        token = m.group(1)
        if token in COVERAGE_TO_OKTA:
            okta = COVERAGE_TO_OKTA[token]
            layers.append((token, 0))
            max_okta = max(max_okta, okta)
            continue
        cov = m.group(2)
        base_ft = int(m.group(3)) * 100
        suffix = m.group(4)
        if not cov or not COVERAGE_TO_OKTA.get(cov):
            continue
        okta = COVERAGE_TO_OKTA[cov]
        layers.append((cov, base_ft))
        if okta > max_okta:
            max_okta = okta
        if okta >= 3 and (lowest_base_ft is None or base_ft < lowest_base_ft):
            lowest_base_ft = base_ft
        if suffix and not genus_hint:
            genus_hint = suffix

    cloud_base_m = lowest_base_ft * 0.3048 if lowest_base_ft is not None else None
    return max_okta, cloud_base_m, genus_hint, layers


def altitude_bucket_from_base_m(base_m: float | None) -> str | None:
    if base_m is None:
        return None
    if base_m < 2000:
        return "low"
    if base_m < 6000:
        return "mid"
    return "high"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def fetch_metar_csv(station: str, start: dt.date, end_inclusive: dt.date) -> list[MetarObs]:
    end = end_inclusive + dt.timedelta(days=1)  # IEM's day2 is exclusive
    url = ASOS_URL.format(
        station=station,
        y1=start.year, m1=start.month, d1=start.day,
        y2=end.year, m2=end.month, d2=end.day,
    )
    print(f"  GET {url}")
    text = http_get(url, timeout=60)

    obs: list[MetarObs] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        raw = (row.get("metar") or "").strip()
        valid_str = (row.get("valid") or "").strip()
        if not raw or not valid_str:
            continue
        try:
            ts = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        okta, base_m, genus, layers = parse_metar_cloud(raw)
        obs.append(MetarObs(
            station=station, timestamp=ts, raw=raw,
            coverage_okta=okta, cloud_base_m=base_m,
            genus_hint=genus, layers=layers,
        ))
    return obs


def lookup_station_latlon(station: str) -> tuple[float, float] | None:
    """Best-effort: try common ASOS networks for the station's lat/lon."""
    common_networks = [
        "BR__ASOS", "AS_ASOS", "INTL", "AWOS",
        f"{station[:2]}_ASOS",  # country-prefix heuristic
    ]
    seen: dict[str, tuple[float, float]] = {}
    for net in dict.fromkeys(common_networks):
        try:
            data = json.loads(http_get(STATION_META_URL.format(network=net), timeout=30))
        except Exception:
            continue
        for s in data.get("features", []):
            sid = s.get("properties", {}).get("sid") or s.get("id")
            if not sid or sid in seen:
                continue
            geom = s.get("geometry") or {}
            coords = geom.get("coordinates")
            if coords and len(coords) == 2:
                seen[sid] = (coords[1], coords[0])  # geojson is (lon, lat)
        if station in seen:
            return seen[station]
    return seen.get(station)


def discover_frames(dataset_glob: str) -> list[tuple[str, dt.datetime]]:
    """Walk dataset_v2_*/meta/*.json (or images/*.jpg) and extract (frame_id, ts UTC)."""
    frames: list[tuple[str, dt.datetime]] = []
    for ds in sorted(PROJECT_ROOT.glob(dataset_glob)):
        meta_dir = ds / "meta"
        if meta_dir.is_dir():
            files = sorted(meta_dir.glob("*.json"))
            stem_source = [f.stem for f in files]
        else:
            files = sorted((ds / "images").glob("*.jpg")) if (ds / "images").is_dir() else []
            stem_source = [f.stem for f in files]
        for stem in stem_source:
            m = re.search(r"(\d{8}_\d{6})", stem)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc)
            frames.append((stem, ts))
    return frames


def write_weak_labels(rows: list[dict]) -> int:
    """Append rows to weak_labels.csv, dedup by (frame_id, source, attribute, source_id)."""
    cols = [
        "frame_id", "source", "attribute", "value", "value_unit",
        "timestamp", "source_distance_km", "source_distance_s",
    ]
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple, dict] = {}
    if WEAK_LABELS_CSV.exists():
        with open(WEAK_LABELS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                existing[(r["frame_id"], r["source"], r["attribute"], r.get("timestamp", ""))] = r
    added = 0
    for row in rows:
        key = (row["frame_id"], row["source"], row["attribute"], row["timestamp"])
        if key not in existing:
            added += 1
        existing[key] = row
    with open(WEAK_LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(existing.values(), key=lambda x: (x["frame_id"], x["attribute"])):
            w.writerow(r)
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True, help="ICAO of nearest reporting airport (e.g. SBPA, KSEA, LFPG)")
    ap.add_argument("--datasets", default="dataset_v2_*", help="glob for dataset directories under project root")
    ap.add_argument("--site-lat", type=float, default=None, help="your sensor latitude (decimal degrees)")
    ap.add_argument("--site-lon", type=float, default=None, help="your sensor longitude (decimal degrees)")
    ap.add_argument("--station-lat", type=float, default=None, help="override station lat if IEM lookup fails")
    ap.add_argument("--station-lon", type=float, default=None, help="override station lon if IEM lookup fails")
    ap.add_argument("--match-window-s", type=int, default=MATCH_WINDOW_S, help="±seconds for METAR-to-frame match")
    args = ap.parse_args()

    frames = discover_frames(args.datasets)
    if not frames:
        sys.exit(f"No frames found under {args.datasets}/{{meta/,images/}}")
    print(f"Discovered {len(frames)} frames")

    days = sorted({ts.date() for _, ts in frames})
    print(f"Date range: {days[0]} → {days[-1]} ({len(days)} days)")

    if args.station_lat is not None and args.station_lon is not None:
        station_latlon = (args.station_lat, args.station_lon)
        print(f"Station {args.station} location (override): lat={station_latlon[0]:.4f} lon={station_latlon[1]:.4f}")
    else:
        station_latlon = lookup_station_latlon(args.station)
        if station_latlon:
            print(f"Station {args.station} location: lat={station_latlon[0]:.4f} lon={station_latlon[1]:.4f}")
        else:
            print(f"Warning: could not look up {args.station} lat/lon — pass --station-lat/--station-lon to override")

    distance_km: float | None = None
    if station_latlon and args.site_lat is not None and args.site_lon is not None:
        distance_km = haversine_km(args.site_lat, args.site_lon, *station_latlon)
        print(f"Sensor → station distance: {distance_km:.1f} km")

    print(f"Fetching METARs for {args.station} {days[0]} → {days[-1]} …")
    obs = fetch_metar_csv(args.station, days[0], days[-1])
    print(f"Got {len(obs)} METAR observations")
    if not obs:
        sys.exit("No METAR observations returned — wrong station ID or no data for that range?")

    # Index METARs by timestamp for fast nearest lookup (sorted bisect would be better but list is short)
    obs.sort(key=lambda o: o.timestamp)
    metar_times = [o.timestamp for o in obs]

    rows = []
    matched = 0
    for frame_id, ts in frames:
        # Find nearest METAR within window
        nearest = None
        nearest_dt = None
        for i, mt in enumerate(metar_times):
            d = abs((ts - mt).total_seconds())
            if nearest_dt is None or d < nearest_dt:
                nearest_dt = d
                nearest = obs[i]
        if nearest is None or nearest_dt is None or nearest_dt > args.match_window_s:
            continue
        matched += 1
        common = {
            "frame_id": frame_id,
            "source": "metar",
            "timestamp": nearest.timestamp.isoformat(),
            "source_distance_km": f"{distance_km:.2f}" if distance_km is not None else "",
            "source_distance_s": int((ts - nearest.timestamp).total_seconds()),
        }

        # Emit one row per attribute
        rows.append({**common, "attribute": "coverage_okta", "value": str(nearest.coverage_okta), "value_unit": "okta"})
        if nearest.cloud_base_m is not None:
            rows.append({**common, "attribute": "cloud_base_height_m", "value": f"{nearest.cloud_base_m:.0f}", "value_unit": "m"})
            ab = altitude_bucket_from_base_m(nearest.cloud_base_m)
            if ab:
                rows.append({**common, "attribute": "altitude_bucket", "value": ab, "value_unit": "class"})
        if nearest.genus_hint:
            rows.append({**common, "attribute": "cloud_genus_hint", "value": nearest.genus_hint, "value_unit": "wmo_code"})
        rows.append({**common, "attribute": "raw_metar", "value": nearest.raw, "value_unit": "string"})

    added = write_weak_labels(rows)
    print(f"Matched {matched}/{len(frames)} frames to METAR within ±{args.match_window_s}s")
    print(f"Wrote {added} new rows to {WEAK_LABELS_CSV}  (total rows in file include cumulative dedup-merged history)")


if __name__ == "__main__":
    main()
