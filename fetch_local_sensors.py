"""
fetch_local_sensors.py — Pull AWNET weather + ESP-derived signals + ephemeris
from the Postgres production database and emit weak_labels rows for every
frame in dataset_v2_*.

Each frame ends up with:
  source=weather_station: solarradiation (W/m^2), uv, windspeedmph→ms, winddir, hourlyrainin
  source=ephemeris:       sun_alt, moon_alt, moon_phase
  source=esp32_sensor:    sky_brightness_mpsas (when present — nighttime only)
  source=derived:         daytime_clear_sky_index when sun_alt > 0

Configure via environment (deploy/.env) or CLI flags:
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

Run:
  python fetch_local_sensors.py
  python fetch_local_sensors.py --pg-host 127.0.0.1 --pg-user myuser --pg-pass <secret>
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import math
import os
import re
import statistics
from pathlib import Path

import psycopg2
import psycopg2.extras

PROJECT_ROOT = Path(__file__).parent.resolve()
LABELS_DIR = PROJECT_ROOT / "labels"
WEAK_LABELS_CSV = LABELS_DIR / "weak_labels.csv"

LABEL_COLS = [
    "frame_id", "source", "attribute", "value", "value_unit",
    "timestamp", "source_distance_km", "source_distance_s",
]

# Haurwitz clear-sky GHI model (W/m^2). Simple, no dependencies, good enough
# for a clear-sky *index* (errors largely cancel in the ratio).
def haurwitz_clear_sky_ghi(sun_alt_deg: float) -> float:
    if sun_alt_deg <= 0:
        return 0.0
    z_rad = math.radians(90.0 - sun_alt_deg)
    cos_z = math.cos(z_rad)
    if cos_z <= 0:
        return 0.0
    return 1098.0 * cos_z * math.exp(-0.057 / cos_z)


def discover_frame_ids(dataset_glob: str) -> set[str]:
    ids = set()
    for ds in PROJECT_ROOT.glob(dataset_glob):
        for p in (ds / "masks").glob("*.png"):
            ids.add(p.stem)
    return ids


def load_existing_labels() -> dict[tuple, dict]:
    if not WEAK_LABELS_CSV.exists():
        return {}
    out = {}
    with open(WEAK_LABELS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["frame_id"], r["source"], r["attribute"], r.get("timestamp", ""))] = r
    return out


def emit_row(rows: list[dict], frame_id: str, source: str, attribute: str,
             value, unit: str, timestamp: dt.datetime, distance_km: float | None,
             distance_s: int | None) -> None:
    if value is None:
        return
    rows.append({
        "frame_id": frame_id,
        "source": source,
        "attribute": attribute,
        "value": f"{value:.4f}" if isinstance(value, float) else str(value),
        "value_unit": unit,
        "timestamp": timestamp.isoformat(),
        "source_distance_km": f"{distance_km:.2f}" if distance_km is not None else "",
        "source_distance_s": str(int(distance_s)) if distance_s is not None else "",
    })


def main():
    ap = argparse.ArgumentParser()
    # Defaults come from env vars (set via deploy/.env) — no hardcoded
    # site-specific values shipped in source.
    ap.add_argument("--pg-host", default=os.environ.get("PG_HOST", "127.0.0.1"))
    ap.add_argument("--pg-port", default=int(os.environ.get("PG_PORT", "5432")), type=int)
    ap.add_argument("--pg-db", default=os.environ.get("PG_DB", "cloud_analysis"))
    ap.add_argument("--pg-user", default=os.environ.get("PG_USER"))
    ap.add_argument("--pg-pass", default=os.environ.get("PG_PASS"))
    ap.add_argument("--datasets", default="dataset_v2_*")
    ap.add_argument("--start", help="ISO date, default = earliest frame date")
    ap.add_argument("--end", help="ISO date EXCLUSIVE, default = day after latest frame date")
    args = ap.parse_args()

    frame_ids = discover_frame_ids(args.datasets)
    print(f"Discovered {len(frame_ids)} dataset frames")
    if not frame_ids:
        return

    # Determine date range from frame_ids. Filename date prefixes are LOCAL
    # dates but captures.timestamp is UTC, so widen by ±1 day to bracket any
    # plausible local-UTC offset; the in-memory frame_id set then filters
    # precisely.
    dates = sorted({m.group(1)[:8] for fid in frame_ids
                   if (m := re.search(r"(\d{8}_\d{6})", fid))})
    start_date = dt.datetime.strptime(dates[0], "%Y%m%d").date() - dt.timedelta(days=1)
    end_date = dt.datetime.strptime(dates[-1], "%Y%m%d").date() + dt.timedelta(days=2)
    start = args.start or start_date.isoformat()
    end = args.end or end_date.isoformat()
    print(f"Querying PG for captures between {start} and {end}")

    conn = psycopg2.connect(host=args.pg_host, port=args.pg_port, dbname=args.pg_db,
                            user=args.pg_user, password=args.pg_pass)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT c.timestamp, c.allsky_path, c.esp32_sensors,
               c.sun_alt, c.moon_alt, c.moon_phase,
               w.raw_data AS awnet, w.timestamp AS awnet_ts
        FROM captures c
        LEFT JOIN weather_records w ON c.weather_record_id = w.id
        WHERE c.timestamp >= %s AND c.timestamp < %s
        ORDER BY c.timestamp
    """, (start, end))
    rows_db = cur.fetchall()
    conn.close()
    print(f"Got {len(rows_db)} capture rows from PG")

    # Pre-pass: build a CSI timeline across ALL captures (not just dataset
    # frames), so we have neighbors for the windowed variance even at the
    # ends of the dataset. CSI variance over a ±5 min window discriminates
    # convective (Cu — clouds pass intermittently, high variance) from
    # stratiform (Sc/St — steady attenuation, low variance). Established
    # solar-forecasting technique (Reno 2013, Stein 2012).
    csi_timeline: list[tuple[dt.datetime, float]] = []
    for r in rows_db:
        ts_r = r["timestamp"]
        if ts_r.tzinfo is None:
            ts_r = ts_r.replace(tzinfo=dt.timezone.utc)
        try:
            sun_alt_r = float(r["sun_alt"] or 0.0)
        except (TypeError, ValueError):
            continue
        if sun_alt_r <= 0:
            continue
        awnet_r = r["awnet"] or {}
        solar = awnet_r.get("solarradiation")
        if solar is None or solar == "":
            continue
        try:
            ghi_cs = haurwitz_clear_sky_ghi(sun_alt_r)
            if ghi_cs <= 50.0:
                continue
            csi_val = max(0.0, min(float(solar) / ghi_cs, 1.2))
            csi_timeline.append((ts_r, csi_val))
        except (TypeError, ValueError):
            continue
    csi_timeline.sort()
    csi_timestamps = [t for t, _ in csi_timeline]
    csi_values_only = [c for _, c in csi_timeline]
    csi_window = dt.timedelta(minutes=5)
    print(f"CSI timeline: {len(csi_timeline)} valid samples for variance computation")

    rows: list[dict] = []
    matched = 0
    daytime = 0
    nighttime = 0
    csi_std_emitted = 0

    for r in rows_db:
        path = r["allsky_path"] or ""
        frame_id = Path(path).stem
        if frame_id not in frame_ids:
            continue
        matched += 1
        ts = r["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)

        # Ephemeris (always available, distance trivially 0)
        sun_alt = float(r["sun_alt"] or 0.0)
        moon_alt = float(r["moon_alt"] or 0.0)
        moon_phase = float(r["moon_phase"] or 0.0)
        emit_row(rows, frame_id, "ephemeris", "sun_alt_deg", sun_alt, "deg", ts, 0.0, 0)
        emit_row(rows, frame_id, "ephemeris", "moon_alt_deg", moon_alt, "deg", ts, 0.0, 0)
        emit_row(rows, frame_id, "ephemeris", "moon_phase_pct", moon_phase, "pct", ts, 0.0, 0)

        # ESP-side illuminance_lux (nighttime skyglow signal)
        esp = r["esp32_sensors"] or {}
        lux = esp.get("illuminance_lux")
        if lux is not None:
            emit_row(rows, frame_id, "esp32_sensor", "illuminance_lux",
                     float(lux), "lux", ts, 0.0, 0)
            nighttime += 1

        # Firmware's own pessimistic cloud verdict + the numeric fractions it
        # uses to derive it (computed on the on-device 24x16 thermal crop, so
        # complementary to the projected-mask thermal_mean_p we compute in the
        # labeling tool).
        if (sky_cond := esp.get("sky_condition")):
            emit_row(rows, frame_id, "esp32_sensor", "sky_condition",
                     sky_cond, "category", ts, 0.0, 0)
        if (scf := esp.get("sky_cloud_fraction")) is not None:
            emit_row(rows, frame_id, "esp32_sensor", "sky_cloud_fraction_delta",
                     float(scf), "ratio", ts, 0.0, 0)
        if (sacf := esp.get("sky_abs_cloud_fraction")) is not None:
            emit_row(rows, frame_id, "esp32_sensor", "sky_cloud_fraction_abs",
                     float(sacf), "ratio", ts, 0.0, 0)
        if (sdm := esp.get("sky_delta_median")) is not None:
            emit_row(rows, frame_id, "esp32_sensor", "sky_delta_median_c",
                     float(sdm), "C", ts, 0.0, 0)

        # AWNET weather station (canonical daytime signal source)
        awnet = r["awnet"] or {}
        awnet_ts = r["awnet_ts"]
        if awnet_ts is not None and awnet:
            if awnet_ts.tzinfo is None:
                awnet_ts = awnet_ts.replace(tzinfo=dt.timezone.utc)
            awnet_offset_s = int((ts - awnet_ts).total_seconds())
            ws_attrs = [
                ("solarradiation", "solar_irradiance_wm2", "W/m^2"),
                ("uv", "uv_index", "index"),
                ("humidity", "humidity_pct", "pct"),
                ("baromrelin", "pressure_hpa", "hPa"),
                ("windspeedmph", "wind_speed_ms", "m/s"),
                ("windgustmph", "wind_gust_ms", "m/s"),
                ("winddir", "wind_dir_deg", "deg"),
                ("hourlyrainin", "rain_1h_mm", "mm"),
            ]
            for src_key, attr, unit in ws_attrs:
                v = awnet.get(src_key)
                if v is None or v == "":
                    continue
                v = float(v)
                # Conversions
                if src_key == "baromrelin":
                    v *= 33.8639  # inHg → hPa
                elif src_key in ("windspeedmph", "windgustmph"):
                    v *= 0.44704  # mph → m/s
                elif src_key == "hourlyrainin":
                    v *= 25.4     # in → mm
                emit_row(rows, frame_id, "weather_station", attr, v, unit,
                         awnet_ts, 0.0, awnet_offset_s)

            # Derived: daytime clear-sky index (sun above horizon)
            if sun_alt > 0:
                solar = awnet.get("solarradiation")
                if solar is not None and solar != "":
                    measured = float(solar)
                    ghi_cs = haurwitz_clear_sky_ghi(sun_alt)
                    if ghi_cs > 50.0:  # avoid divide-by-tiny near horizon
                        csi = max(0.0, min(measured / ghi_cs, 1.2))
                        emit_row(rows, frame_id, "derived", "daytime_clear_sky_index",
                                 csi, "ratio", ts, 0.0, 0)
                        daytime += 1

                        # CSI variance over ±5 min — discriminates convective
                        # (Cu, high variance) from stratiform (Sc/St, low).
                        # Needs ≥3 samples in the window for a meaningful std.
                        lo = bisect.bisect_left(csi_timestamps, ts - csi_window)
                        hi = bisect.bisect_right(csi_timestamps, ts + csi_window)
                        nearby = csi_values_only[lo:hi]
                        if len(nearby) >= 3:
                            csi_std = statistics.pstdev(nearby)
                            emit_row(rows, frame_id, "derived", "csi_std_10min",
                                     csi_std, "ratio", ts, 0.0, 0)
                            csi_std_emitted += 1

    print(f"Matched {matched}/{len(frame_ids)} dataset frames")
    print(f"  Daytime frames with CSI computed: {daytime}")
    print(f"  Daytime frames with CSI 10-min std: {csi_std_emitted}")
    print(f"  Nighttime frames with mpsas recorded: {nighttime}")

    # Merge into existing weak_labels.csv (dedup by composite key)
    existing = load_existing_labels()
    new_count = 0
    for row in rows:
        key = (row["frame_id"], row["source"], row["attribute"], row["timestamp"])
        if key not in existing:
            new_count += 1
        existing[key] = row

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEAK_LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLS)
        w.writeheader()
        for r in sorted(existing.values(), key=lambda x: (x["frame_id"], x["source"], x["attribute"])):
            w.writerow(r)
    print(f"Wrote {new_count} new rows ({len(existing)} total) to {WEAK_LABELS_CSV}")


if __name__ == "__main__":
    main()
