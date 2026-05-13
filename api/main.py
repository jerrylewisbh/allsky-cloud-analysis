from fastapi import FastAPI, Depends, Request, Response
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from . import models
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import time
from datetime import datetime, timezone
import socket
import urllib.parse
import ephem
import math

app = FastAPI(title="Allsky Cloud Analysis API")

# Config
POSTGRES_USER = os.getenv("POSTGRES_USER", "allsky")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "allsky")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cloud_analysis")
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"

LAT = os.getenv("LOCATION_LATITUDE", "33.0")
LON = os.getenv("LOCATION_LONGITUDE", "-84.0")
CWOP_ID = os.getenv("CWOP_ID", "")

engine = create_engine(SQLALCHEMY_DATABASE_URL)

connected = False
while not connected:
    try:
        models.Base.metadata.create_all(bind=engine)
        connected = True
        print(f"DB Connected. Location: {LAT}, {LON}")
    except Exception as e:
        print(f"Waiting for DB... {e}")
        time.sleep(2)


def send_to_cwop(raw_data, cwop_id, lat, lon):
    try:
        # 1. Format Time: DDHHMMz (UTC)
        time_str = datetime.now(timezone.utc).strftime("%d%H%Mz")

        # 2. Format Coordinates: DDMM.hhN / DDDMM.hhW
        lat_deg = int(abs(lat))
        lat_min = (abs(lat) - lat_deg) * 60
        lat_str = f"{lat_deg:02d}{lat_min:05.2f}{'N' if lat >= 0 else 'S'}"

        lon_deg = int(abs(lon))
        lon_min = (abs(lon) - lon_deg) * 60
        lon_str = f"{lon_deg:03d}{lon_min:05.2f}{'E' if lon >= 0 else 'W'}"

        # 3. Extract & Format Weather Values
        wdir = int(float(raw_data.get('winddir', 0)))
        wspd = int(float(raw_data.get('windspeedmph', 0)))
        wgst = int(float(raw_data.get('windgustmph', 0)))
        temp = int(float(raw_data.get('tempf', 0)))
        
        # Rainfall (hundredths of an inch)
        r_1h = int(float(raw_data.get('hourlyrainin', 0)) * 100)
        p_24h = int(float(raw_data.get('dailyrainin', 0)) * 100)
        
        # Barometer: inches Hg to tenths of millibars
        barom_mb_tenths = int(float(raw_data.get('baromrelin', 0)) * 338.639)

        # Humidity (00 = 100%)
        hum = int(float(raw_data.get('humidity', 0)))
        hum_str = "00" if hum == 100 else f"{hum:02d}"

        # 4. Construct the APRS String
        wx_payload = f"{time_str}{lat_str}/{lon_str}_c{wdir:03d}s{wspd:03d}g{wgst:03d}t{temp:03d}r{r_1h:03d}p{p_24h:03d}P{p_24h:03d}b{barom_mb_tenths:05d}h{hum_str}"
        packet = f"{cwop_id}>APRS,TCPXX*:@{wx_payload}Allsky Cloud Analysis\r\n"

        # 5. Send via TCP to CWOP Servers
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect(("cwop.aprs.net", 14580))
            s.sendall(f"user {cwop_id} pass -1 vers AllskyCloud 1.0\r\n".encode('utf-8'))
            s.sendall(packet.encode('utf-8'))
            print(f"CWOP Update Sent: {packet.strip()}")
            
    except Exception as e:
        print(f"CWOP Error: {e}")

def get_db():
    db = Session(bind=engine)
    try:
        yield db
    finally:
        db.close()

def calculate_ephemeris(dt):
    try:
        obs = ephem.Observer()
        obs.lat = str(LAT)
        obs.lon = str(LON)
        obs.date = dt
        
        sun = ephem.Sun(obs)
        moon = ephem.Moon(obs)
        
        results = {
            "sun_alt": float(math.degrees(sun.alt)),
            "moon_alt": float(math.degrees(moon.alt)),
            "moon_phase": float(moon.moon_phase * 100)
        }
        print(f"EPH: Sun {results['sun_alt']:.1f}, Moon {results['moon_alt']:.1f}, Phase {results['moon_phase']:.1f}")
        return results
    except Exception as e:
        print(f"EPH ERROR: {e}")
        return {"sun_alt": 0.0, "moon_alt": 0.0, "moon_phase": 0.0}

@app.middleware("http")
async def ambient_weather_interceptor(request: Request, call_next):
    full_url = str(request.url)
    if "PASSKEY=" in full_url or "stationtype=" in full_url:
        start_marker = "PASSKEY=" if "PASSKEY=" in full_url else "stationtype="
        _, params_part = full_url.split(start_marker, 1)
        params_str = start_marker + params_part
        params = urllib.parse.parse_qs(params_str)
        data = {k: v[0] for k, v in params.items()}
        if data:
            db = Session(bind=engine)
            try:
                new_record = models.WeatherRecord(raw_data=data)
                db.add(new_record)
                db.commit()
                if CWOP_ID:
                    send_to_cwop(data, CWOP_ID, float(LAT), float(LON))
                return Response(content='{"status":"success"}', media_type="application/json")
            except Exception:
                pass
            finally:
                db.close()
    return await call_next(request)

class IngestPayload(BaseModel):
    allsky_path: str
    thermal_path: str
    thermal_frame: List[float]
    esp32_sensors: Dict
    captured_at: Optional[datetime] = None

@app.post("/ingest")
def ingest_data(payload: IngestPayload, db: Session = Depends(get_db)):
    sync_time = payload.captured_at or datetime.utcnow()
    
    # Calculate Sun/Moon
    eph = calculate_ephemeris(sync_time)
    
    # Sync Weather
    closest_weather = db.query(models.WeatherRecord)\
        .order_by(func.abs(func.extract('epoch', models.WeatherRecord.timestamp) - func.extract('epoch', sync_time)))\
        .first()

    db_capture = models.Capture(
        timestamp=sync_time,
        allsky_path=payload.allsky_path,
        thermal_path=payload.thermal_path,
        thermal_frame=payload.thermal_frame,
        esp32_sensors=payload.esp32_sensors,
        sun_alt=eph["sun_alt"],
        moon_alt=eph["moon_alt"],
        moon_phase=eph["moon_phase"],
        weather_record_id=closest_weather.id if closest_weather else None
    )
    
    db.add(db_capture)
    db.commit()
    db.refresh(db_capture)
    print(f"INGEST SUCCESS: id {db_capture.id}, synced weather {db_capture.weather_record_id}")
    return {"status": "success", "id": db_capture.id}

@app.get("/sqm")
def get_sqm_unihedron(db: Session = Depends(get_db)):
    """Mimics a Unihedron SQM-LE 'rx' response for 3rd party networks."""
    latest = db.query(models.Capture).filter(models.Capture.esp32_sensors['sky_brightness_mpsas'].astext != None).order_by(models.Capture.timestamp.desc()).first()
    if not latest:
        return Response(content="r, 00.00m, 000000000000, 000000000000, 000000000000, 000.0\r\n", media_type="text/plain")
    
    sqm = latest.esp32_sensors.get("sky_brightness_mpsas", 0.0)
    temp = latest.esp32_sensors.get("temp", 0.0)
    
    # Format: r, <mpsas>m, <id...>, <id...>, <id...>, <temp>
    # This is the exact format Unihedron SQM-LE uses for the 'rx' command
    response = f"r, {sqm:05.2f}m, 000000000000, 000000000000, 000000000000, {temp:05.1f}\r\n"
    return Response(content=response, media_type="text/plain")

@app.get("/latest")
def get_latest(db: Session = Depends(get_db)):
    latest_capture = db.query(models.Capture).order_by(models.Capture.timestamp.desc()).first()
    latest_weather = db.query(models.WeatherRecord).order_by(models.WeatherRecord.timestamp.desc()).first()
    
    return {
        "capture": latest_capture,
        "weather": latest_weather.raw_data if latest_weather else None,
        "timestamp": latest_capture.timestamp if latest_capture else None
    }

@app.get("/health")
def health():
    return {"status": "ok"}
