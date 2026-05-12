from fastapi import FastAPI, Depends, Request, Response
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from . import models
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import time
from datetime import datetime
import urllib.parse
import ephem
import math

app = FastAPI(title="Allsky Cloud Analysis API")

# Config
POSTGRES_USER = os.getenv("POSTGRES_USER", "allsky")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "allsky")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cloud_analysis")
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"

LAT = float(os.getenv("LOCATION_LATITUDE", "33.0"))
LON = float(os.getenv("LOCATION_LONGITUDE", "-84.0"))

engine = create_engine(SQLALCHEMY_DATABASE_URL)

connected = False
while not connected:
    try:
        models.Base.metadata.create_all(bind=engine)
        connected = True
    except Exception:
        time.sleep(2)

def get_db():
    db = Session(bind=engine)
    try:
        yield db
    finally:
        db.close()

def calculate_ephemeris(dt):
    obs = ephem.Observer()
    obs.lat = str(LAT)
    obs.lon = str(LON)
    obs.date = dt
    
    sun = ephem.Sun(obs)
    moon = ephem.Moon(obs)
    
    return {
        "sun_alt": math.degrees(sun.alt),
        "moon_alt": math.degrees(moon.alt),
        "moon_phase": moon.moon_phase * 100
    }

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
                return Response(content='{"status":"success"}', media_type="application/json")
            finally:
                db.close()
    return await call_next(request)

@app.get("/health")
def health():
    return {"status": "ok"}

class IngestPayload(BaseModel):
    allsky_path: str
    thermal_path: str
    thermal_frame: List[float]
    esp32_sensors: Dict
    captured_at: Optional[datetime] = None

@app.post("/ingest")
def ingest_data(payload: IngestPayload, db: Session = Depends(get_db)):
    sync_time = payload.captured_at or datetime.utcnow()
    
    # 1. Ephemeris calculation
    eph = calculate_ephemeris(sync_time)
    
    # 2. Nearest neighbor weather sync
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
    return {"status": "success", "id": db_capture.id}
