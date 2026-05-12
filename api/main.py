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

app = FastAPI(title="Allsky Cloud Analysis API")

# DB Connection
POSTGRES_USER = os.getenv("POSTGRES_USER", "allsky")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "allsky")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cloud_analysis")
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"

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

@app.middleware("http")
async def ambient_weather_interceptor(request: Request, call_next):
    full_url = str(request.url)
    
    # If the URL contains weather parameters anywhere (path or query)
    if "PASSKEY=" in full_url or "stationtype=" in full_url:
        print(f"DEBUG: Intercepted potential weather URL: {full_url}")
        
        # Find where the parameters actually start
        # They might start with PASSKEY or stationtype
        start_marker = "PASSKEY=" if "PASSKEY=" in full_url else "stationtype="
        _, params_part = full_url.split(start_marker, 1)
        params_str = start_marker + params_part
        
        # Standardize: parse_qs expects a standard query string
        # If there are any remaining / or leading &, they will be handled by parse_qs
        params = urllib.parse.parse_qs(params_str)
        data = {k: v[0] for k, v in params.items()}
        
        if data:
            db = Session(bind=engine)
            try:
                new_record = models.WeatherRecord(raw_data=data)
                db.add(new_record)
                db.commit()
                print(f"SUCCESS: Saved weather data for date: {data.get('dateutc', 'unknown')}")
                # Return 200 OK immediately to satisfy the station
                return Response(content='{"status":"success"}', media_type="application/json")
            except Exception as e:
                print(f"ERROR: Failed to save weather record: {e}")
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

@app.post("/ingest")
def ingest_data(payload: IngestPayload, db: Session = Depends(get_db)):
    now = datetime.utcnow()
    closest_weather = db.query(models.WeatherRecord)\
        .order_by(func.abs(func.extract('epoch', models.WeatherRecord.timestamp) - func.extract('epoch', now)))\
        .first()

    db_capture = models.Capture(
        allsky_path=payload.allsky_path,
        thermal_path=payload.thermal_path,
        thermal_frame=payload.thermal_frame,
        esp32_sensors=payload.esp32_sensors,
        weather_record_id=closest_weather.id if closest_weather else None
    )
    
    db.add(db_capture)
    db.commit()
    db.refresh(db_capture)
    return {"status": "success", "id": db_capture.id, "synced_weather_id": db_capture.weather_record_id}
