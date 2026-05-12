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

# Connect to DB
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

# --- AMBIENT WEATHER MIDDLEWARE ---
# Intercepts every request to check if it's a "broken" Ambient Weather ping
@app.middleware("http")
async def ambient_weather_interceptor(request: Request, call_next):
    # Get the raw path and query string
    path = request.url.path
    query = request.url.query
    full_str = f"{path}?{query}" if query else path
    
    # Ambient stations often send "weather&PASSKEY=..." which comes in as one big path string
    if "PASSKEY" in full_str or "stationtype" in full_str:
        # Clean up the string (remove leading / and 'weather' prefix if present)
        clean_str = full_str.lstrip("/").replace("weather", "", 1).lstrip("&").lstrip("?")
        
        # Parse into dictionary
        params = urllib.parse.parse_qs(clean_str)
        data = {k: v[0] for k, v in params.items()}
        
        if data:
            # Save to DB manually since we are in middleware
            db = Session(bind=engine)
            try:
                new_record = models.WeatherRecord(raw_data=data)
                db.add(new_record)
                db.commit()
                print(f"Middleware Captured weather: {data.get('dateutc', 'unknown')}")
            finally:
                db.close()
            
            # Return success to the station immediately
            return Response(content='{"status":"success"}', media_type="application/json")

    return await call_next(request)

# --- STANDARD ROUTES ---

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
    # Nearest neighbor sync
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

