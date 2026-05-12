from fastapi import FastAPI, Depends, Request
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

# Catch-all handler for weird Ambient Weather paths
@app.get("/{path:path}")
async def catch_all_weather(path: str, request: Request, db: Session = Depends(get_db)):
    # If the URL looks like "weather&PASSKEY=..." it comes in as the path
    # We parse the full raw URL string to extract the real parameters
    full_path = str(request.url)
    
    # Extract everything after the server address
    # We look for common Ambient keywords like PASSKEY or stationtype
    if "PASSKEY" in full_path or "stationtype" in full_path:
        # Split by both ? and & to find all key=value pairs
        query_str = full_path.split("?", 1)[-1] if "?" in full_path else path
        # Normalize: sometimes the station sends /weather&... 
        if query_str.startswith("weather"):
            query_str = query_str.replace("weather", "", 1).lstrip("&")
            
        params = urllib.parse.parse_qs(query_str)
        # Flatten the list values from parse_qs
        data = {k: v[0] for k, v in params.items()}
        
        if data:
            new_record = models.WeatherRecord(raw_data=data)
            db.add(new_record)
            db.commit()
            print(f"Captured weather data from weird path: {data.get('dateutc', 'unknown time')}")
            return {"status": "success"}

    return {"status": "ignored", "path": path}

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
    return {"status": "success", "id": db_capture.id}

@app.get("/health")
def health():
    return {"status": "ok"}
