from fastapi import FastAPI, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from . import models
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import time
from datetime import datetime

app = FastAPI(title="Allsky Cloud Analysis API")

# DB Connection
POSTGRES_USER = os.getenv("POSTGRES_USER", "allsky")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "allsky")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cloud_analysis")
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"

# Robust database initialization with retries
engine = create_engine(SQLALCHEMY_DATABASE_URL)

connected = False
while not connected:
    try:
        models.Base.metadata.create_all(bind=engine)
        connected = True
        print("Successfully connected to the database and verified tables.")
    except Exception as e:
        print(f"Database not ready yet... retrying in 2 seconds.")
        time.sleep(2)

def get_db():
    db = Session(bind=engine)
    try:
        yield db
    finally:
        db.close()

@app.get("/weather")
async def receive_weather(request: Request, db: Session = Depends(get_db)):
    data = dict(request.query_params)
    if not data:
        return {"status": "no data"}
    
    new_record = models.WeatherRecord(raw_data=data)
    db.add(new_record)
    db.commit()
    return {"status": "success"}

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

@app.get("/health")
def health():
    return {"status": "ok"}
