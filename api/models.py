from sqlalchemy import Column, Integer, String, DateTime, JSON, ARRAY, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class WeatherRecord(Base):
    __tablename__ = "weather_records"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    raw_data = Column(JSON) # Stores the full ping from the station

class Capture(Base):
    __tablename__ = "captures"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    allsky_path = Column(String)
    thermal_path = Column(String)
    
    thermal_frame = Column(ARRAY(Float))
    esp32_sensors = Column(JSON)
    
    # New ephemeris fields
    sun_alt = Column(Float)
    moon_alt = Column(Float)
    moon_phase = Column(Float) # 0-100
    
    # We will link the closest weather record here
    weather_record_id = Column(Integer, ForeignKey("weather_records.id"), nullable=True)
