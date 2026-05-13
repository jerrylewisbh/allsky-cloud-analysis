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
import requests

app = FastAPI(title="Allsky Cloud Analysis API")

# Config
POSTGRES_USER = os.getenv("POSTGRES_USER", "allsky")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "allsky")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cloud_analysis")
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"

LAT = float(os.getenv("LOCATION_LATITUDE", "51.05"))
LON = float(os.getenv("LOCATION_LONGITUDE", "-114.07"))
CWOP_ID = os.getenv("CWOP_ID", "")
LPM_API_KEY = os.getenv("LPM_API_KEY", "")

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


def send_to_cwop(raw_data, cwop_id, lat, lon, source="ESP32"):
    """Sends sensor data to CWOP/findu.com"""
    try:
        if not cwop_id:
            return

        print(f"[{datetime.now()}] Attempting CWOP push from {source}...")

        # 1. Format Time: DDHHMMz (UTC)
        time_str = datetime.now(timezone.utc).strftime("%d%H%Mz")

        # 2. Format Coordinates: DDMM.hhN / DDDMM.hhW
        lat_deg = int(abs(lat))
        lat_min = (abs(lat) - lat_deg) * 60
        lat_str = f"{lat_deg:02d}{lat_min:05.2f}{'N' if lat >= 0 else 'S'}"

        lon_deg = int(abs(lon))
        lon_min = (abs(lon) - lon_deg) * 60
        lon_str = f"{lon_deg:03d}{lon_min:05.2f}{'E' if lon >= 0 else 'W'}"

        # 3. Extract & Convert Values
        temp_f_raw = raw_data.get('tempf')
        if temp_f_raw is None:
            temp_c = raw_data.get('temp')
            temp_f = int(round(temp_c * 1.8 + 32)) if temp_c is not None else None
        else:
            temp_f = int(round(float(temp_f_raw)))

        hum_raw = raw_data.get('humidity', raw_data.get('hum'))
        hum_val = int(round(float(hum_raw))) if hum_raw is not None else None
        
        pres_hpa = raw_data.get('pres')
        if pres_hpa is not None:
            barom_mb_tenths = int(round(pres_hpa * 10))
        else:
            barom_in = raw_data.get('baromrelin')
            barom_mb_tenths = int(round(float(barom_in) * 33.8639 * 10)) if barom_in is not None else None

        # Wind & Rain parameters
        wdir = int(round(float(raw_data.get('winddir', 0))))
        wind_kmh = raw_data.get('wind')
        
        if wind_kmh is not None:
            wspd_mph = int(round(wind_kmh * 0.621371))
            wgst_mph = "..." # ESP32 has no gust memory
        else:
            wspd_mph = int(round(float(raw_data.get('windspeedmph', 0))))
            # Wind Gust (mph)
            wgst_val = raw_data.get('windgustmph')
            wgst_mph = f"{int(round(float(wgst_val))):03d}" if wgst_val is not None else "..."
        
        # Rain: CWOP expects hundredths of an inch (0.15 inches = 15)
        # r = rain last hour, p = rain last 24h, P = rain since midnight
        rain_1h_in = raw_data.get('hourlyrainin')
        r_1h = f"{int(round(float(rain_1h_in) * 100)):03d}" if rain_1h_in is not None else "..."
        
        rain_24h_in = raw_data.get('dailyrainin')
        p_24h = f"{int(round(float(rain_24h_in) * 100)):03d}" if rain_24h_in is not None else "..."
        
        # Ambient doesn't explicitly send "since midnight" separately from daily usually,
        # but we can populate P with daily if it resets at midnight.
        P_mid = p_24h

        # Solar Radiation (Luminosity) - CWOP 'L' or 'l'
        # Format for L: L followed by 3 digits for Watts/m2. If > 999, use 'l' and add 1000.
        solar_rad = raw_data.get('solarradiation')
        solar_str = ""
        if solar_rad is not None:
            solar_w = int(round(float(solar_rad)))
            if solar_w <= 999:
                solar_str = f"L{solar_w:03d}"
            else:
                solar_str = f"l{(solar_w - 1000):03d}"
        else:
            # Try to convert TSL2591 lux to W/m2 approximation (lux / 120 is a rough daylight estimate)
            lux = raw_data.get('lux')
            if lux is not None:
                solar_w = int(round(float(lux) / 120))
                if solar_w <= 999:
                    solar_str = f"L{solar_w:03d}"
                else:
                    solar_str = f"l{(solar_w - 1000):03d}"
        
        if temp_f is None or barom_mb_tenths is None:
            print(f"  CWOP Skip: Missing Temp ({temp_f}) or Baro ({barom_mb_tenths})")
            return

        # Temperature string (handles negative)
        temp_str = f"t{temp_f:03d}" if temp_f >= 0 else f"t-{abs(temp_f):02d}"

        # Humidity string (00 = 100%)
        hum_str = ".."
        if hum_val is not None:
            hum_str = "00" if hum_val >= 100 else f"{hum_val:02d}"

        # 4. Construct the APRS String
        wx_payload = f"{time_str}{lat_str}/{lon_str}_c{wdir:03d}s{wspd_mph:03d}g{wgst_mph}{temp_str}r{r_1h}p{p_24h}P{P_mid}h{hum_str}b{barom_mb_tenths:05d}{solar_str}"
        packet = f"{cwop_id}>APRS,TCPIP*:@{wx_payload}AllskyBridge\r\n"

        # 5. Send via TCP to CWOP Servers
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(("cwop.aprs.net", 14580))
            s.sendall(f"user {cwop_id} pass -1 vers AllskyCloud 1.3\r\n".encode('utf-8'))
            s.sendall(packet.encode('utf-8'))
            print(f"  CWOP Success: {packet.strip()}")
            
    except Exception as e:
        print(f"  CWOP Error: {e}")

def send_to_lpm(sensors, lpm_key):
    """Sends SQM data to LightPollutionMap.info"""
    try:
        if not lpm_key: return
        mpsas = sensors.get("sky_brightness_mpsas")
        if mpsas is None or mpsas < 15.0:
            return

        url = "https://www.lightpollutionmap.info/sqm/submit_sqm.php"
        data = {
            'key': lpm_key,
            'sqm': f"{mpsas:.2f}",
            'dt': datetime.now().strftime('%Y-%m-%d %H:%M:%00')
        }
        r = requests.post(url, data=data, timeout=10)
        print(f"[{datetime.now()}] LPM Success ({mpsas:.2f}): {r.text.strip()}")
    except Exception as e:
        print(f"LPM Error: {e}")

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
                    send_to_cwop(data, CWOP_ID, LAT, LON, source="AmbientWeather")
                return Response(content='{"status":"success"}', media_type="application/json")
            except Exception as e:
                print(f"Interceptor Error: {e}")
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
    
    # Trigger External Updates (from ESP32)
    if CWOP_ID:
        send_to_cwop(payload.esp32_sensors, CWOP_ID, LAT, LON, source="ESP32")
    if LPM_API_KEY:
        send_to_lpm(payload.esp32_sensors, LPM_API_KEY)
        
    return {"status": "success", "id": db_capture.id}

@app.get("/sqm")
def get_sqm_unihedron(db: Session = Depends(get_db)):
    """Mimics a Unihedron SQM-LE 'rx' response."""
    latest = db.query(models.Capture).filter(models.Capture.esp32_sensors['sky_brightness_mpsas'].astext != None).order_by(models.Capture.timestamp.desc()).first()
    if not latest:
        return Response(content="r, 00.00m,0000000000Hz,0000000000c,0000000.000s, 000.0C\r\n", media_type="text/plain")
    
    sqm = latest.esp32_sensors.get("sky_brightness_mpsas", 0.0)
    temp = latest.esp32_sensors.get("temp", 0.0)
    response = f"r, {sqm:05.2f}m,0000000000Hz,0000000000c,0000000.000s, {temp:05.1f}C\r\n"
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
