#!/usr/bin/env python3
import sys
import json
import argparse
import logging
import urllib.request
import requests
from datetime import datetime, timezone
from pathlib import Path

# =====================
# === CONFIG (edit) ===
# =====================
SKY_THERMAL_URL          = 'http://10.0.0.242'
ANALYSIS_API_URL         = 'http://YOUR_UBUNTU_SERVER_IP:8000/ingest'
THERMAL_IMAGE_FOLDER     = '/var/www/html/allsky/thermal'
INDI_ALLSKY_IMAGE_FOLDER = '/var/www/html/allsky/images'
SKY_THERMAL_TIMEOUT      = 3
# =====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_json(url):
    try:
        with urllib.request.urlopen(url, timeout=SKY_THERMAL_TIMEOUT) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to fetch JSON from {url}: {e}")
        return None

def compute_thermal_path(image_path, image_folder, thermal_folder):
    try:
        rel = Path(image_path).resolve().relative_to(Path(image_folder).resolve())
    except ValueError:
        rel = Path(image_path.name)
    return (Path(thermal_folder) / rel).with_suffix('.bmp')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', help='Image File', type=str)
    args = parser.parse_args()

    image_path = Path(args.image_file)
    if not image_path.is_file():
        return 0

    # 1) Paths
    bmp_path = compute_thermal_path(args.image_file, INDI_ALLSKY_IMAGE_FOLDER, THERMAL_IMAGE_FOLDER)
    json_local_path = bmp_path.with_suffix('.json')

    # 2) Fetch Thermal Data from ESP32
    esp_data = fetch_json(f"{SKY_THERMAL_URL}/json")
    if not esp_data:
        return 0

    # Save local files
    bmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(f"{SKY_THERMAL_URL}/thermal.bmp", timeout=SKY_THERMAL_TIMEOUT) as resp:
            with open(bmp_path, 'wb') as f:
                f.write(resp.read())
        esp_data['fetched_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        with open(json_local_path, 'w') as f:
            json.dump(esp_data, f)
    except Exception as e:
        logger.error(f"Local save failed: {e}")

    # 3) Push to Backend (Backend handles weather station sync automatically)
    payload = {
        "allsky_path": str(image_path),
        "thermal_path": str(bmp_path),
        "thermal_frame": esp_data.get("frame", []),
        "esp32_sensors": esp_data.get("sensors", {})
    }

    try:
        requests.post(ANALYSIS_API_URL, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Failed to push to analysis API: {e}")

    return 0

if __name__ == '__main__':
    main()
