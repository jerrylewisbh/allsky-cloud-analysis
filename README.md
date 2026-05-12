# Allsky Cloud Analysis

This repository contains tools for aligning, projecting, and analyzing thermal imaging data alongside visual Allsky fisheye captures, including a complete containerized data ingestion pipeline.

## 🚀 Data Ingestion Pipeline (Backend)

The backend provides a central database and API to store allsky metadata, thermal frames, and synchronized weather data.

### Components
*   **Database (PostgreSQL):** Stores capture logs, raw thermal arrays, and weather station telemetry.
*   **API (FastAPI):** Orchestrates data ingestion and automatic time-syncing between cameras and weather stations.
*   **Orchestration (Docker):** Deployable on any Ubuntu/Linux server with Docker installed.

### Deployment on Ubuntu Server
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd allsky-cloud-analysis
    ```
2.  **Spin up the stack:**
    ```bash
    docker compose up -d
    ```
3.  **Configure Weather Station:**
    In your Ambient Weather (AWNET) settings, set the **Custom Server** to:
    *   **IP:** `YOUR_SERVER_IP`
    *   **Path:** `/weather`
    *   **Port:** `8000`

### Client Setup (Allsky Pi)
Move `scripts/sky_thermal_postsave_hook.py` to your Allsky Pi and configure the `ANALYSIS_API_URL` to point to your server. Add it as a post-save hook in your Allsky configuration.

---

## 🛠️ Analysis & Calibration Tools

### 1. `visual_aligner.py`
Interactive GUI for spatial alignment.
*   **Usage:** `python3 visual_aligner.py <path_to_image>`
*   Supports Allsky JPG or Thermal BMP/JSON.
*   **Controls:** Sliders for FOV, Rotation, X/Y Offsets, Distortion, and Mirroring.
*   **Keyboard:** `d` (Next), `a` (Prev), `p` (Print CLI params), `q` (Quit).

### 2. `matched_crop.py`
Generates perfectly aligned, identical-area crops of both visual and thermal images for scientific comparison.
*   **Usage:** `python3 matched_crop.py --allsky <path> --thermal <path> [calibration params]`

### 3. `overlay_timelapse.py`
Generates high-resolution timelapses with HDR thermal data mathematically projected onto the fisheye view.
*   **Usage:** `python3 overlay_timelapse.py --date 20260510 [calibration params]`

### 4. `side_by_side_timelapse.py`
Generates comparative stacked videos of the raw allsky and thermal heatmaps.

## 📦 Requirements
*   Python 3.9+
*   OpenCV (`opencv-python`)
*   NumPy
*   Requests (for hook)
```bash
pip install -r requirements.txt
```
