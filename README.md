# Allsky Cloud Analysis

This repository contains tools for aligning, projecting, and rendering high-dynamic-range (HDR) thermal imaging data over visual Allsky fisheye images to aid in cloud analysis.

## Tools Included

### 1. `visual_aligner.py`
An interactive GUI tool for calibrating the spatial alignment between the thermal sensor and the fisheye lens.
* **Usage:** `python3 visual_aligner.py <path_to_allsky_jpg_OR_thermal_bmp>`
* Automatically finds the matching pair and loads the rest of the folder.
* Provides sliders for FOV, Rotation, X/Y Offset, and Distortion Curve.
* Press `d` to skip to next frame, `a` for previous.
* Press `p` to print the exact CLI arguments needed for the timelapse generators.

### 2. `overlay_timelapse.py`
Generates a video with the thermal data mathematically projected onto the fisheye view using equidistant projection mapping.
* Automatically parses `.json` files to render high-contrast HDR thermal clouds.
* Supports arbitrary tilts and offsets via calibration parameters.

### 3. `side_by_side_timelapse.py`
Generates a comparative side-by-side or stacked vertical video of the raw allsky and thermal frames.

### 4. `8way_test.py`
A debugging script to generate a 2x4 grid testing all possible physical sensor orientations (rotations + mirrors) to determine correct mounting alignment.

## Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```
