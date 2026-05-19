# Allsky & Thermal Cloud Analysis: Core Algorithms

This document outlines the mathematical and statistical algorithms used to process, align, and analyze the thermal weather station data.

## 1. Multi-Modal Alignment (Projection Math)

Mapping a low-resolution rectilinear thermal sensor (MLX90640) onto a high-resolution hemispherical fisheye image requires spherical-to-planar geometric projection. This logic powers `visual_aligner.py` and all overlay/crop scripts.

### Pinhole to Spherical Mapping
1. **Fisheye Model:** The All-Sky camera uses an equidistant fisheye lens. The distance from the center of the image ($r$) is linearly proportional to the zenith angle ($\theta$).
2. **Thermal Model:** The MLX90640 is modeled as a standard rectilinear (pinhole) lens.
3. **The Projection:** 
   * We calculate the polar coordinates $(r, \phi)$ of every pixel in the visual all-sky image.
   * From $r$, we derive the zenith angle $\theta$. 
   * We project that angle onto the flat thermal plane using $d_t = f_t \cdot \tan(\theta)$, where $f_t$ is the focal length of the thermal lens derived from its configured Horizontal Field of View (FOV).
4. **Distortion Correction:** To handle physical imperfections in the fisheye lens, we apply a non-linear `distortion` curve ($\gamma = 2^{\text{dist}}$). The zenith angle is adjusted as $\theta = (r_{norm}^\gamma) \cdot \theta_{max}$.

## 2. Auto-Alignment (Mutual Information Optimization)

Direct pixel comparison (like Mean Squared Error or structural similarity) fails when aligning visual and thermal data because they measure fundamentally different physical properties (reflected light vs. emitted infrared radiation).

### Mutual Information (MI)
The `auto_align.py` script uses **Mutual Information**, a concept from information theory, to measure the statistical dependence between the two modalities.
* It computes a 2D joint histogram: visual intensities (X-axis) vs. thermal temperatures (Y-axis).
* **Misaligned:** The joint histogram is diffuse and noisy. Knowing a pixel is visually bright tells us nothing about its temperature.
* **Aligned:** Bright/warm pixels (clouds) and dark/cold pixels (clear sky) form tight, highly predictable clusters. The entropy is minimized, and the MI score peaks:
  $$MI(X,Y) = \sum_{x,y} P_{XY}(x,y) \log \frac{P_{XY}(x,y)}{P_X(x) P_Y(y)}$$

### Coordinate Descent 
To find the exact physical mounting parameters (FOV, Rotation, X Offset, Y Offset):
* The optimizer loads a batch of image pairs to avoid overfitting to a single cloud shape.
* It tests small steps (e.g., $+1.0$ and $-1.0$) in each dimensional axis.
* If a step increases the average MI score across the batch, it adopts the new parameters.
* When a local maximum is reached, it halves the step size (zooming in for sub-pixel accuracy) and repeats until convergence.

## 3. Thermal Cloud Detection (ESPHome Firmware)

To eliminate false positives caused by the warm horizon (air mass thickness) and internal electronic case heat, the raw 32x24 thermal array is **cropped to 24x16 (384 pixels)** directly on the ESP32. 

The weather condition is then calculated using two concurrent masking methods:

### Absolute vs. Relative Thresholding
1. **Absolute Threshold:** Any pixel reading warmer than `-5.0°C` is classified as a cloud. Water vapor in the troposphere rarely drops below this in the IR window, making it a reliable ambient-independent cutoff.
2. **Ambient-Relative Threshold (Delta):** Any pixel warmer than `Ambient - 10.0°C` is classified as a cloud. This dynamically catches high, thin cirrus clouds even on extremely cold winter nights when the absolute threshold might fail.

**The Verdict:** The firmware calculates a Cloud Fraction (%) for both methods, takes the most "pessimistic" (highest) fraction, and maps it to a standard meteorological label (e.g., Very Clear, Partly Cloudy, Overcast).
