import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse

def load_thermal_image(thermal_bmp_path):
    json_path = thermal_bmp_path.with_suffix('.json')
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if 'frame' in data:
                raw_frame = np.array(data['frame'], dtype=np.float32).reshape((24, 32))
                min_val, max_val = np.min(raw_frame), np.max(raw_frame)
                if max_val > min_val:
                    norm_frame = ((raw_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    norm_frame = np.zeros_like(raw_frame, dtype=np.uint8)
                return cv2.applyColorMap(norm_frame, cv2.COLORMAP_INFERNO)
        except Exception:
            pass
    return cv2.imread(str(thermal_bmp_path))

def build_flat_map(a_w, a_h, t_w, t_h, allsky_fov_deg, thermal_fov_deg, rot_deg, offset_x_pct, offset_y_pct):
    cx_a = (a_w / 2.0) + (offset_x_pct * (a_w / 2.0))
    cy_a = (a_h / 2.0) + (offset_y_pct * (a_h / 2.0))
    cx_t, cy_t = t_w / 2.0, t_h / 2.0
    
    X, Y = np.meshgrid(np.arange(a_w), np.arange(a_h))
    dx = X - cx_a
    dy = Y - cy_a
    
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) + np.radians(rot_deg)
    
    scale = (thermal_fov_deg / allsky_fov_deg) * a_w
    s_factor = t_w / scale
    
    dx_rot = r * np.cos(phi)
    dy_rot = r * np.sin(phi)
    
    map_x = cx_t + dx_rot * s_factor
    map_y = cy_t + dy_rot * s_factor
    
    invalid = (map_x < 0) | (map_x >= t_w) | (map_y < 0) | (map_y >= t_h)
    map_x[invalid] = -1
    map_y[invalid] = -1
    return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid

def generate_8way_grid():
    date_str = "20260508"
    allsky_dir = Path("/Volumes/allsky_images/images") / date_str
    thermal_dir = Path("/Volumes/astro_image_thermal/ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212/exposures") / date_str

    allsky_files = sorted(list(allsky_dir.glob("**/*.jpg")))
    thermal_files = list(thermal_dir.glob("**/*.bmp"))
    thermal_index = {f.stem: f for f in thermal_files}
    
    valid_pairs = []
    for allsky_p in allsky_files:
        if "thumbnails" in str(allsky_p): continue
        if allsky_p.stem in thermal_index:
            valid_pairs.append((allsky_p, thermal_index[allsky_p.stem]))
            
    valid_pairs = valid_pairs[800:1000]
    
    if not valid_pairs:
        print("No valid pairs found in range.")
        return

    img_a_test = cv2.imread(str(valid_pairs[0][0]))
    a_h, a_w = img_a_test.shape[:2]
    
    # We will use FLAT projection to rule out fisheye math weirdness for this directional test.
    # Base params (from your previous tests)
    fov, rot, x_off, y_off = 40, 28, -0.03, 0.08
    alpha = 0.8
    t_h, t_w = 24, 32

    # Precompute maps for the 4 base rotations
    maps = {}
    for r_base in [0, 90, 180, 270]:
        maps[r_base] = build_flat_map(a_w, a_h, t_w, t_h, 180.0, fov, rot + r_base, x_off, y_off)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    grid_size = 500
    out_w = grid_size * 4
    out_h = grid_size * 2
    video_writer = cv2.VideoWriter("8way_directional_test.mp4", fourcc, 10.0, (out_w, out_h))

    count = 0
    for allsky_p, thermal_p in valid_pairs:
        img_a = cv2.imread(str(allsky_p))
        img_t_raw = load_thermal_image(thermal_p)
        
        panels = []
        
        # Top Row: Normal (Not Flipped)
        for r_base in [0, 90, 180, 270]:
            map_x, map_y, valid_mask = maps[r_base]
            warped = cv2.remap(img_t_raw, map_x, map_y, cv2.INTER_CUBIC, borderValue=(0,0,0))
            blended = img_a.copy()
            blended[valid_mask] = (blended[valid_mask] * (1-alpha) + warped[valid_mask] * alpha).astype(np.uint8)
            resized = cv2.resize(blended, (grid_size, grid_size))
            cv2.putText(resized, f"Norm, Rot+{r_base}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            panels.append(resized)
            
        top_row = np.hstack(panels)
        
        # Bottom Row: Flipped Horizontally
        panels = []
        img_t_flipped = cv2.flip(img_t_raw, 1) # Flip horizontally
        for r_base in [0, 90, 180, 270]:
            map_x, map_y, valid_mask = maps[r_base]
            warped = cv2.remap(img_t_flipped, map_x, map_y, cv2.INTER_CUBIC, borderValue=(0,0,0))
            blended = img_a.copy()
            blended[valid_mask] = (blended[valid_mask] * (1-alpha) + warped[valid_mask] * alpha).astype(np.uint8)
            resized = cv2.resize(blended, (grid_size, grid_size))
            cv2.putText(resized, f"Flip H, Rot+{r_base}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            panels.append(resized)
            
        bottom_row = np.hstack(panels)
        
        final_grid = np.vstack([top_row, bottom_row])
        video_writer.write(final_grid)
        count += 1
        print(f"Processed {count}/200...", end='\r')
        
    video_writer.release()
    print("\nSaved 8way_directional_test.mp4")

if __name__ == "__main__":
    generate_8way_grid()