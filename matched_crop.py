import cv2
import numpy as np
import json
import argparse
from pathlib import Path

def load_thermal_image(thermal_bmp_path):
    json_path = Path(thermal_bmp_path).with_suffix('.json')
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
        except Exception: pass
    return cv2.imread(str(thermal_bmp_path))

def build_remap_matrices(allsky_w, allsky_h, thermal_w, thermal_h, 
                         allsky_fov_deg, thermal_fov_deg, rotation_deg, offset_x_pct, offset_y_pct, distortion, proj_on):
    cx_a = (allsky_w / 2.0) + (offset_x_pct * (allsky_w / 2.0))
    cy_a = (allsky_h / 2.0) + (offset_y_pct * (allsky_h / 2.0))
    R_a = allsky_w / 2.0
    max_theta_a = np.radians(allsky_fov_deg / 2.0)
    
    thermal_fov_rad = np.radians(thermal_fov_deg)
    f_t = (thermal_w / 2.0) / np.tan(thermal_fov_rad / 2.0)
    cx_t, cy_t = thermal_w / 2.0, thermal_h / 2.0
    
    X, Y = np.meshgrid(np.arange(allsky_w), np.arange(allsky_h))
    dx, dy = X - cx_a, Y - cy_a
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) + np.radians(rotation_deg)
    
    if not proj_on:
        scale = (thermal_fov_deg / allsky_fov_deg) * allsky_w
        s_factor = thermal_w / scale
        dx_rot, dy_rot = r * np.cos(phi), r * np.sin(phi)
        map_x, map_y = cx_t + dx_rot * s_factor, cy_t + dy_rot * s_factor
        invalid = (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
        map_x[invalid], map_y[invalid] = -1, -1
        return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid

    r_norm = r / R_a
    gamma = 2.0 ** distortion
    theta = (r_norm ** gamma) * max_theta_a
    valid_theta = theta < (np.pi / 2 - 0.01)
    d_t = np.zeros_like(theta)
    d_t[valid_theta] = f_t * np.tan(theta[valid_theta])
    map_x, map_y = cx_t + d_t * np.cos(phi), cy_t + d_t * np.sin(phi)
    invalid = ~valid_theta | (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
    map_x[invalid], map_y[invalid] = -1, -1
    return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid

def main():
    parser = argparse.ArgumentParser(description='Create matched crops of Allsky and Thermal images')
    parser.add_argument('--allsky', type=str, required=True)
    parser.add_argument('--thermal', type=str, required=True)
    parser.add_argument('--thermal-fov', type=float, default=55.0)
    parser.add_argument('--rotation', type=float, default=0.0)
    parser.add_argument('--offset-x', type=float, default=0.0)
    parser.add_argument('--offset-y', type=float, default=0.0)
    parser.add_argument('--distortion', type=float, default=0.0)
    parser.add_argument('--no-projection', action='store_true')
    parser.add_argument('--flip-h', action='store_true')
    parser.add_argument('--flip-v', action='store_true')
    parser.add_argument('--output-dir', type=str, default='crops')
    parser.add_argument('--size', type=int, default=0, help='Force output crops to square size (e.g. 512)')
    
    args = parser.parse_args()
    
    img_a = cv2.imread(args.allsky)
    img_t_raw = load_thermal_image(args.thermal)
    
    if args.flip_h and args.flip_v: img_t_raw = cv2.flip(img_t_raw, -1)
    elif args.flip_h: img_t_raw = cv2.flip(img_t_raw, 1)
    elif args.flip_v: img_t_raw = cv2.flip(img_t_raw, 0)
    
    a_h, a_w = img_a.shape[:2]
    t_h, t_w = img_t_raw.shape[:2]
    
    map_x, map_y, mask = build_remap_matrices(a_w, a_h, t_w, t_h, 180.0, 
                                            args.thermal_fov, args.rotation, 
                                            args.offset_x, args.offset_y, 
                                            args.distortion, not args.no_projection)
    
    warped_t = cv2.remap(img_t_raw, map_x, map_y, cv2.INTER_CUBIC, borderValue=(0,0,0))
    
    # Find bounding box of the valid thermal area
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    crop_a = img_a[y0:y1, x0:x1]
    crop_t = warped_t[y0:y1, x0:x1]
    
    if args.size > 0:
        crop_a = cv2.resize(crop_a, (args.size, args.size), interpolation=cv2.INTER_AREA)
        crop_t = cv2.resize(crop_t, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
    
    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True)
    
    base = Path(args.allsky).stem
    cv2.imwrite(str(out_path / f"{base}_allsky.jpg"), crop_a)
    cv2.imwrite(str(out_path / f"{base}_thermal.jpg"), crop_t)
    
    print(f"Saved matched crops to {args.output_dir}/")

if __name__ == "__main__":
    main()
