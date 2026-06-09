import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from thermal_utils import reshape_thermal, fill_corners_clear, ambient_from_sensors

def load_thermal_data(thermal_bmp_path):
    json_path = Path(thermal_bmp_path).with_suffix('.json')
    if not json_path.exists(): return None, None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'frame' in data:
            ambient = ambient_from_sensors(data.get('sensors', {}))
            frame2d, _ = reshape_thermal(data['frame'])
            # Clipped corners read warm -> would become false cloud after warp.
            return fill_corners_clear(frame2d), ambient
    except Exception: pass
    return None, None

def get_hybrid_mask(img_rgb, thermal_frame, ambient, abs_thresh=-3.0):  # windowed (was -20 bare-sky)
    h, w = img_rgb.shape[:2]
    
    # 1. Thermal Confidence (The Absolute Truth)
    t_min = abs_thresh - 10.0
    t_max = abs_thresh + 10.0
    t_conf = np.clip((thermal_frame - t_min) / (t_max - t_min), 0, 1)
    t_mask = cv2.resize(t_conf, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. Visual Features (The Edge Refiners)
    b, g, r = cv2.split(img_rgb.astype(np.float32))
    nrbr = (r - b) / (r + b + 1e-6)
    v_color = np.clip((nrbr + 0.35) / 0.4, 0, 1)
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    v_text = np.clip(np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1)**2) / 25.0, 0, 1)
    v_mask = (v_color * 0.7) + (v_text * 0.3)

    # 3. Dynamic Calibration
    # Find the average intensity of the visual mask where thermal says it is CLEAR SKY
    sky_mask = (t_mask < 0.1).astype(np.float32)
    if np.sum(sky_mask) > 100:
        sky_bias = np.sum(v_mask * sky_mask) / np.sum(sky_mask)
        # Subtract the sky bias from the whole visual mask to remove atmospheric ghosting
        v_mask = np.clip(v_mask - (sky_bias * 0.8), 0, 1)
        # Boost the remaining visual signal to match thermal cloud intensity
        v_mask = np.clip(v_mask * 1.5, 0, 1)

    # 4. Seamless Fusion
    # We use a soft transition mask (feather) to hide the boundary
    feather = np.ones((h, w), dtype=np.float32)
    fs = int(min(h, w) * 0.2)
    cv2.rectangle(feather, (0,0), (w-1, h-1), 0, fs*2)
    feather = cv2.stackBlur(feather, (fs*2+1, fs*2+1))

    # Inside: result = thermal weighted by visual texture
    # Outside: result = visual only
    # The trick: use thermal to "gate" the visual signal in the center
    t_gated_v = np.maximum(t_mask, v_mask * 0.5) 
    
    combined = (t_gated_v * feather) + (v_mask * (1.0 - feather))
    
    # Final sharpening
    combined = np.power(np.clip(combined, 0, 1), 1.2)
    
    return (combined * 255).astype(np.uint8)

def build_remap_matrices(allsky_w, allsky_h, thermal_w, thermal_h, 
                         allsky_fov_deg, thermal_fov_deg, rotation_deg, offset_x_pct, offset_y_pct, distortion, proj_on):
    cx_a, cy_a = (allsky_w / 2.0) + (offset_x_pct * (allsky_w / 2.0)), (allsky_h / 2.0) + (offset_y_pct * (allsky_h / 2.0))
    R_a = allsky_w / 2.0
    max_theta_a = np.radians(allsky_fov_deg / 2.0)
    thermal_fov_rad = np.radians(thermal_fov_deg)
    f_t = (thermal_w / 2.0) / np.tan(thermal_fov_rad / 2.0)
    cx_t, cy_t = thermal_w / 2.0, thermal_h / 2.0
    X, Y = np.meshgrid(np.arange(allsky_w), np.arange(allsky_h))
    dx, dy = X - cx_a, Y - cy_a
    r, phi = np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx) + np.radians(rotation_deg)
    if not proj_on:
        scale = (thermal_fov_deg / allsky_fov_deg) * allsky_w
        s_factor = thermal_w / scale
        dx_rot, dy_rot = r * np.cos(phi), r * np.sin(phi)
        map_x, map_y = cx_t + dx_rot * s_factor, cy_t + dy_rot * s_factor
        invalid = (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
        map_x[invalid], map_y[invalid] = -1, -1
        return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid
    r_norm = r / R_a
    theta = (r_norm ** (2.0 ** distortion)) * max_theta_a
    valid_theta = theta < (np.pi / 2 - 0.01)
    d_t = np.zeros_like(theta)
    d_t[valid_theta] = f_t * np.tan(theta[valid_theta])
    map_x, map_y = cx_t + d_t * np.cos(phi), cy_t + d_t * np.sin(phi)
    invalid = ~valid_theta | (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
    map_x[invalid], map_y[invalid] = -1, -1
    return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid

def process_pair(allsky_p, thermal_p, config, args):
    img_a_full = cv2.imread(str(allsky_p))
    thermal_raw, ambient = load_thermal_data(thermal_p)
    if img_a_full is None or thermal_raw is None: return
    if config.get('flip_h', 0) and config.get('flip_v', 1): thermal_raw = np.flip(thermal_raw, (0, 1))
    elif config.get('flip_h', 0): thermal_raw = np.flip(thermal_raw, 1)
    elif config.get('flip_v', 1): thermal_raw = np.flip(thermal_raw, 0)
    a_h, a_w = img_a_full.shape[:2]
    t_h, t_w = thermal_raw.shape[:2]
    map_x, map_y, valid_mask = build_remap_matrices(a_w, a_h, t_w, t_h, 180.0, config['fov'], config['rot'], config['x_off'], config['y_off'], config['dist'], config.get('proj_on', 1))
    coords = np.argwhere(valid_mask)
    if len(coords) == 0: return
    pad = 30
    y0, x0 = max(0, coords[:,0].min()-pad), max(0, coords[:,1].min()-pad)
    y1, x1 = min(a_h, coords[:,0].max()+pad), min(a_w, coords[:,1].max()+pad)
    img_crop, map_x_crop, map_y_crop = img_a_full[y0:y1, x0:x1], map_x[y0:y1, x0:x1], map_y[y0:y1, x0:x1]
    warped_thermal = cv2.remap(thermal_raw, map_x_crop, map_y_crop, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hybrid_mask = get_hybrid_mask(img_crop, warped_thermal, ambient, abs_thresh=args.abs_thresh)
    if args.size > 0:
        img_crop = cv2.resize(img_crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
        hybrid_mask = cv2.resize(hybrid_mask, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True)
    (out_path / 'images').mkdir(exist_ok=True); (out_path / 'masks').mkdir(exist_ok=True)
    cv2.imwrite(str(out_path / 'images' / f'{allsky_p.stem}.jpg'), img_crop)
    cv2.imwrite(str(out_path / 'masks' / f'{allsky_p.stem}.png'), hybrid_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--allsky-root', type=str, default='/Volumes/allsky_images')
    parser.add_argument('--thermal-root', type=str, default='/Volumes/astro_image_thermal')
    parser.add_argument('--uuid', type=str, default='ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212')
    parser.add_argument('--config', default='allsky-cloud-analysis/alignment_config.json')
    parser.add_argument('--output-dir', type=str, default='ml_dataset_hybrid')
    parser.add_argument('--size', type=int, default=256)
    # Windowed-hardware cutoff (ZnSe ~+19C warm offset; clear sky ~-6C). Was
    # -20 (bare sky) which marked everything as cloud. Matches the firmware's
    # CLOUD_PIXEL_ABS_CUTOFF (-3). Refine on a confirmed clear night.
    parser.add_argument('--abs-thresh', type=float, default=-3.0)
    parser.add_argument('--max-pairs', type=int, default=0)
    args = parser.parse_args()
    input_p = Path(args.input_path).resolve()
    with open(args.config, 'r') as f: config = json.load(f)
    pairs = []
    if input_p.is_dir():
        all_jpgs = sorted(list(input_p.rglob('*.[jJ][pP][gG]')))
        for a_p in all_jpgs:
            if 'thumbnails' in str(a_p): continue
            parts = a_p.parts
            try:
                img_idx = parts.index('images')
                rel = Path(*parts[img_idx+1:])
                t_p = Path(args.thermal_root) / args.uuid / 'exposures' / rel.with_suffix('.bmp')
                if t_p.exists(): pairs.append((a_p, t_p))
            except ValueError: pass
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        indices = np.linspace(0, len(pairs)-1, args.max_pairs, dtype=int)
        pairs = [pairs[i] for i in indices]
    print(f'Generating Calibrated Hybrid dataset for {len(pairs)} pairs...')
    for i, (a_p, t_p) in enumerate(pairs):
        process_pair(a_p, t_p, config, args)
        if i % 5 == 0: print(f'  {i}/{len(pairs)}...', end='\r')
    print(f'\nDone.')

if __name__ == '__main__':
    main()
