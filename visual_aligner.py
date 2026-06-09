import cv2
import numpy as np
import argparse
from pathlib import Path
from thermal_utils import reshape_thermal, fill_corners_clear, KEEP

def build_maps(a_w, a_h, t_w, t_h, allsky_fov_deg, thermal_fov_deg, rot_deg, offset_x_pct, offset_y_pct, distortion, proj_on):
    # Apply offsets to the optical center of the allsky image
    cx_a = (a_w / 2.0) + (offset_x_pct * (a_w / 2.0))
    cy_a = (a_h / 2.0) + (offset_y_pct * (a_h / 2.0))
    
    R_a = a_w / 2.0
    max_theta_a = np.radians(allsky_fov_deg / 2.0)
    
    # Avoid division by zero if slider goes to 0
    thermal_fov_deg = max(1.0, thermal_fov_deg)
    thermal_fov_rad = np.radians(thermal_fov_deg)
    f_t = (t_w / 2.0) / np.tan(thermal_fov_rad / 2.0)
    cx_t, cy_t = t_w / 2.0, t_h / 2.0
    
    X, Y = np.meshgrid(np.arange(a_w), np.arange(a_h))
    dx = X - cx_a
    dy = Y - cy_a
    
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) + np.radians(rot_deg)

    if not proj_on:
        # Flat/linear mapping (bypass fisheye/spherical projection)
        # Use thermal_fov_deg as a generic scale factor
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
    
    # Distortion: smoothly curves the projection mapping
    # distortion=0 means linear (gamma=1.0)
    r_norm = r / R_a
    gamma = 2.0 ** distortion
    theta = (r_norm ** gamma) * max_theta_a
    
    valid_theta = theta < (np.pi / 2 - 0.01)
    
    d_t = np.zeros_like(theta)
    d_t[valid_theta] = f_t * np.tan(theta[valid_theta])
    
    map_x = cx_t + d_t * np.cos(phi)
    map_y = cy_t + d_t * np.sin(phi)
    
    invalid = ~valid_theta | (map_x < 0) | (map_x >= t_w) | (map_y < 0) | (map_y >= t_h)
    map_x[invalid] = -1
    map_y[invalid] = -1
    
    return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid

import json

def load_thermal_image(thermal_bmp_path):
    # Check if a .json file exists alongside the .bmp
    json_path = thermal_bmp_path.with_suffix('.json')
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if 'frame' in data:
                raw_frame = fill_corners_clear(reshape_thermal(data['frame'])[0])
                
                # Normalize the temperatures to 0-255
                min_val = np.min(raw_frame)
                max_val = np.max(raw_frame)
                
                if max_val > min_val:
                    norm_frame = ((raw_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    norm_frame = np.zeros_like(raw_frame, dtype=np.uint8)
                
                return norm_frame  # Return 1-channel grayscale to be colored dynamically
        except Exception as e:
            print(f"Failed to parse json {json_path}: {e}")
            print("Falling back to BMP.")

    # Fallback to the original BMP if json doesn't exist or failed
    return cv2.imread(str(thermal_bmp_path))

def nothing(x):
    pass

def main(input_file, allsky_root, thermal_root, ccd_uuid):
    input_p = Path(input_file).resolve()
    
    # Determine if input is allsky or thermal based on the path
    is_allsky = 'images' in input_p.parts
    is_thermal = 'exposures' in input_p.parts
    
    if is_allsky:
        try:
            idx = input_p.parts.index('images')
            rel_parts = input_p.parts[idx+1:] if input_p.is_dir() else input_p.parts[idx+1:-1]
        except ValueError:
            print("Could not parse allsky path. Make sure it contains an 'images' directory.")
            return
            
        allsky_dir = input_p.parent
        thermal_dir = Path(thermal_root) / ccd_uuid / "exposures" / Path(*rel_parts)
    elif is_thermal:
        try:
            idx = input_p.parts.index('exposures')
            rel_parts = input_p.parts[idx+1:] if input_p.is_dir() else input_p.parts[idx+1:-1]
        except ValueError:
            print("Could not parse thermal path. Make sure it contains an 'exposures' directory.")
            return
            
        thermal_dir = input_p.parent
        allsky_dir = Path(allsky_root) / "images" / Path(*rel_parts)
    else:
        print("Input path must contain either 'images' (Allsky) or 'exposures' (Thermal) in its directory structure.")
        return

    if not allsky_dir.exists():
        print(f"Allsky directory not found: {allsky_dir}")
        return
    if not thermal_dir.exists():
        print(f"Thermal directory not found: {thermal_dir}")
        return

        print(f"Scanning Allsky: {allsky_dir}")
    print(f"Scanning Thermal: {thermal_dir}")
    allsky_files = sorted(list(allsky_dir.glob("*.[jJ][pP][gG]")))
    print(f"  Found {len(allsky_files)} Allsky files.")
    
    valid_pairs = []
    start_idx = 0
    for f in allsky_files:
        if "thumbnails" in str(f):
            continue
        t_file = thermal_dir / f"{f.stem}.bmp"
        if t_file.exists():
            valid_pairs.append((f, t_file))
            if f.name == input_p.name or t_file.name == input_p.name:
                start_idx = len(valid_pairs) - 1

    if not valid_pairs:
        print("No matching image pairs found in the corresponding directories!")
        return

    print(f"Found {len(valid_pairs)} matching pairs.")
    
    # Set up the window so it fits on screen and the sliders don't disappear on macOS
    cv2.namedWindow('Visual Aligner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Visual Aligner', 800, 1000)
    
    # Load Config
    config_path = Path("alignment_config.json")
    config = {
        "proj_on": 1, "flip_h": 0, "flip_v": 1,
        "fov": 74, "rot": 202, "x_off": -0.04,
        "y_off": 0.12, "dist": 0.0, "alpha": 1.0, "cmap": 11, "render_mode": 0, "threshold_val": 128
    }
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config.update(json.load(f))
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Failed to load config: {e}")

    # Create Trackbars
    cv2.createTrackbar('Proj 1=On', 'Visual Aligner', config['proj_on'], 1, nothing)
    cv2.createTrackbar('Flip H', 'Visual Aligner', config['flip_h'], 1, nothing)
    cv2.createTrackbar('Flip V', 'Visual Aligner', config['flip_v'], 1, nothing)
    cv2.createTrackbar('FOV (deg)', 'Visual Aligner', int(config['fov']), 120, nothing)
    cv2.createTrackbar('Rot (deg)', 'Visual Aligner', int(config['rot']), 360, nothing)
    cv2.createTrackbar('X Offset', 'Visual Aligner', int((config['x_off'] * 100) + 100), 200, nothing)
    cv2.createTrackbar('Y Offset', 'Visual Aligner', int((config['y_off'] * 100) + 100), 200, nothing)
    cv2.createTrackbar('Distort', 'Visual Aligner', int((config['dist'] * 50) + 100), 200, nothing)
    cv2.createTrackbar('Alpha %', 'Visual Aligner', int(config['alpha'] * 100), 100, nothing)
    cv2.createTrackbar('Colormap', 'Visual Aligner', int(config.get('cmap', 11)), 21, nothing)
    cv2.createTrackbar('Render Mode', 'Visual Aligner', config.get('render_mode', 0), 3, nothing)
    cv2.createTrackbar('Thresh Val', 'Visual Aligner', config.get('threshold_val', 128), 255, nothing)

    print("\n==========================================")
    print("             VISUAL ALIGNER               ")
    print("==========================================")
    print(" Adjust the sliders to align the overlay. ")
    print("------------------------------------------")
    print(" CONTROLS:")
    print(" [ a ] or [ <- ] : Previous Image")
    print(" [ d ] or [ -> ] : Next Image")
    print(" [ p ]           : Print Parameters")
    print(" [ s ]           : Save Parameters to Config")
    print(" [ q ] or ESC    : Quit")
    print("==========================================\n")

    current_idx = start_idx
    prev_state = None
    force_update = True
    
    img_a_ui = None
    img_t_raw = None
    a_h = a_w = t_h = t_w = 0
    map_x = map_y = valid_mask = None

    def load_images(idx):
        nonlocal img_a_ui, img_t_raw, a_h, a_w, t_h, t_w, force_update, current_idx
        n = len(valid_pairs)
        # Find the next loadable pair starting at idx; skip unreadable/corrupt
        # frames instead of crashing on a None image.
        for off in range(n):
            j = (idx + off) % n
            a_path, t_path = valid_pairs[j]
            img_a = cv2.imread(str(a_path))
            it = load_thermal_image(t_path)
            if img_a is None or it is None:
                print(f"  skip [{j+1}/{n}] {a_path.name}: unreadable allsky or thermal")
                continue
            img_t_raw = it
            current_idx = j
            UI_SIZE = 800  # Resize allsky for a fluid real-time UI
            img_a_ui = cv2.resize(img_a, (UI_SIZE, UI_SIZE), interpolation=cv2.INTER_AREA)
            a_h, a_w = img_a_ui.shape[:2]
            t_h, t_w = img_t_raw.shape[:2]
            print(f"Loaded [{j+1}/{n}]: {a_path.name}")
            force_update = True
            return True
        print("No loadable thermal/allsky pairs found.")
        return False

    # Load the initial image
    if not load_images(current_idx):
        return

    while True:
        proj_on = cv2.getTrackbarPos('Proj 1=On', 'Visual Aligner')
        flip_h = cv2.getTrackbarPos('Flip H', 'Visual Aligner')
        flip_v = cv2.getTrackbarPos('Flip V', 'Visual Aligner')
        fov = cv2.getTrackbarPos('FOV (deg)', 'Visual Aligner')
        rot = cv2.getTrackbarPos('Rot (deg)', 'Visual Aligner')
        x_off_raw = cv2.getTrackbarPos('X Offset', 'Visual Aligner')
        y_off_raw = cv2.getTrackbarPos('Y Offset', 'Visual Aligner')
        dist_raw = cv2.getTrackbarPos('Distort', 'Visual Aligner')
        alpha_pct = cv2.getTrackbarPos('Alpha %', 'Visual Aligner')
        cmap_idx = cv2.getTrackbarPos('Colormap', 'Visual Aligner')
        render_mode = cv2.getTrackbarPos('Render Mode', 'Visual Aligner')
        thresh_val = cv2.getTrackbarPos('Thresh Val', 'Visual Aligner')

        # Convert trackbar values to actual math values
        x_off = (x_off_raw - 100) / 100.0  # -1.0 to 1.0
        y_off = (y_off_raw - 100) / 100.0  # -1.0 to 1.0
        dist = (dist_raw - 100) / 50.0     # -2.0 to 2.0
        alpha = alpha_pct / 100.0

        current_state = (proj_on, flip_h, flip_v, fov, rot, x_off, y_off, dist, alpha, cmap_idx, render_mode, thresh_val)

        if current_state != prev_state or force_update:
            # Apply flips to thermal image
            img_t = img_t_raw.copy()
            if flip_h and flip_v:
                img_t = cv2.flip(img_t, -1)
            elif flip_h:
                img_t = cv2.flip(img_t, 1)
            elif flip_v:
                img_t = cv2.flip(img_t, 0)

            if current_state[:8] != (prev_state[:8] if prev_state else None) or force_update:
                # Recalculate maps if anything but alpha changed
                map_x, map_y, valid_mask = build_maps(a_w, a_h, t_w, t_h, 180.0, fov, rot, x_off, y_off, dist, proj_on)
            
            # Warp thermal (1-channel grayscale)
            warped_gray = cv2.remap(img_t, map_x, map_y, cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))
            
            if render_mode == 1:
                # Mode 1: Canny Edges (Neon Green)
                edges = cv2.Canny(warped_gray, 40, 120)
                thermal_warped = np.zeros((a_h, a_w, 3), dtype=np.uint8)
                thermal_warped[edges > 0] = [0, 255, 0]
            elif render_mode == 2:
                # Mode 2: Iso-Contours (Cyan Topography)
                quantized = (warped_gray // 32) * 32
                edges = cv2.Canny(quantized.astype(np.uint8), 10, 50)
                thermal_warped = np.zeros((a_h, a_w, 3), dtype=np.uint8)
                thermal_warped[edges > 0] = [255, 255, 0]
            elif render_mode == 3:
                # Mode 3: Threshold Mask (Black & White shape)
                _, thresh = cv2.threshold(warped_gray, thresh_val, 255, cv2.THRESH_BINARY)
                thermal_warped = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            else:
                # Mode 0: Dynamic Colormap
                thermal_warped = cv2.applyColorMap(warped_gray, cmap_idx)
            
            # Blend
            alpha_mask = np.zeros((a_h, a_w, 1), dtype=np.float32)
            alpha_mask[valid_mask] = alpha

            # Clipped enclosure corners -> 0 alpha (they vanish). Warp the corner
            # KEEP mask through the same projection and fold it into the alpha.
            keep_t = KEEP.astype(np.float32)
            if keep_t.shape == (t_h, t_w):
                kf = keep_t
                if flip_h and flip_v: kf = cv2.flip(kf, -1)
                elif flip_h: kf = cv2.flip(kf, 1)
                elif flip_v: kf = cv2.flip(kf, 0)
                warped_keep = cv2.remap(kf, map_x, map_y, cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                alpha_mask = alpha_mask * warped_keep[:, :, None]

            blended = img_a_ui.astype(np.float32) * (1.0 - alpha_mask) + \
                      thermal_warped.astype(np.float32) * alpha_mask
            
            display_img = blended.astype(np.uint8)
            
            # Add crosshairs to help find the center
            cv2.line(display_img, (a_w//2, 0), (a_w//2, a_h), (0, 255, 0), 1)
            cv2.line(display_img, (0, a_h//2), (a_w, a_h//2), (0, 255, 0), 1)
            
            cv2.imshow('Visual Aligner', display_img)
            prev_state = current_state
            force_update = False

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # ESC
            break
        elif key == ord('p'):
            proj_str = "" if proj_on else " --no-projection"
            fh_str = " --flip-h" if flip_h else ""
            fv_str = " --flip-v" if flip_v else ""
            print("\n--- Current Parameters ---")
            print(f"--thermal-fov {fov} --rotation {rot} --offset-x {x_off:.2f} --offset-y {y_off:.2f} --distortion {dist:.2f} --alpha {alpha:.2f}{proj_str}{fh_str}{fv_str} (Colormap: {cmap_idx})")
        elif key == ord('s'):
            save_data = {
                "proj_on": proj_on, "flip_h": flip_h, "flip_v": flip_v,
                "fov": fov, "rot": rot, "x_off": x_off,
                "y_off": y_off, "dist": dist, "alpha": alpha, "cmap": cmap_idx, "render_mode": render_mode, "threshold_val": thresh_val
            }
            try:
                with open(config_path, 'w') as f:
                    json.dump(save_data, f, indent=4)
                    print(f"\n[SUCCESS] Saved current parameters to {config_path}")
            except Exception as e:
                print(f"\n[ERROR] Failed to save config: {e}")
        elif key == ord('d') or key == 83:  # 'd' or Right Arrow
            if current_idx < len(valid_pairs) - 1:
                current_idx += 1
                load_images(current_idx)
        elif key == ord('a') or key == 81:  # 'a' or Left Arrow
            if current_idx > 0:
                current_idx -= 1
                load_images(current_idx)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive Thermal-Allsky Aligner')
    parser.add_argument('input_file', type=str, help='Path to either an Allsky JPG or a Thermal BMP (tool will automatically find the matching pair)')
    parser.add_argument('--allsky-root', type=str, default='/Volumes/allsky_images', help='Allsky root path')
    parser.add_argument('--thermal-root', type=str, default='/Volumes/astro_image_thermal', help='Thermal root path')
    parser.add_argument('--uuid', type=str, default='ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212', help='Thermal CCD UUID')
    args = parser.parse_args()
    
    main(args.input_file, args.allsky_root, args.thermal_root, args.uuid)