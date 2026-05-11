import cv2
import numpy as np
import argparse
from pathlib import Path

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
                raw_frame = np.array(data['frame'], dtype=np.float32)
                # Reshape to 24x32
                raw_frame = raw_frame.reshape((24, 32))
                
                # Normalize the temperatures to 0-255
                min_val = np.min(raw_frame)
                max_val = np.max(raw_frame)
                
                if max_val > min_val:
                    norm_frame = ((raw_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    norm_frame = np.zeros_like(raw_frame, dtype=np.uint8)
                
                # Apply a high dynamic range colormap (INFERNO is great for heat)
                color_frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_INFERNO)
                return color_frame
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
            rel_parts = input_p.parts[idx+1:-1]
        except ValueError:
            print("Could not parse allsky path. Make sure it contains an 'images' directory.")
            return
            
        allsky_dir = input_p.parent
        thermal_dir = Path(thermal_root) / ccd_uuid / "exposures" / Path(*rel_parts)
    elif is_thermal:
        try:
            idx = input_p.parts.index('exposures')
            rel_parts = input_p.parts[idx+1:-1]
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

    print(f"Scanning for matching pairs in {rel_parts}...")
    allsky_files = sorted(list(allsky_dir.glob("*.jpg")))
    
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
    
    # Create Trackbars
    cv2.createTrackbar('Proj 1=On', 'Visual Aligner', 1, 1, nothing)
    cv2.createTrackbar('Flip H', 'Visual Aligner', 0, 1, nothing)
    cv2.createTrackbar('Flip V', 'Visual Aligner', 0, 1, nothing)
    cv2.createTrackbar('FOV (deg)', 'Visual Aligner', 55, 120, nothing)
    cv2.createTrackbar('Rot (deg)', 'Visual Aligner', 0, 360, nothing)
    cv2.createTrackbar('X Offset', 'Visual Aligner', 100, 200, nothing)  # 100 is 0 offset
    cv2.createTrackbar('Y Offset', 'Visual Aligner', 100, 200, nothing)  # 100 is 0 offset
    cv2.createTrackbar('Distort', 'Visual Aligner', 100, 200, nothing)   # 100 is 0 distortion
    cv2.createTrackbar('Alpha %', 'Visual Aligner', 65, 100, nothing)

    print("\n==========================================")
    print("             VISUAL ALIGNER               ")
    print("==========================================")
    print(" Adjust the sliders to align the overlay. ")
    print("------------------------------------------")
    print(" CONTROLS:")
    print(" [ a ] or [ <- ] : Previous Image")
    print(" [ d ] or [ -> ] : Next Image")
    print(" [ p ]           : Print Parameters")
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
        nonlocal img_a_ui, img_t_raw, a_h, a_w, t_h, t_w, force_update
        a_path, t_path = valid_pairs[idx]
        img_a = cv2.imread(str(a_path))
        img_t_raw = load_thermal_image(t_path)
        
        # Resize allsky to a manageable size for fluid real-time UI
        UI_SIZE = 800
        img_a_ui = cv2.resize(img_a, (UI_SIZE, UI_SIZE), interpolation=cv2.INTER_AREA)
        
        a_h, a_w = img_a_ui.shape[:2]
        t_h, t_w = img_t_raw.shape[:2]
        
        print(f"Loaded [{idx+1}/{len(valid_pairs)}]: {a_path.name}")
        force_update = True

    # Load the initial image
    load_images(current_idx)

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

        # Convert trackbar values to actual math values
        x_off = (x_off_raw - 100) / 100.0  # -1.0 to 1.0
        y_off = (y_off_raw - 100) / 100.0  # -1.0 to 1.0
        dist = (dist_raw - 100) / 50.0     # -2.0 to 2.0
        alpha = alpha_pct / 100.0

        current_state = (proj_on, flip_h, flip_v, fov, rot, x_off, y_off, dist, alpha)

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
            
            # Warp thermal
            thermal_warped = cv2.remap(img_t, map_x, map_y, cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
            # Blend
            alpha_mask = np.zeros((a_h, a_w, 1), dtype=np.float32)
            alpha_mask[valid_mask] = alpha
            
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
            print(f"\n--- Current Parameters ---")
            print(f"--thermal-fov {fov} --rotation {rot} --offset-x {x_off:.2f} --offset-y {y_off:.2f} --distortion {dist:.2f} --alpha {alpha:.2f}{proj_str}{fh_str}{fv_str}")
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