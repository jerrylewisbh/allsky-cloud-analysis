import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse
from datetime import datetime

def load_thermal_image(thermal_bmp_path):
    # Check if a .json file exists alongside the .bmp
    json_path = thermal_bmp_path.with_suffix('.json')
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if 'frame' in data:
                raw_frame_1d = np.array(data['frame'], dtype=np.float32)
                if len(raw_frame_1d) == 768:
                    raw_frame = raw_frame_1d.reshape((24, 32))
                elif len(raw_frame_1d) == 384:
                    raw_frame = raw_frame_1d.reshape((16, 24))
                else:
                    raise ValueError(f"Unknown frame size: {len(raw_frame_1d)}")
                
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

def build_remap_matrices(allsky_w, allsky_h, thermal_w, thermal_h, 
                         allsky_fov_deg, thermal_fov_deg, rotation_deg, offset_x_pct, offset_y_pct, distortion, proj_on):
    
    # Allsky parameters
    cx_a = (allsky_w / 2.0) + (offset_x_pct * (allsky_w / 2.0))
    cy_a = (allsky_h / 2.0) + (offset_y_pct * (allsky_h / 2.0))
    R_a = allsky_w / 2.0  # Assuming the allsky circle touches the edges of the image
    max_theta_a = np.radians(allsky_fov_deg / 2.0)
    
    # Thermal parameters (Assuming thermal_fov_deg is the horizontal FOV)
    thermal_fov_rad = np.radians(thermal_fov_deg)
    f_t = (thermal_w / 2.0) / np.tan(thermal_fov_rad / 2.0)
    cx_t, cy_t = thermal_w / 2.0, thermal_h / 2.0
    
    # Create meshgrid for the allsky image
    print("Pre-computing projection matrices...")
    X, Y = np.meshgrid(np.arange(allsky_w), np.arange(allsky_h))
    
    dx = X - cx_a
    dy = Y - cy_a
    
    # Polar coordinates in allsky plane
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    
    # Apply user-defined rotation to match physical camera alignment
    phi = phi + np.radians(rotation_deg)
    
    if not proj_on:
        # Flat/linear mapping (bypass fisheye/spherical projection)
        # Use thermal_fov_deg as a generic scale factor
        scale = (thermal_fov_deg / allsky_fov_deg) * allsky_w
        s_factor = thermal_w / scale
        
        dx_rot = r * np.cos(phi)
        dy_rot = r * np.sin(phi)
        
        map_x = cx_t + dx_rot * s_factor
        map_y = cy_t + dy_rot * s_factor
        
        invalid = (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
        map_x[invalid] = -1
        map_y[invalid] = -1
        return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid

    # Calculate zenith angle theta for each allsky pixel
    r_norm = r / R_a
    gamma = 2.0 ** distortion
    theta = (r_norm ** gamma) * max_theta_a
    
    # We only care about angles within the forward hemisphere 
    valid_theta = theta < (np.pi / 2 - 0.01)
    
    # Calculate distance from center in the thermal image plane (pinhole projection)
    # tan(theta) = d_t / f_t
    d_t = np.zeros_like(theta)
    d_t[valid_theta] = f_t * np.tan(theta[valid_theta])
    
    # Convert back to Cartesian coordinates in the thermal image
    map_x = cx_t + d_t * np.cos(phi)
    map_y = cy_t + d_t * np.sin(phi)
    
    # Mask out pixels that fall outside the thermal sensor bounds
    invalid = ~valid_theta | (map_x < 0) | (map_x >= thermal_w) | (map_y < 0) | (map_y >= thermal_h)
    map_x[invalid] = -1
    map_y[invalid] = -1
    
    return map_x.astype(np.float32), map_y.astype(np.float32), ~invalid

def generate_overlay(date_str, allsky_root, thermal_root, output_file, ccd_uuid, 
                     allsky_fov, thermal_fov, rotation, offset_x, offset_y, distortion, proj_on, flip_h, flip_v, alpha, out_size, start_frame, max_frames):
                     
    allsky_dir = Path(allsky_root) / "images" / date_str
    thermal_dir = Path(thermal_root) / ccd_uuid / "exposures" / date_str

    if not allsky_dir.exists():
        print(f"Allsky directory not found: {allsky_dir}")
        return
    if not thermal_dir.exists():
        print(f"Thermal directory not found: {thermal_dir}")
        return

    print(f"Scanning Allsky in {allsky_dir}")
    allsky_files = sorted(list(allsky_dir.glob("**/*.jpg")))
    
    print(f"Scanning Thermal in {thermal_dir}")
    thermal_files = list(thermal_dir.glob("**/*.bmp"))
    thermal_index = {f.stem: f for f in thermal_files}
    
    # Filter only overlapping frames
    valid_pairs = []
    for allsky_p in allsky_files:
        if "thumbnails" in str(allsky_p):
            continue
        base_name = allsky_p.stem
        if base_name in thermal_index:
            valid_pairs.append((allsky_p, thermal_index[base_name]))
            
    total_valid = len(valid_pairs)
    
    if start_frame and start_frame > 0:
        valid_pairs = valid_pairs[start_frame:]
        
    if max_frames and max_frames > 0:
        valid_pairs = valid_pairs[:max_frames]
        print(f"Processing slice: frames {start_frame} to {start_frame + len(valid_pairs)} (of {total_valid} total).")
    else:
        print(f"Found {len(valid_pairs)} overlapping frames.")
    if not valid_pairs:
        return

    # Read first pair to establish dimensions and projection matrices
    img_a_test = cv2.imread(str(valid_pairs[0][0]))
    img_t_test = cv2.imread(str(valid_pairs[0][1]))
    
    a_h, a_w = img_a_test.shape[:2]
    t_h, t_w = img_t_test.shape[:2]
    
    map_x, map_y, valid_mask = build_remap_matrices(
        a_w, a_h, t_w, t_h, allsky_fov, thermal_fov, rotation, offset_x, offset_y, distortion, proj_on
    )
    
    # Prepare alpha blending mask for fast vectorized operations
    alpha_mask = np.zeros((a_h, a_w, 1), dtype=np.float32)
    alpha_mask[valid_mask] = alpha

    video_writer = None
    
    count = 0
    for allsky_p, thermal_p in valid_pairs:
        img_allsky = cv2.imread(str(allsky_p))
        img_thermal = load_thermal_image(thermal_p)
        
        if img_allsky is None or img_thermal is None:
            continue
            
        # Apply flips
        if flip_h and flip_v:
            img_thermal = cv2.flip(img_thermal, -1)
        elif flip_h:
            img_thermal = cv2.flip(img_thermal, 1)
        elif flip_v:
            img_thermal = cv2.flip(img_thermal, 0)

        # Warp thermal image to allsky fisheye projection
        # INTER_CUBIC gives a nice smooth upscale for the tiny 16x24 thermal data
        thermal_warped = cv2.remap(img_thermal, map_x, map_y, cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # Blend the images where the thermal camera overlays
        blended = img_allsky.astype(np.float32) * (1.0 - alpha_mask) + \
                  thermal_warped.astype(np.float32) * alpha_mask
        
        blended = blended.astype(np.uint8)
        
        # Resize output video to save space / playback speed
        out_frame = cv2.resize(blended, (out_size, out_size), interpolation=cv2.INTER_AREA)
        
        # Add timestamp label
        cv2.putText(out_frame, allsky_p.stem, (20, out_size - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(out_frame, "Allsky + Thermal Overlay", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file, fourcc, 10.0, (out_size, out_size))
        
        video_writer.write(out_frame)
        count += 1
        
        if count % 10 == 0:
            print(f"Processed {count}/{len(valid_pairs)} frames...", end='\r')

    if video_writer:
        video_writer.release()
        print(f"\nOverlay Timelapse saved to {output_file} ({count} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate fisheye overlay timelapse')
    parser.add_argument('--date', type=str, required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--allsky', type=str, default='/Volumes/allsky_images', help='Allsky root path')
    parser.add_argument('--thermal', type=str, default='/Volumes/astro_image_thermal', help='Thermal root path')
    parser.add_argument('--uuid', type=str, default='ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212', help='Thermal CCD UUID')
    parser.add_argument('--output', type=str, default='overlay_timelapse.mp4', help='Output filename')
    
    # Projection tuning parameters
    parser.add_argument('--allsky-fov', type=float, default=180.0, help='Allsky Field of View in degrees')
    parser.add_argument('--thermal-fov', type=float, default=55.0, help='Thermal Horizontal Field of View in degrees')
    parser.add_argument('--rotation', type=float, default=0.0, help='Rotation angle of thermal camera in degrees')
    parser.add_argument('--offset-x', type=float, default=0.0, help='X offset percentage (-1.0 to 1.0)')
    parser.add_argument('--offset-y', type=float, default=0.0, help='Y offset percentage (-1.0 to 1.0)')
    parser.add_argument('--distortion', type=float, default=0.0, help='Radial distortion power curve (-2.0 to 2.0)')
    parser.add_argument('--no-projection', action='store_true', help='Disable spherical projection math (flat 2D scale)')
    parser.add_argument('--flip-h', action='store_true', help='Flip thermal image horizontally')
    parser.add_argument('--flip-v', action='store_true', help='Flip thermal image vertically')
    parser.add_argument('--alpha', type=float, default=0.65, help='Opacity of thermal overlay (0.0 to 1.0)')
    parser.add_argument('--out-size', type=int, default=1200, help='Output video width/height')
    parser.add_argument('--start-frame', type=int, default=0, help='Start processing from frame index N')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit processing to N frames for quick testing')

    args = parser.parse_args()
    
    proj_on = not args.no_projection
    
    generate_overlay(args.date, args.allsky, args.thermal, args.output, args.uuid,
                     args.allsky_fov, args.thermal_fov, args.rotation, args.offset_x, args.offset_y, args.distortion, proj_on, args.flip_h, args.flip_v, args.alpha, args.out_size, args.start_frame, args.max_frames)
