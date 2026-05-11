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

def generate_side_by_side(date_str, allsky_root, thermal_root, output_file, ccd_uuid, vertical=False, flip_h=False, flip_v=False):
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
    print(f"Found {len(allsky_files)} allsky images")

    # Index thermal files by basename
    print(f"Scanning Thermal in {thermal_dir}")
    thermal_files = list(thermal_dir.glob("**/*.bmp"))
    thermal_index = {f.stem: f for f in thermal_files}
    print(f"Found {len(thermal_files)} thermal images")

    # Common dimension for the timelapse
    TARGET_SIZE = 1000
    
    video_writer = None
    
    count = 0
    for allsky_p in allsky_files:
        if "thumbnails" in str(allsky_p):
            continue
            
        base_name = allsky_p.stem
        if base_name in thermal_index:
            thermal_p = thermal_index[base_name]
            
            # Load images
            img_allsky = cv2.imread(str(allsky_p))
            img_thermal = load_thermal_image(thermal_p)
            
            if img_allsky is None or img_thermal is None:
                continue
                
            # Apply flips to thermal
            if flip_h and flip_v:
                img_thermal = cv2.flip(img_thermal, -1)
            elif flip_h:
                img_thermal = cv2.flip(img_thermal, 1)
            elif flip_v:
                img_thermal = cv2.flip(img_thermal, 0)
                
            if vertical:
                # Top/Bottom layout: Match widths
                h, w = img_allsky.shape[:2]
                scale_allsky = TARGET_SIZE / w
                allsky_resized = cv2.resize(img_allsky, (TARGET_SIZE, int(h * scale_allsky)))
                
                h_t, w_t = img_thermal.shape[:2]
                scale_thermal = TARGET_SIZE / w_t
                # Using INTER_CUBIC for a smoother look as requested
                thermal_resized = cv2.resize(img_thermal, (TARGET_SIZE, int(h_t * scale_thermal)), interpolation=cv2.INTER_CUBIC)
                
                # Combine vertically
                combined = np.vstack((allsky_resized, thermal_resized))
                thermal_offset_y = allsky_resized.shape[0]
                thermal_offset_x = 0
            else:
                # Left/Right layout: Match heights
                h, w = img_allsky.shape[:2]
                scale_allsky = TARGET_SIZE / h
                allsky_resized = cv2.resize(img_allsky, (int(w * scale_allsky), TARGET_SIZE))
                
                h_t, w_t = img_thermal.shape[:2]
                scale_thermal = TARGET_SIZE / h_t
                thermal_resized = cv2.resize(img_thermal, (int(w_t * scale_thermal), TARGET_SIZE), interpolation=cv2.INTER_CUBIC)
                
                # Combine horizontally
                combined = np.hstack((allsky_resized, thermal_resized))
                thermal_offset_y = 0
                thermal_offset_x = allsky_resized.shape[1]
            
            # Add labels
            cv2.putText(combined, "Allsky", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Thermal", (thermal_offset_x + 10, thermal_offset_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, base_name, (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if video_writer is None:
                height, width = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_file, fourcc, 10.0, (width, height))
            
            video_writer.write(combined)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} frames...", end='\r')

    if video_writer:
        video_writer.release()
        print(f"\nTimelapse saved to {output_file} ({count} frames)")
    else:
        print("\nNo matching frames found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate side-by-side timelapse')
    parser.add_argument('--date', type=str, required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--allsky', type=str, default='/Volumes/allsky_images', help='Allsky root path')
    parser.add_argument('--thermal', type=str, default='/Volumes/astro_image_thermal', help='Thermal root path')
    parser.add_argument('--uuid', type=str, default='ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212', help='Thermal CCD UUID')
    parser.add_argument('--output', type=str, default='comparison_timelapse.mp4', help='Output filename')
    parser.add_argument('--vertical', action='store_true', help='Use vertical (stacked) layout')
    parser.add_argument('--flip-h', action='store_true', help='Flip thermal image horizontally')
    parser.add_argument('--flip-v', action='store_true', help='Flip thermal image vertically')

    args = parser.parse_args()
    generate_side_by_side(args.date, args.allsky, args.thermal, args.output, args.uuid, args.vertical, args.flip_h, args.flip_v)
