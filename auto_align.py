import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from thermal_utils import reshape_thermal, fill_corners_clear, KEEP

def load_thermal(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        frame2d, _ = reshape_thermal(data['frame'])
    except ValueError:
        return None
    # Clipped corners (warm enclosure) would bias the MI alignment -> fill clear.
    return fill_corners_clear(frame2d)

def get_mutual_information(img1, img2, bins=20):
    hgram, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def build_warped_thermal(thermal_raw, a_w, a_h, fov, rot, x_off, y_off, dist):
    t_h, t_w = thermal_raw.shape
    cx_a = (a_w / 2.0) + (x_off * (a_w / 2.0))
    cy_a = (a_h / 2.0) + (y_off * (a_h / 2.0))
    R_a = a_w / 2.0
    max_theta_a = np.radians(180.0 / 2.0)
    
    f_t = (t_w / 2.0) / np.tan(np.radians(fov) / 2.0)
    cx_t, cy_t = t_w / 2.0, t_h / 2.0
    
    X, Y = np.meshgrid(np.arange(a_w), np.arange(a_h))
    dx, dy = X - cx_a, Y - cy_a
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) + np.radians(rot)
    
    r_norm = r / R_a
    gamma = 2.0 ** dist
    theta = (r_norm ** gamma) * max_theta_a
    
    valid_theta = theta < (np.pi / 2 - 0.01)
    d_t = np.zeros_like(theta)
    d_t[valid_theta] = f_t * np.tan(theta[valid_theta])
    
    map_x = (cx_t + d_t * np.cos(phi)).astype(np.float32)
    map_y = (cy_t + d_t * np.sin(phi)).astype(np.float32)
    
    invalid = ~valid_theta | (map_x < 0) | (map_x >= t_w) | (map_y < 0) | (map_y >= t_h)
    map_x[invalid], map_y[invalid] = -1, -1
    
    warped = cv2.remap(thermal_raw, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    return warped, ~invalid

def evaluate_batch(loaded_pairs, params, dist):
    """Calculates the average Mutual Information across all loaded image pairs."""
    total_mi = 0
    valid_count = 0
    for img_a, thermal_raw in loaded_pairs:
        warped, mask = build_warped_thermal(thermal_raw, 400, 400, *params, dist)
        if np.sum(mask) > 0:
            total_mi += get_mutual_information(img_a[mask], warped[mask])
            valid_count += 1
    return total_mi / valid_count if valid_count > 0 else -1

def main():
    parser = argparse.ArgumentParser(description='Auto-align Thermal and Allsky images (Single or Batch)')
    parser.add_argument('input_path', type=str, help='Path to an Allsky/Thermal file OR directory')
    parser.add_argument('--allsky-root', type=str, default='/Volumes/allsky_images', help='Allsky root path')
    parser.add_argument('--thermal-root', type=str, default='/Volumes/astro_image_thermal', help='Thermal root path')
    parser.add_argument('--uuid', type=str, default='ccd_25ccc900-4f15-4ac2-9d29-507e89f7c212', help='Thermal CCD UUID')
    parser.add_argument('--config', default='alignment_config.json', help='Path to config file')
    parser.add_argument('--max-samples', type=int, default=10, help='Max images to sample from a directory (for speed)')
    parser.add_argument('--min-struct-std', type=float, default=2.0,
                        help='Skip frames whose kept-region thermal std is below this. MI needs '
                             'cloud structure to lock; clear/uniform-overcast frames just add noise.')
    parser.add_argument('--search-bound', type=float, nargs=4, default=[3.0, 3.0, 0.03, 0.03],
                        metavar=('FOV', 'ROT', 'XOFF', 'YOFF'),
                        help='Max +/- deviation from the seed (manual config) for each param. Refine-only.')
    parser.add_argument('--improve-margin', type=float, default=0.01,
                        help='Required relative MI gain over the seed to accept (default 1%%).')
    parser.add_argument('--overwrite', action='store_true',
                        help='If improved by the margin, overwrite the config in place. Default is '
                             'non-destructive: write alignment_config.auto.json and keep the manual config.')
    args = parser.parse_args()

    input_p = Path(args.input_path).resolve()
    
    is_allsky = 'images' in input_p.parts
    is_thermal = 'exposures' in input_p.parts
    
    if not (is_allsky or is_thermal):
        print("Input path must contain either 'images' (Allsky) or 'exposures' (Thermal) in its directory structure.")
        return

    try:
        idx = input_p.parts.index('images' if is_allsky else 'exposures')
        if input_p.is_dir():
            rel_parts = input_p.parts[idx+1:]
        else:
            rel_parts = input_p.parts[idx+1:-1]
    except ValueError:
        print("Path parsing failed.")
        return

    pairs_to_process = []
    
    # 1. Gather all valid pairs
    if input_p.is_file():
        if is_allsky:
            t_path = Path(args.thermal_root) / args.uuid / "exposures" / Path(*rel_parts) / f"{input_p.stem}.json"
            pairs_to_process.append((input_p, t_path))
        else:
            a_path = Path(args.allsky_root) / "images" / Path(*rel_parts) / f"{input_p.stem}.jpg"
            pairs_to_process.append((a_path, input_p.with_suffix('.json')))
    elif input_p.is_dir():
        print(f"Scanning directory for valid pairs...")
        if is_allsky:
            allsky_dir = input_p
            thermal_dir = Path(args.thermal_root) / args.uuid / "exposures" / Path(*rel_parts)
            print(f"  Allsky path: {allsky_dir}")
            print(f"  Thermal path: {thermal_dir}")
            for a_path in sorted(allsky_dir.glob("*.[jJ][pP][gG]")) + sorted(allsky_dir.glob("*.jpg")):

                if "thumbnails" in str(a_path): continue
                t_path = thermal_dir / f"{a_path.stem}.json"
                if t_path.exists(): pairs_to_process.append((a_path, t_path))
        else:
            thermal_dir = input_p
            allsky_dir = Path(args.allsky_root) / "images" / Path(*rel_parts)
            for t_path in sorted(thermal_dir.glob("*.[jJ][sS][oO][nN]")):
                a_path = allsky_dir / f"{t_path.stem}.jpg"
                if a_path.exists(): pairs_to_process.append((a_path, t_path))

    valid_pairs = [p for p in pairs_to_process if p[0].exists() and p[1].exists()]
    
    if not valid_pairs:
        print("Error: No matching valid pairs found. Make sure .json thermal data exists.")
        return

    # 2. Downsample if there are too many (even spacing)
    if len(valid_pairs) > args.max_samples:
        indices = np.linspace(0, len(valid_pairs) - 1, args.max_samples, dtype=int)
        valid_pairs = [valid_pairs[i] for i in indices]
        
    print(f"Selected {len(valid_pairs)} pairs for joint optimization.")

    # 3. Load config and apply flips
    config_path = Path(args.config)
    p = {
        "proj_on": 1, "flip_h": 0, "flip_v": 1,
        "fov": 74, "rot": 202, "x_off": -0.04,
        "y_off": 0.12, "dist": 0.0, "alpha": 1.0
    }
    if config_path.exists():
        with open(config_path, 'r') as f:
            p.update(json.load(f))
            
    # 4. Pre-load all images into memory
    print("Loading images into memory...")
    loaded_pairs = []
    skipped_uniform = 0
    for a_path, t_path in valid_pairs:
        img_a = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
        if img_a is None: continue
        img_a = cv2.resize(img_a, (400, 400)) # Downsample for speed

        thermal_raw = load_thermal(str(t_path))
        if thermal_raw is None: continue

        # MI alignment needs cloud structure; skip near-uniform (clear / flat
        # overcast) frames where the MI surface is flat and the fit drifts.
        sstd = float(np.std(thermal_raw[KEEP])) if thermal_raw.shape == KEEP.shape else float(np.std(thermal_raw))
        if sstd < args.min_struct_std:
            skipped_uniform += 1
            continue

        if p.get('flip_h', 0) and p.get('flip_v', 1):
            thermal_raw = np.flip(thermal_raw, (0, 1))
        elif p.get('flip_h', 0):
            thermal_raw = np.flip(thermal_raw, 1)
        elif p.get('flip_v', 1):
            thermal_raw = np.flip(thermal_raw, 0)
            
        loaded_pairs.append((img_a, thermal_raw))

    if skipped_uniform:
        print(f"  Skipped {skipped_uniform} near-uniform frame(s) (std < {args.min_struct_std}C) — no MI structure.")
    if not loaded_pairs:
        print(f"No frames with enough cloud structure (std >= {args.min_struct_std}C). "
              f"Alignment needs broken cloud; nothing changed. Try a cloudier window or lower --min-struct-std.")
        return
    print(f"Using {len(loaded_pairs)} structured frame(s) for alignment.")

    # 5. Optimization Loop (refine-only: bounded around the seed/manual config)
    current = [p['fov'], p['rot'], p['x_off'], p['y_off']]
    seed = list(current)
    bound = args.search_bound
    steps = [1.0, 1.0, 0.01, 0.01]

    best_mi = evaluate_batch(loaded_pairs, current, p['dist'])
    initial_mi = best_mi
    
    print(f"\nStarting batch auto-alignment search...")
    print(f"Initial: FOV={current[0]}, Rot={current[1]}, X={current[2]}, Y={current[3]} (Base MI: {best_mi:.4f})")

    for iteration in range(50):
        improved = False
        for i in range(len(current)):
            for direction in [-1, 1]:
                test_params = list(current)
                test_params[i] += steps[i] * direction
                # refine-only: clamp within +/- bound of the seed (manual config)
                lo, hi = seed[i] - bound[i], seed[i] + bound[i]
                test_params[i] = max(lo, min(hi, test_params[i]))
                if test_params[i] == current[i]:
                    continue  # at the bound, no move

                mi = evaluate_batch(loaded_pairs, test_params, p['dist'])
                
                if mi > best_mi:
                    best_mi = mi
                    current = test_params
                    improved = True
                    print(f"Iter {iteration:02d}: MI={mi:.4f} -> FOV={current[0]:.1f}, Rot={current[1]:.1f}, X={current[2]:.2f}, Y={current[3]:.2f}")
        
        if not improved:
            steps = [s * 0.5 for s in steps]
            if steps[0] < 0.1: break # Converged

    # 6. Save results — guarded so a low-signal run can't silently degrade the
    #    trusted manual config.
    p.update({
        "fov": round(current[0], 2),
        "rot": round(current[1], 2),
        "x_off": round(current[2], 3),
        "y_off": round(current[3], 3)
    })
    improvement = best_mi - initial_mi
    rel = improvement / abs(initial_mi) if initial_mi not in (0, -1) else 0.0
    print(f"\nMI: seed {initial_mi:.4f} -> final {best_mi:.4f}  (delta {improvement:+.4f}, {rel*100:+.1f}%)")
    print(f"Params: FOV={current[0]:.2f} Rot={current[1]:.2f} X={current[2]:.3f} Y={current[3]:.3f}"
          f"  (seed FOV={seed[0]:.2f} Rot={seed[1]:.2f} X={seed[2]:.3f} Y={seed[3]:.3f})")

    accepted = rel >= args.improve_margin
    if args.overwrite and accepted:
        target = config_path
        print(f"[ACCEPTED] gain >= {args.improve_margin*100:.0f}% -> overwriting {target}")
    else:
        target = config_path.with_suffix('.auto.json')
        if args.overwrite and not accepted:
            print(f"[KEPT MANUAL] gain < {args.improve_margin*100:.0f}% -> did NOT touch {config_path}. "
                  f"Candidate written to {target} for inspection.")
        else:
            print(f"[NON-DESTRUCTIVE] candidate written to {target} (re-run with --overwrite to promote when gain is real).")

    with open(target, 'w') as f:
        json.dump(p, f, indent=4)

if __name__ == "__main__":
    main()
