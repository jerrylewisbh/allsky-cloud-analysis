"""
calibrate_thermal.py — Find the right thermal cloud-probability thresholds for
the (new) ZnSe-windowed MLX90640, by sweeping ABS_THRESHOLD_C / SIGMOID_SIGMA_C
against the raw exposures and scoring each candidate with the FIRMWARE's own
clear/cloud calls as ground truth.

WHY THIS EXISTS
---------------
The ZnSe window warms the apparent sky temperature by ~+19 C (clear sky reads
~-6 C now, not ~-25 C bare). The firmware was recalibrated for this (hard
cutoffs in external_components_local/sky_thermal/sky_thermal.h), but the offline
soft-sigmoid in make_masks_v2.py was not: with ABS=-3, SIGMA=3 the sigmoid's
50%% point sits *inside* the clear-sky temperature band, so a clear sky scores a
mean cloud probability of ~0.18-0.46 instead of ~0. That biases every soft
segmentation mask (future CNN labels) and saturates thermal_mean_p, which
poisons auto_classify.py.

This tool DOES NOT WRITE ANYTHING. It only reads raw exposures + existing meta
sidecars and prints a recommendation. Nothing in the live pipeline changes until
you re-run make_masks_v2.py with the env overrides it suggests.

GROUND TRUTH
------------
Each dataset_v2_*/meta/<frame>.json carries the firmware's per-frame cloud
fraction (fw_sky_abs_cloud_fraction / fw_sky_cloud_fraction). The firmware is
the recalibrated reference, so:
    clear  = fw fraction < --clear-max   (default 0.05)
    cloud  = fw fraction > --cloud-min   (default 0.95)
We sweep the offline sigmoid and ask: which (ABS, SIGMA) makes the offline
*mean cloud probability* read ~0 on clear frames and high on cloud frames,
with the cleanest separation?

NOTE ON FAITHFULNESS
--------------------
make_masks_v2 computes its mean over the spatially-warped probability map (RGB
resolution, FOV-eroded). This tool computes the thermal term over the raw
sensor's kept sky pixels (thermal_utils.KEEP corner mask). The *temperature
distribution of sky pixels* is what sets where the sigmoid should sit, so the
recommended thresholds transfer; always confirm with a real regen (the command
is printed at the end). Daytime RGB refinement is not modelled here — it only
*refines* the thermal term, so calibrating the thermal term is the right lever.

Run (on the server, in the venv):
    .venv/bin/python calibrate_thermal.py
    .venv/bin/python calibrate_thermal.py --night-only        # pure thermal path
    .venv/bin/python calibrate_thermal.py --abs -6:6:0.5 --sigma 1.0,1.5,2.0,3.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from thermal_utils import reshape_thermal, apply_corner_mask, ambient_from_sensors

PROJECT_ROOT = Path(__file__).parent.resolve()

# Current production defaults (mirror make_masks_v2.py) — shown as the baseline.
CUR_ABS = -3.0
CUR_REL = -12.0
CUR_SIGMA = 3.0

LUX_NIGHT = 1000.0  # mirror make_masks_v2: below this, the pipeline is thermal-only


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _parse_range(spec: str) -> list[float]:
    """'-6:6:0.5' -> inclusive arange; '1.0,1.5,2.0' -> explicit list."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + i * step, 6) for i in range(n)]
    return [float(x) for x in spec.split(",")]


def load_ground_truth(datasets_glob: str) -> dict[str, dict]:
    """{frame_id: {fw_frac, ambient_c, lux}} from dataset_v2_*/meta/*.json."""
    gt: dict[str, dict] = {}
    for ds in sorted(PROJECT_ROOT.glob(datasets_glob)):
        for p in (ds / "meta").glob("*.json"):
            try:
                m = json.load(open(p))
            except (OSError, ValueError):
                continue
            fw = m.get("fw_sky_abs_cloud_fraction")
            if fw is None:
                fw = m.get("fw_sky_cloud_fraction")
            if fw is None:
                continue
            gt[p.stem] = {
                "fw_frac": float(fw),
                "ambient_c": m.get("ambient_c"),
                "lux": m.get("lux"),
            }
    return gt


def index_exposures(thermal_root: Path) -> dict[str, Path]:
    """{frame_id (stem): exposure_json_path} by walking <thermal_root>/exposures."""
    root = thermal_root / "exposures"
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in root.rglob("*.json"):
        out[p.stem] = p
    return out


def kept_sky_temps(exposure_path: Path) -> np.ndarray | None:
    """Per-pixel sky temperatures over the kept (corner-masked) sensor pixels.

    Mirrors thermal_utils corner handling. Flips are irrelevant to the *set* of
    temperatures, so they're skipped. Returns 1-D float array (NaNs removed) or
    None if the frame is unreadable / wrong length.
    """
    try:
        data = json.load(open(exposure_path))
    except (OSError, ValueError):
        return None
    frame = data.get("frame")
    if not frame:
        return None
    try:
        arr, _ = reshape_thermal(frame)
    except ValueError:
        return None
    arr = apply_corner_mask(arr, fill=np.nan)  # NaN the clipped corners (full frames)
    t = arr[np.isfinite(arr)].astype(np.float64)
    return t if t.size else None


def frame_mean_prob(temps: np.ndarray, ambient: float,
                    abs_c: float, rel_c: float, sigma: float) -> tuple[float, float]:
    """(mean_prob, fraction>=0.5) for the firmware physics: p = max(p_abs, p_rel)."""
    p_abs = sigmoid((temps - abs_c) / sigma)
    p_rel = sigmoid((temps - (ambient + rel_c)) / sigma)
    p = np.maximum(p_abs, p_rel)
    return float(p.mean()), float((p >= 0.5).mean())


def pctl(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="dataset_v2_*",
                    help="glob of dataset dirs holding meta/ ground truth")
    ap.add_argument("--thermal-root",
                    default=os.environ.get("NAS_THERMAL_PATH", "/mnt/astro_image_thermal")
                            + "/" + os.environ.get("NAS_THERMAL_UUID", ""),
                    help="root containing exposures/ (raw thermal JSON)")
    ap.add_argument("--abs", dest="abs_spec", default="-6:4:1",
                    help="ABS sweep: 'lo:hi:step' or comma list (default -6:4:1)")
    ap.add_argument("--sigma", dest="sigma_spec", default="1.0,1.5,2.0,2.5,3.0",
                    help="SIGMA sweep: comma list or 'lo:hi:step'")
    ap.add_argument("--rel", type=float, default=CUR_REL,
                    help=f"REL_DELTA_C, held fixed during the sweep (default {CUR_REL})")
    ap.add_argument("--clear-max", type=float, default=0.05,
                    help="firmware frac below this = ground-truth CLEAR")
    ap.add_argument("--cloud-min", type=float, default=0.95,
                    help="firmware frac above this = ground-truth CLOUD")
    ap.add_argument("--night-only", action="store_true",
                    help="only frames with lux < LUX_NIGHT (pure thermal path)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap frames loaded (0 = all; useful for a quick look)")
    args = ap.parse_args()

    abs_vals = _parse_range(args.abs_spec)
    sigma_vals = _parse_range(args.sigma_spec)

    print(f"Ground truth from: {args.datasets}/meta/*.json")
    gt = load_ground_truth(args.datasets)
    print(f"  frames with firmware ground truth: {len(gt)}")

    thermal_root = Path(args.thermal_root)
    print(f"Raw exposures from: {thermal_root}/exposures")
    expo = index_exposures(thermal_root)
    print(f"  exposure files indexed: {len(expo)}")
    if not expo:
        print("\nERROR: no exposures found. Pass --thermal-root pointing at the dir "
              "that contains exposures/ (same root make_masks_v2.py uses).")
        return

    # ---- load temps once per frame; bucket into clear / cloud ----
    clear, cloud = [], []  # each: (temps, ambient)
    skipped = 0
    for fid, g in gt.items():
        ep = expo.get(fid)
        if ep is None:
            skipped += 1
            continue
        if args.night_only and (g["lux"] is None or g["lux"] >= LUX_NIGHT):
            continue
        temps = kept_sky_temps(ep)
        if temps is None:
            skipped += 1
            continue
        ambient = ambient_from_sensors({"ambient": g["ambient_c"]})
        rec = (temps, ambient)
        if g["fw_frac"] < args.clear_max:
            clear.append(rec)
        elif g["fw_frac"] > args.cloud_min:
            cloud.append(rec)
        if args.max_frames and (len(clear) + len(cloud)) >= args.max_frames:
            break

    print(f"  matched & loaded — CLEAR: {len(clear)}   CLOUD: {len(cloud)}   "
          f"(skipped {skipped} unmatched/unreadable)")
    if not clear or not cloud:
        print("\nERROR: need both clear and cloud frames to calibrate. "
              "Loosen --clear-max / --cloud-min or widen --datasets.")
        return

    def eval_params(a, s):
        cm = [frame_mean_prob(t, amb, a, args.rel, s)[0] for t, amb in clear]
        km = [frame_mean_prob(t, amb, a, args.rel, s)[0] for t, amb in cloud]
        clear_p90 = pctl(cm, 90)
        cloud_p10 = pctl(km, 10)
        return {
            "clear_med": pctl(cm, 50), "clear_p90": clear_p90,
            "cloud_med": pctl(km, 50), "cloud_p10": cloud_p10,
            "margin": cloud_p10 - clear_p90,  # >0 means the two never overlap (10/90)
        }

    # ---- baseline (current production) ----
    base = eval_params(CUR_ABS, CUR_SIGMA)
    print("\n" + "=" * 78)
    print(f"  BASELINE (current): ABS={CUR_ABS}  SIGMA={CUR_SIGMA}  REL={args.rel}")
    print("=" * 78)
    print(f"  clear mean_prob: med={base['clear_med']:.3f}  p90={base['clear_p90']:.3f}"
          f"   (want ~0)")
    print(f"  cloud mean_prob: med={base['cloud_med']:.3f}  p10={base['cloud_p10']:.3f}"
          f"   (want high)")
    print(f"  separation margin (cloud_p10 - clear_p90): {base['margin']:+.3f}")

    # ---- sweep ----
    print("\n" + "=" * 78)
    print(f"  SWEEP  (REL fixed at {args.rel})")
    print("=" * 78)
    print(f"  {'ABS':>5s} {'SIGMA':>6s} | {'clear_med':>9s} {'clear_p90':>9s} "
          f"| {'cloud_med':>9s} {'cloud_p10':>9s} | {'margin':>7s}")
    print("  " + "-" * 74)
    results = []
    for s in sigma_vals:
        for a in abs_vals:
            r = eval_params(a, s)
            results.append((a, s, r))
            flag = ""
            if r["clear_p90"] < 0.10 and r["margin"] > 0:
                flag = "  <-- clean"
            print(f"  {a:>5.1f} {s:>6.2f} | {r['clear_med']:>9.3f} {r['clear_p90']:>9.3f} "
                  f"| {r['cloud_med']:>9.3f} {r['cloud_p10']:>9.3f} | {r['margin']:>+7.3f}{flag}")

    # ---- recommend: largest margin among candidates whose clear_p90 < 0.05 ----
    ok = [(a, s, r) for a, s, r in results if r["clear_p90"] < 0.05]
    pool = ok if ok else results
    best_a, best_s, best_r = max(pool, key=lambda x: x[2]["margin"])
    print("\n" + "=" * 78)
    print("  RECOMMENDATION")
    print("=" * 78)
    if not ok:
        print("  (no candidate hit clear_p90 < 0.05 — widen --abs upward or shrink "
              "--sigma; the warm clear-sky tail may be edge/corner contamination,\n"
              "   in which case set corner_mask_rows/cols in alignment_config.json)")
    print(f"  ABS={best_a}  SIGMA={best_s}  REL={args.rel}")
    print(f"    clear: med={best_r['clear_med']:.3f} p90={best_r['clear_p90']:.3f}   "
          f"cloud: med={best_r['cloud_med']:.3f} p10={best_r['cloud_p10']:.3f}   "
          f"margin={best_r['margin']:+.3f}")
    print("\n  Validate with a real regen to a SCRATCH dir (does not touch prod):")
    print(f"    THERMAL_ABS_C={best_a} THERMAL_REL_C={args.rel} THERMAL_SIGMA_C={best_s} \\")
    print(f"        .venv/bin/python make_masks_v2.py --day <YYYYMMDD> "
          f"--output-root /tmp/recal_smoke --max-pairs 200")
    print("    # then eyeball /tmp/recal_smoke/meta/*.json: firmware-clear frames "
          "should now show mean_cloud_prob < 0.05")
    print("\n  Re-score the classifier against hand labels (no prod clobber):")
    print("    .venv/bin/python auto_classify_batch.py --datasets '/tmp/recal_smoke' "
          "--out /tmp/auto_recal.csv")
    print("    .venv/bin/python analyze_labels.py --auto /tmp/auto_recal.csv")


if __name__ == "__main__":
    main()
