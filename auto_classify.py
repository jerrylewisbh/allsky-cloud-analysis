"""
auto_classify.py — Rule-based pre-classifier using ground + satellite weak labels.

Given the multimodal weak labels for a frame plus optional local thermal stats,
returns a (label, confidence, reasoning) verdict among the 9 classes defined
in docs/labeling-protocol.md.

Used by the labeling UI to pre-select the radio button so hand-labelers verify
or override instead of annotating from scratch. The goal: drop manual labeling
effort by ~10× while preserving genus-level accuracy on the cases that
fundamentally require RGB texture analysis.

Decision logic (see docs/weak-labels-reference.md §"How to combine signals"):

  0. Thermal spatial variance (broken-cloud texture) → cu / sc / ac_as (medium)
  1. Precipitation + (optional 4-AND CB signature) → ns_cb (medium or high)
  2. GOES high-ice cloud + locally clear + no rain → ci (medium)  [targeted Ci]
  3. Vote across thermal_mean_p, GOES mask, CSI, mpsas, METAR okta, sky_cond,
     rgb_nrbr (day), rgb_v_night (night):
       - all signals agree clear → clear (high, capped at medium in twilight)
       - all signals agree cloud → continue to family (medium)
       - local-overhead signals dominate → trust them unless METAR>=6 → multi
  4. METAR CB genus + confident cloud + GOES ice top → ns_cb (medium)
  5. Family from GOES height (preferred) or METAR altitude_bucket
  6. Genus within family:
       - high + night + low thermal → ci (medium); else cs_cc (low, needs RGB)
       - mid  → ac_as (medium)
       - low  → st if humidity > 90 (medium)
              → cu if METAR genus CU/TCU OR CSI variance high (medium)
              → sc (low) otherwise [needs RGB texture]
       - missing family → multi (low)

Confidence semantics:
  high   — verification should be one click; signals leave no doubt
  medium — directionally correct (family + presence) but genus may need adjustment
  low    — signals disagree OR fundamental ambiguity (Cu↔Sc, Ci↔Cs, multi-cloud)
"""
from __future__ import annotations

from typing import Optional

CLASSES = ["clear", "ci", "cs_cc", "ac_as", "cu", "sc", "st", "ns_cb", "multi"]


def _get(weak: dict, source: str, attr: str,
         default=None, as_float=False, as_int=False):
    row = weak.get((source, attr))
    if not row:
        return default
    v = row["value"]
    try:
        if as_float:
            return float(v)
        if as_int:
            return int(float(v))
        return v
    except (ValueError, TypeError):
        return default


def classify(weak: dict[tuple, dict],
             thermal_mean_p: Optional[float] = None,
             rgb_nrbr_p95: Optional[float] = None,
             rgb_v_mean: Optional[float] = None,
             rgb_v_std: Optional[float] = None,
             thermal_std: Optional[float] = None) -> tuple[str, str, str]:
    """Returns (class, confidence, reasoning).

    Parameters
    ----------
    weak : {(source, attribute): row_dict}
        Loaded from labels/weak_labels.csv, same shape as labeling_tool uses.
    thermal_mean_p : float, optional
        Mean cloud probability over the valid pixels of the local thermal mask.
    rgb_nrbr_p95 : float, optional
        95th percentile of Normalized Red-Blue Ratio (R-B)/(R+B) over the 
        thermal-valid region of the RGB crop. Captures the 'whitest' (cloudiest)
        peak signal, preventing small puffs from being washed out by blue sky.
    rgb_v_mean : float, optional
        Mean V (HSV brightness, 0–255) over the thermal-valid region of the
        long-exposure RGB crop. 
    rgb_v_std : float, optional
        Standard deviation of V channel. High variance at night indicates 
        texture (clouds reflecting skyglow) vs uniform clear sky.
    thermal_std : float, optional
        Spatial standard deviation of cloud probability across valid mask
        pixels. Discriminates clouds with internal structure (Cu fragments
        with blue gaps → high std; broken Ac/Sc → mid std) from uniform
        sky (clear → low std; overcast Sc/St → low std). High std + mid
        mean is a textbook convective Cu signature regardless of what the
        vote cascade says about overall cloud presence.
    """
    # ---- pull signals ----
    sun_alt = _get(weak, "ephemeris", "sun_alt_deg", as_float=True)
    csi = _get(weak, "derived", "daytime_clear_sky_index", as_float=True)
    mpsas = _get(weak, "esp32_sensor", "sky_brightness_mpsas", as_float=True)
    lux = _get(weak, "esp32_sensor", "illuminance_lux", as_float=True)
    sky_cond = _get(weak, "esp32_sensor", "sky_condition")  # firmware's own verdict
    humidity = _get(weak, "weather_station", "humidity_pct", as_float=True)
    rain_mm = _get(weak, "weather_station", "rain_1h_mm", as_float=True)
    goes_mask = _get(weak, "goes19_acmc", "cloud_present", as_int=True)
    goes_phase = _get(weak, "goes19_actpc", "cloud_top_phase")
    goes_height = _get(weak, "goes19_achac", "cloud_top_height_m", as_float=True)
    metar_okta = _get(weak, "metar", "coverage_okta", as_int=True)
    metar_genus = _get(weak, "metar", "cloud_genus_hint")
    metar_bucket = _get(weak, "metar", "altitude_bucket")

    is_day = sun_alt is not None and sun_alt > 6.0
    is_night = sun_alt is not None and sun_alt < -6.0
    csi_std = _get(weak, "derived", "csi_std_10min", as_float=True)

    # ---- Rule 0: thermal spatial-variance texture (overrides clear cascade) ----
    # Clear sky and overcast sheets both have low spatial std in the thermal
    # mask (uniform cold / uniform warm). High std means warm puffs against
    # cold gaps, i.e. discrete cloud elements (Cu fragments, broken Sc/Ac).
    # Combined with a mid-range mean, this is a confident "broken-cloud"
    # signature that the per-frame vote cascade can't see — the cascade
    # operates on the SCALAR mean. METAR okta then disambiguates Cu (scattered)
    # from broken Sc (more coverage).
    #
    # Guard: only fire when GOES height is unknown OR puts the cloud in
    # low/mid family (<6km). At high family the "texture" is more often
    # patchy cirrus (Ci) than convective Cu — observed v1: 7 hand-ci
    # frames got mis-predicted as cu by this rule when GOES top was >6km.
    high_family_from_goes = (goes_height is not None and goes_height >= 6000)
    if (not high_family_from_goes
            and thermal_std is not None and thermal_std > 0.20
            and thermal_mean_p is not None and 0.05 < thermal_mean_p < 0.65):
        texture_note = f"thermal std={thermal_std:.2f} mean={thermal_mean_p:.2f}"
        if is_day:
            if metar_okta is not None and metar_okta >= 5:
                return "sc", "medium", \
                       f"daytime broken-cloud texture ({texture_note}) + METAR {metar_okta}/8 → broken Sc"
            return "cu", "medium", \
                   f"daytime broken-cloud texture ({texture_note}) → Cu over blue gaps"
        if is_night:
            return "ac_as", "medium", \
                   f"night broken-cloud texture ({texture_note}) → Ac/broken Sc"
        # twilight: ambiguous, fall through to vote cascade

    # ---- Rule 1: active precipitation suggests ns_cb ----
    if rain_mm is not None and rain_mm > 0.5:
        cb_signature = (goes_phase == "ice"
                        and goes_height is not None and goes_height > 6000
                        and humidity is not None and humidity > 90)
        if cb_signature:
            return "ns_cb", "high", \
                   (f"rain {rain_mm:.1f}mm + GOES ice top {goes_height:.0f}m + "
                    f"humidity {humidity:.0f}% → deep convective ns_cb")
        return "ns_cb", "medium", \
               f"rain_1h_mm={rain_mm:.1f} > 0.5 (visual genus may differ — verify)"

    # ---- Rule 2: targeted high-ice Ci (locals can't see thin cirrus) ----
    if (goes_phase == "ice"
            and goes_height is not None and goes_height > 6000
            and goes_mask == 1
            and metar_okta is not None and metar_okta <= 3
            and thermal_mean_p is not None and thermal_mean_p < 0.15):
        return "ci", "medium", \
               (f"GOES high-ice cloud (top {goes_height:.0f}m, mask=1) + "
                f"METAR {metar_okta}/8 + thermal_p={thermal_mean_p:.2f} → thin Ci")

    # ---- Rule 3: cloud-presence vote with three-tier semantics ----
    votes: list[tuple[str, bool | None, str, bool]] = []

    if thermal_mean_p is not None:
        if thermal_mean_p > 0.4:
            v = True
        elif thermal_mean_p > 0.2:
            v = None
        elif thermal_mean_p < 0.05:
            v = False
        elif metar_okta is not None and metar_okta >= 5:
            v = None
        else:
            v = False
        votes.append(("thermal", v, f"thermal_p={thermal_mean_p:.2f}", True))

    if goes_mask is not None:
        if goes_mask == 1:
            v = True
        elif goes_phase and goes_phase not in ("clear", None):
            v = None
        else:
            v = False
        votes.append(("goes", v, f"goes_mask={goes_mask} phase={goes_phase}", False))

    if is_day and csi is not None:
        if csi > 1.1 or csi < 0.7:
            v = True
        elif csi > 1.05 or csi < 0.85:
            v = None
        else:
            v = False
        votes.append(("csi", v, f"csi={csi:.2f}", True))

    is_deep_night = sun_alt is not None and sun_alt < -18.0
    if is_deep_night:
        night_cloud = False
        night_clear = False
        night_weak = False

        if lux is not None:
            if lux > 0.08: night_cloud = True
            elif lux < 0.005: night_clear = True
            else: night_weak = True
            votes.append(("lux", night_cloud if not night_weak else None, f"lux={lux:.3f}", True))

        elif mpsas is not None:
            if mpsas < 16.0: night_cloud = True
            elif mpsas >= 18.5: night_clear = True
            else: night_weak = True
            votes.append(("mpsas", night_cloud if not night_weak else None, f"mpsas={mpsas:.2f}", True))

    if metar_okta is not None:
        if metar_okta >= 5:
            v = True
        elif metar_okta >= 3:
            v = None
        else:
            v = False
        votes.append(("metar", v, f"metar_okta={metar_okta}", False))

    if sky_cond:
        if sky_cond in ("mostly_cloudy", "overcast"):
            v = True
        elif sky_cond in ("mostly_clear", "partly_cloudy"):
            v = None
        elif sky_cond in ("very_clear", "clear"):
            v = False
        else:
            v = None
        votes.append(("sky_cond", v, f"sky_condition={sky_cond}", True))

    if is_day and rgb_nrbr_p95 is not None:
        if rgb_nrbr_p95 > -0.15:
            v = True
        elif rgb_nrbr_p95 < -0.50:
            v = False
        else:
            v = None
        votes.append(("rgb_nrbr_peak", v, f"nrbr_p95={rgb_nrbr_p95:+.2f}", True))

    if is_night and rgb_v_mean is not None and rgb_v_std is not None:
        if rgb_v_std > 15.0 or rgb_v_mean > 100:
            v = True
        elif rgb_v_mean < 50 and rgb_v_std < 5.0:
            v = False
        else:
            v = None
        votes.append(("rgb_v_night", v, f"v_mean={rgb_v_mean:.0f} v_std={rgb_v_std:.1f}", True))

    if not votes:
        return "clear", "low", "no signals available"

    strong_cloud = [v for v in votes if v[1] is True]
    strong_clear = [v for v in votes if v[1] is False]
    weak_cloud   = [v for v in votes if v[1] is None]
    n_sc, n_scl, n_w = len(strong_cloud), len(strong_clear), len(weak_cloud)

    local_sc  = sum(1 for v in strong_cloud if v[3])
    local_scl = sum(1 for v in strong_clear if v[3])
    local_w   = sum(1 for v in weak_cloud   if v[3])

    cloud_evidence       = n_sc  + 0.5 * n_w
    local_cloud_evidence = local_sc + 0.5 * local_w

    # 1. All-strong-clear with at most one boundary signal.
    if n_scl >= 3 and n_w <= 1 and n_sc == 0:
        twilight = sun_alt is not None and -18.0 <= sun_alt < -6.0
        sig = ", ".join(v[2] for v in strong_clear)
        weak_note = "" if n_w == 0 else f" (one boundary signal: {weak_cloud[0][2]})"
        conf = "medium" if twilight else "high"
        return "clear", conf, f"{n_scl} signals strongly clear ({sig}){weak_note}"

    # 2. Local clear majority - TRUST THE THERMAL PATCH
    if local_scl >= 2 and local_scl > local_cloud_evidence:
        if metar_okta is not None and metar_okta >= 6:
            confident_cloud = False
        else:
            sig = ", ".join(v[2] for v in strong_clear if v[3])
            return "clear", "medium", f"local says clear ({sig}); regional/weak signals disagree"

    # 3. Cloud evidence dominates
    if n_sc >= 2 or (n_sc >= 1 and n_w >= 2):
        confident_cloud = (n_scl == 0)
        # Fall through to family resolution

    # 4. Truly mixed
    else:
        cl_src = [v[0] for v in strong_cloud] + [f"~{v[0]}" for v in weak_cloud]
        cr_src = [v[0] for v in strong_clear]
        return "multi", "low", f"signals split cloud={cl_src} clear={cr_src}"

    # ---- Rule 4: deep convection (CB) via METAR genus ----
    cb_in_metar = metar_genus in ("CB", "TCU")
    cb_signals_strong = (cb_in_metar
                         and confident_cloud
                         and goes_phase in ("mixed", "ice")
                         and goes_height is not None and goes_height > 6000)
    if cb_signals_strong:
        return "ns_cb", "medium", \
               f"METAR {metar_genus} + GOES {goes_phase} top {goes_height:.0f}m"

    # ---- Rule 5: family from GOES height (preferred) or METAR ----
    family = None
    family_reason = ""
    if goes_height is not None and goes_height > 0:
        if goes_height < 2000:
            family = "low"
        elif goes_height < 6000:
            family = "mid"
        else:
            family = "high"
        family_reason = f"GOES height {goes_height:.0f}m → {family}"
    elif metar_bucket in ("low", "mid", "high"):
        family = metar_bucket
        family_reason = f"METAR bucket → {family}"

    if family is None:
        return "multi", "low", "cloud present but altitude family unknown"

    base_conf = "medium" if confident_cloud else "low"
    reasoning_bits = [family_reason]
    if goes_phase:
        reasoning_bits.append(f"phase={goes_phase}")
    if cb_in_metar:
        reasoning_bits.append(f"METAR genus={metar_genus}")

    # ---- Rule 6: genus within family ----
    if family == "high":
        if is_night and (thermal_mean_p is None or thermal_mean_p < 0.3):
            return "ci", "medium", "; ".join(
                reasoning_bits + ["nighttime high cloud, thermal_p<0.3 → Ci-like thin"])
        return "cs_cc", "low", "; ".join(reasoning_bits + ["Ci/Cs/Cc need RGB texture"])

    if family == "mid":
        return "ac_as", base_conf, "; ".join(reasoning_bits)

    if family == "low":
        if humidity is not None and humidity > 90:
            return "st", "medium", "; ".join(
                reasoning_bits + [f"humidity {humidity:.0f}% suggests St/fog"])
        if metar_genus in ("CU", "TCU"):
            return "cu", "medium", "; ".join(
                reasoning_bits + [f"METAR genus {metar_genus}"])
        if is_day and csi_std is not None and csi_std > 0.10:
            return "cu", "medium", "; ".join(reasoning_bits + [
                f"day + CSI 10-min std={csi_std:.2f} (convective shading) → Cu"])
        if (is_day and metar_okta is not None and 1 <= metar_okta <= 4
                and csi is not None and 0.55 <= csi <= 1.05):
            return "cu", "low", "; ".join(reasoning_bits + [
                f"day + METAR {metar_okta}/8 scattered + CSI {csi:.2f} → Cu over Sc"])
        return "sc", "low", "; ".join(reasoning_bits + ["Cu vs Sc needs RGB texture"])

    return "multi", "low", "fell through family rules"


if __name__ == "__main__":
    # Test cases removed for brevity — script is primarily for logic export.
    pass
