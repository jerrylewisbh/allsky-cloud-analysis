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

  1. Precipitation OR strong CB hint → ns_cb (high)
  2. Vote across thermal_mean_p, GOES mask, daytime CSI, nighttime mpsas, METAR okta:
       - all signals agree clear → clear (high)
       - all signals agree cloud → continue to family (high)
       - local-overhead signals dominate → trust them, regional disagrees → multi (low)
  3. Family from GOES height (preferred) or METAR altitude_bucket
  4. Genus within family:
       - high → cs_cc (low, needs RGB)  [Ci is morphologically distinct]
       - mid  → ac_as (medium)
       - low  → st if humidity > 90 (medium), else sc (low, needs RGB)
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
             rgb_nrbr_mean: Optional[float] = None,
             rgb_v_mean: Optional[float] = None) -> tuple[str, str, str]:
    """Returns (class, confidence, reasoning).

    Parameters
    ----------
    weak : {(source, attribute): row_dict}
        Loaded from labels/weak_labels.csv, same shape as labeling_tool uses.
    thermal_mean_p : float, optional
        Mean cloud probability over the valid pixels of the local thermal mask.
    rgb_nrbr_mean : float, optional
        Mean Normalized Red-Blue Ratio (R-B)/(R+B) over the thermal-valid
        region of the RGB crop. Daytime cloud signal that bypasses the
        thermal-weak failure mode (Heinle 2010 §3). Daytime only.
    rgb_v_mean : float, optional
        Mean V (HSV brightness, 0–255) over the thermal-valid region of the
        long-exposure RGB crop. Nighttime cloud signal — thin cirrus reflects
        Calgary city skyglow and appears brighter than dark clear sky. Noisy
        because of exposure variation + moon, but catches some thin nighttime
        cloud the multimodal physics sensors all miss. Nighttime only.
    """
    # ---- pull signals ----
    sun_alt = _get(weak, "ephemeris", "sun_alt_deg", as_float=True)
    csi = _get(weak, "derived", "daytime_clear_sky_index", as_float=True)
    mpsas = _get(weak, "esp32_sensor", "sky_brightness_mpsas", as_float=True)
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

    # ---- Rule 1: active precipitation suggests ns_cb (but not high-confidence) ----
    # Observed at this site (2026-05-20 event): the visually-Ns frames LEAD the
    # AWNET rain measurement by ~25–30 min — the dense overcast moved through
    # before the gauge caught rain, and the trailing frames where the gauge
    # measured rain showed thinner/broken Sc with residual precipitation
    # falling through. So "rain at gauge" and "Ns visible overhead" are not the
    # same population. The rule still suggests ns_cb (precipitation is the most
    # actionable single signal) but at medium confidence — the labeler may
    # legitimately override to sc/multi when the cloud doesn't look the part.
    if rain_mm is not None and rain_mm > 0.5:
        return "ns_cb", "medium", \
               f"rain_1h_mm={rain_mm:.1f} > 0.5 (visual genus may differ — verify)"

    # ---- Rule 2: cloud-presence vote with three-tier semantics ----
    # Each vote: (signal_name, vote, reasoning_fragment, is_local)
    #   vote == True  → strong cloud
    #   vote == False → strong clear
    #   vote == None  → weak cloud (signal in the "thermal-weak" / boundary
    #                   range where cloud is plausibly present but the sensor
    #                   can't confirm). Weak votes downgrade confidence and
    #                   tilt the verdict cloud-ward without forcing it.
    votes: list[tuple[str, bool | None, str, bool]] = []

    if thermal_mean_p is not None:
        # The MLX90640 only sees the ~110°×75° patch directly overhead, not
        # the full fisheye view. A "clear" thermal reading means "no cloud
        # in the center patch" — clouds at the horizon are invisible to it.
        # So thermal-clear is downgraded to weak when METAR reports regional
        # BKN/OVC: a clear pocket overhead is plausible, but so is thin cloud
        # outside the narrow FOV that the labeler can see in the fisheye.
        if thermal_mean_p > 0.4:
            v = True
        elif thermal_mean_p > 0.2:
            v = None  # boundary: thermal sees something between noise and cloud
        elif thermal_mean_p < 0.05:
            v = False  # extremely confident clear in the center patch
        elif metar_okta is not None and metar_okta >= 5:
            v = None  # narrow-FOV clear contradicted by METAR BKN/OVC → weak
        else:
            v = False
        votes.append(("thermal", v, f"thermal_p={thermal_mean_p:.2f}", True))

    if goes_mask is not None:
        if goes_mask == 1:
            v = True
        elif goes_phase and goes_phase not in ("clear", None):
            # Mask says clear but the phase algorithm sees cloud nearby —
            # often means thin cloud just outside the cloud-mask threshold.
            v = None
        else:
            v = False
        votes.append(("goes", v, f"goes_mask={goes_mask} phase={goes_phase}", True))

    if is_day and csi is not None:
        # CSI > 1.1 or < 0.7: strong cloud (attenuation or enhancement).
        # CSI 1.05-1.1 or 0.7-0.85: weak — boundary, common in scattered Cu.
        # 0.85-1.05: strong clear.
        if csi > 1.1 or csi < 0.7:
            v = True
        elif csi > 1.05 or csi < 0.85:
            v = None
        else:
            v = False
        votes.append(("csi", v, f"csi={csi:.2f}", True))

    if is_night and mpsas is not None:
        # Urban Calgary: <17 strong cloud, 17-18 boundary, ≥18 clear.
        if mpsas < 17.0:
            v = True
        elif mpsas < 18.0:
            v = None
        else:
            v = False
        votes.append(("mpsas", v, f"mpsas={mpsas:.2f}", True))

    if metar_okta is not None:
        # BKN/OVC = strong cloud, SCT = weak (could be patchy), FEW/SKC = clear.
        if metar_okta >= 5:
            v = True
        elif metar_okta >= 3:
            v = None
        else:
            v = False
        votes.append(("metar", v, f"metar_okta={metar_okta}", False))

    if sky_cond:
        # Firmware's pessimistic-of-three. mostly_clear + partly_cloudy now
        # count as weak cloud (they previously got skipped, which is exactly
        # how high-confidence-wrong "clear" verdicts happened).
        if sky_cond in ("mostly_cloudy", "overcast"):
            v = True
        elif sky_cond in ("mostly_clear", "partly_cloudy"):
            v = None
        elif sky_cond in ("very_clear", "clear"):
            v = False
        else:
            v = None  # unknown firmware verdict — be cautious
        votes.append(("sky_cond", v, f"sky_condition={sky_cond}", True))

    if is_day and rgb_nrbr_mean is not None:
        # Normalized Red-Blue Ratio (R-B)/(R+B):
        #   ≲ -0.30  = blue sky (R much less than B)
        #   ≈ 0      = white (R ≈ B, classic cloud signature)
        #   > 0      = red-shifted (sunset/smoke/very thin haze near sun)
        # Captures visible cloud the thermal sensor + firmware miss
        # (Cu, thin Sc, daytime thin cirrus). Daytime only — RGB at night
        # carries no cloud signal without sun.
        if rgb_nrbr_mean > -0.10:
            v = True
        elif rgb_nrbr_mean < -0.30:
            v = False
        else:
            v = None
        votes.append(("rgb_nrbr", v, f"nrbr={rgb_nrbr_mean:+.2f}", True))

    if is_night and rgb_v_mean is not None:
        # Mean HSV V over the thermal-valid region of the long-exposure RGB.
        # Calibrated against the hand-labeled subset: cs_cc (high cirrus)
        # has median V≈126 (p25≈107), while clear frames extend up to ~80
        # because of Calgary's urban skyglow. Threshold at V=80 puts the
        # cutoff just above the clear distribution.
        #   V > 80 : strong cloud
        #   V > 50 : weak cloud (overlaps clear / sc / multi — half-weight)
        #   V < 20 : strong clear (genuinely dark)
        if rgb_v_mean > 80:
            v = True
        elif rgb_v_mean > 50:
            v = None  # weak — pushes toward cloud but doesn't override
        elif rgb_v_mean < 20:
            v = False
        else:
            v = None  # 20–50 is also weak (noise floor + dim skyglow)
        votes.append(("rgb_v_night", v, f"v_night={rgb_v_mean:.0f}", True))

    if not votes:
        return "clear", "low", "no signals available"

    strong_cloud = [v for v in votes if v[1] is True]
    strong_clear = [v for v in votes if v[1] is False]
    weak_cloud   = [v for v in votes if v[1] is None]
    n_sc, n_scl, n_w = len(strong_cloud), len(strong_clear), len(weak_cloud)

    local_sc  = sum(1 for v in strong_cloud if v[3])
    local_scl = sum(1 for v in strong_clear if v[3])
    local_w   = sum(1 for v in weak_cloud   if v[3])

    # cloud_evidence treats weak votes as half-weight cloud signals.
    cloud_evidence       = n_sc  + 0.5 * n_w
    local_cloud_evidence = local_sc + 0.5 * local_w

    # 1. All-strong-clear with at most one boundary signal.
    #    Requires 3+ strong-clear votes so one weak signal can be dismissed
    #    as noise. The previous version required zero weak signals, which
    #    with 6+ sources reporting per frame was essentially unreachable.
    #
    #    Regime-aware confidence cap: in NAUTICAL/ASTRO twilight (sun_alt
    #    −18°..−6°), thin Ci is optically invisible to every physics sensor
    #    (thermal, mpsas, GOES, METAR can all read "clear") while remaining
    #    visually obvious to a labeler watching wispy streaks against the
    #    post-sunset sky. Empirically, of 31 NAUTICAL frames where four
    #    sensors agreed clear, only ~20% were truly clear — the rest had
    #    Ci the sensors couldn't see. So we cap the verdict at "medium" in
    #    twilight. The previous attempt (raising the threshold to n_scl≥4)
    #    didn't help because the offending frames easily clear that bar —
    #    the information for thin-Ci-at-twilight simply isn't in the signals.
    #    "high" is reserved for DAY (sun visible — if it's truly clear,
    #    GOES + METAR + AWNET all confirm it) and DARK (no twilight Ci
    #    advantage to the human eye over the sensors).
    if n_sc == 0 and n_w <= 1 and n_scl >= 3:
        twilight = sun_alt is not None and -18.0 <= sun_alt < -6.0
        sig = ", ".join(v[2] for v in strong_clear)
        weak_note = "" if n_w == 0 else f" (one boundary signal: {weak_cloud[0][2]})"
        conf = "medium" if twilight else "high"
        twi_note = " — capped at medium (twilight Ci sensor blind spot)" if twilight else ""
        return "clear", conf, f"{n_scl} signals strongly clear ({sig}){weak_note}{twi_note}"

    # 2. No signals at all says cloud — but only one signal available.
    if n_sc == 0 and n_w == 0:
        return "clear", "low", "only one signal available, says clear"

    # 3. Weak hints + strong clear majority (and majority is LOCAL):
    #    means clear pocket with thin cloud nearby — predict clear, medium.
    if n_sc == 0 and local_scl >= 2 and local_scl > local_w * 1.5:
        weak_src = [v[0] for v in weak_cloud]
        sig = ", ".join(v[2] for v in strong_clear if v[3])
        return "clear", "medium", \
               f"local strongly clear ({sig}); weak cloud hints from {weak_src} insufficient"

    # 4. Weak cloud signals dominate with no strong cloud:
    #    proceed to family classification at LOW confidence.
    #    This is the new path for "thermal-weak cloud" frames.
    if n_sc == 0 and (n_w >= 2 or local_w >= 1):
        confident_cloud = False
        # Fall through to family rules below

    # 5. Strong cloud votes with no strong clear conflict:
    elif n_scl == 0 and n_sc >= 2:
        confident_cloud = True

    # 6. Local cloud signals dominate (with weak votes weighted):
    elif local_cloud_evidence > local_scl and (local_sc + local_w) >= 2:
        confident_cloud = (local_scl == 0 and local_sc >= 2)

    # 7. Local clear signals dominate — but defer to METAR if it sees regional
    #    BKN/OVC. A clear pocket overhead is plausible, but so is "thin cloud
    #    outside the narrow-FOV thermal sensor that the labeler can see in the
    #    full fisheye." When METAR contradicts, fall through to family rules
    #    as low-confidence cloud instead of forcing clear.
    elif local_scl >= 2 and local_scl > local_cloud_evidence:
        if metar_okta is not None and metar_okta >= 6:
            confident_cloud = False  # fall through to family resolution below
        else:
            sig = ", ".join(v[2] for v in strong_clear if v[3])
            return "clear", "medium", \
                   f"local says clear ({sig}); regional/weak signals disagree"

    # 8. Truly mixed — humans should adjudicate.
    else:
        cl_src = [v[0] for v in strong_cloud] + [f"~{v[0]}" for v in weak_cloud]
        cr_src = [v[0] for v in strong_clear]
        return "multi", "low", f"signals split cloud={cl_src} clear={cr_src}"

    # ---- Rule 3: deep convection (CB) ----
    cb_in_metar = metar_genus in ("CB", "TCU")
    cb_signals_strong = (cb_in_metar
                         and confident_cloud
                         and goes_phase in ("mixed", "ice")
                         and goes_height is not None and goes_height > 6000)
    if cb_signals_strong:
        return "ns_cb", "medium", \
               f"METAR {metar_genus} + GOES {goes_phase} top {goes_height:.0f}m"

    # ---- Rule 4: family from GOES height (preferred) or METAR ----
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

    # ---- Rule 5: genus within family ----
    if family == "high":
        # Ci (thin streaks) vs Cs (sheet) vs Cc (ripples) need RGB texture in
        # general, but there's one regime where physics decides: at night,
        # thin cirrus is optically thin, so GOES sees a high-altitude cloud
        # top while the local thermal sensor barely registers it. When the
        # thermal vote is absent or sub-0.3 despite a confirmed high-family
        # cloud, Ci is more likely than dense Cs/Cc.
        if is_night and (thermal_mean_p is None or thermal_mean_p < 0.3):
            return "ci", "medium", "; ".join(
                reasoning_bits + ["nighttime high cloud, thermal_p<0.3 → Ci-like thin"])
        return "cs_cc", "low", "; ".join(reasoning_bits + ["Ci/Cs/Cc need RGB texture"])

    if family == "mid":
        return "ac_as", base_conf, "; ".join(reasoning_bits)

    if family == "low":
        # Stratus / fog signature: high humidity at surface
        if humidity is not None and humidity > 90:
            return "st", "medium", "; ".join(
                reasoning_bits + [f"humidity {humidity:.0f}% suggests St/fog"])
        # Cumulus is convective (daytime) and discontinuous (METAR not OVC,
        # CSI showing intermittent shading rather than steady attenuation).
        # METAR genus hint is the most direct signal when available.
        if metar_genus in ("CU", "TCU"):
            return "cu", "medium", "; ".join(
                reasoning_bits + [f"METAR genus {metar_genus}"])
        if (is_day and metar_okta is not None and 1 <= metar_okta <= 4
                and csi is not None and 0.55 <= csi <= 1.05):
            return "cu", "low", "; ".join(reasoning_bits + [
                f"day + METAR {metar_okta}/8 scattered + CSI {csi:.2f} → Cu over Sc"])
        # Cu vs Sc otherwise requires RGB texture; default to Sc (more common
        # when local signals confirm continuous low cloud cover)
        return "sc", "low", "; ".join(reasoning_bits + ["Cu vs Sc needs RGB texture"])

    return "multi", "low", "fell through family rules"


# ---- self-test against the two walkthrough frames ----
if __name__ == "__main__":
    # Frame 1: ccd1_20260519_130836 — daytime overcast deck with CSI=1.2 enhancement
    f1 = {
        ("ephemeris", "sun_alt_deg"):      {"value": "58.5813"},
        ("derived",   "daytime_clear_sky_index"): {"value": "1.2000"},
        ("weather_station", "humidity_pct"):     {"value": "28.0"},
        ("weather_station", "rain_1h_mm"):       {"value": "0.0"},
        ("goes19_acmc",  "cloud_present"):       {"value": "1"},
        ("goes19_actpc", "cloud_top_phase"):     {"value": "mixed"},
        ("goes19_achac", "cloud_top_height_m"):  {"value": "3233.0"},
        ("metar", "coverage_okta"):              {"value": "8"},
        ("metar", "altitude_bucket"):            {"value": "mid"},
    }
    print("Frame 1 (ac_as expected):", classify(f1, thermal_mean_p=0.68))

    # Frame 2: ccd1_20260518_225250 — nautical twilight, METAR BKN+CB regional,
    # local sensors say clear. Post-patch: defers to METAR (Rule 7 guard) and
    # returns "sc low" instead of "clear medium". This is the intended fix:
    # the labeler often disagrees with the old "confident clear" verdict when
    # METAR reports substantial cloud the narrow-FOV sensors can't see.
    f2 = {
        ("ephemeris", "sun_alt_deg"):      {"value": "-11.06"},
        ("ephemeris", "moon_alt_deg"):     {"value": "12.37"},
        ("ephemeris", "moon_phase_pct"):   {"value": "8.6"},
        ("esp32_sensor", "sky_brightness_mpsas"): {"value": "18.61"},
        ("weather_station", "humidity_pct"):     {"value": "69.0"},
        ("weather_station", "rain_1h_mm"):       {"value": "0.0"},
        ("goes19_acmc",  "cloud_present"):       {"value": "0"},
        ("goes19_actpc", "cloud_top_phase"):     {"value": "water"},
        ("goes19_achac", "cloud_top_height_m"):  {"value": "1472.9"},
        ("metar", "coverage_okta"):              {"value": "6"},
        ("metar", "altitude_bucket"):            {"value": "mid"},
        ("metar", "cloud_genus_hint"):           {"value": "CB"},
    }
    print("Frame 2 (sc low expected — METAR override):", classify(f2, thermal_mean_p=0.01))

    # Edge: a raining frame (should short-circuit to ns_cb)
    f3 = {
        ("ephemeris", "sun_alt_deg"): {"value": "20.0"},
        ("weather_station", "rain_1h_mm"): {"value": "2.5"},
    }
    print("Frame 3 (ns_cb medium expected):", classify(f3))

    # Edge: no signals
    print("Frame 4 (clear low expected):", classify({}))
