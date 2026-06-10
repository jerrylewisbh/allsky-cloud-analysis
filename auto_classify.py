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
# The classifier may also emit "unknown" when signals can't be reconciled
# (vote cascade fails, family logic falls through, etc.). "unknown" is NOT
# a hand-label option — it's a fallback that means "labeler should decide".
# Kept distinct from "multi" so the diagnostic doesn't conflate them.


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
    moon_alt = _get(weak, "ephemeris", "moon_alt_deg", as_float=True)
    moon_phase = _get(weak, "ephemeris", "moon_phase_pct", as_float=True)
    csi = _get(weak, "derived", "daytime_clear_sky_index", as_float=True)
    # ESP32 lux/mpsas/sky_condition intentionally not used — sensor is unreliable.
    # Night cloud presence now relies on thermal_mean_p, GOES, METAR, and rgb_v_night.
    humidity = _get(weak, "weather_station", "humidity_pct", as_float=True)
    rain_mm = _get(weak, "weather_station", "rain_1h_mm", as_float=True)
    goes_mask = _get(weak, "goes19_acmc", "cloud_present", as_int=True)
    goes_phase = _get(weak, "goes19_actpc", "cloud_top_phase")
    goes_height = _get(weak, "goes19_achac", "cloud_top_height_m", as_float=True)
    metar_okta = _get(weak, "metar", "coverage_okta", as_int=True)
    metar_genus = _get(weak, "metar", "cloud_genus_hint")
    metar_bucket = _get(weak, "metar", "altitude_bucket")

    # Day/night split at the horizon — twilight frames need to contribute too.
    # CSI is None below sun_alt ≈ 0 (clear-sky model unstable), so its vote
    # simply won't fire there; rgb_nrbr is meaningful any time the sun is up.
    # rgb_v_night is the long-exposure RGB and is meaningful any time the sun
    # is down, including civil/nautical twilight.
    is_day = sun_alt is not None and sun_alt > 0.0
    # Exactly at the horizon (sun_alt == 0.0) counts as night so the texture
    # branch in Rule 0 and the night genus logic still fire — otherwise both
    # is_day and is_night would be False and a textured frame would silently
    # fall through with its evidence discarded. When sun_alt is None (no
    # ephemeris) both stay False, which is the correct "unknown" degradation.
    is_night = sun_alt is not None and sun_alt <= 0.0
    csi_std = _get(weak, "derived", "csi_std_10min", as_float=True)

    # ---- Rule -1: layered multi-cloud (high cirrus over a lower warm layer) ----
    # GOES high COLD ice cloud cannot produce warm thermal pixels. When GOES
    # reports an ice top while the local thermal patch shows WARM, textured
    # pixels, the warmth must come from a second, lower layer below/between the
    # cirrus → multi. Must run before Rule 0, whose night-texture branch would
    # otherwise return ac_as. The mean ceiling (<=0.65) keeps this disjoint from
    # the opaque glaciated-deck case (Rule 5 ns_cb / Sc, mean>0.6 + uniform).
    #
    # Yield to Rule 4.5 when its stronger layered signature holds (top >8km,
    # mean>0.30, std>0.07): both rules emit "multi", but 4.5 is medium-confidence
    # and this one is low, so without this guard the higher-confidence verdict is
    # unreachable in the overlap. Safe to fall through: goes_height>=7000 already
    # skips Rule 0 (high_family_from_goes), so these frames can't be misrouted to
    # Rule 0's night branch — they reach Rule 4.5 via the vote cascade.
    strong_layered = (goes_height is not None and goes_height > 8000
                      and thermal_mean_p is not None and thermal_mean_p > 0.30
                      and thermal_std is not None and thermal_std > 0.07)
    if (not strong_layered
            and goes_phase == "ice" and goes_height is not None and goes_height > 6000
            and thermal_mean_p is not None and 0.25 < thermal_mean_p <= 0.65
            and thermal_std is not None and thermal_std > 0.10):
        return "multi", "low", (
            f"GOES high ice top {goes_height:.0f}m + warm textured thermal "
            f"(mean={thermal_mean_p:.2f}, std={thermal_std:.2f}) → "
            f"high cirrus over a lower layer = multi"
        )

    # ---- Rule 0: thermal spatial-variance texture (overrides clear cascade) ----
    # True-clear thermal_std tops out near 0.04 in this dataset, so 0.07 is a
    # safe floor for "structured". The mean lower bound is dropped because
    # broken Cu against a narrow thermal FOV often has very low overall
    # mean_p (~0.04) — the texture is in the std, not the mean.
    # High-family threshold is 7000m (not 6000m): at mid-latitudes Ac tops
    # can reach 5-7 km in spring/summer, while genuine Ci typically lives at
    # 8 km+. The 6 km cutoff routed too many Ac frames to cs_cc.
    high_family_from_goes = (goes_height is not None and goes_height >= 7000)
    if (not high_family_from_goes
            and thermal_std is not None and thermal_std > 0.07
            and thermal_mean_p is not None and thermal_mean_p < 0.70):
        texture_note = f"thermal std={thermal_std:.2f} mean={thermal_mean_p:.2f}"
        if is_day:
            # METAR is authoritative when present
            if metar_okta is not None and metar_okta >= 5:
                return "sc", "medium", \
                       f"daytime broken-cloud texture ({texture_note}) + METAR {metar_okta}/8 → broken Sc"
            # METAR missing or scattered: thermal_mean_p disambiguates cu vs sc.
            # Cu = discrete cells with blue gaps → low mean (most pixels are clear sky).
            # Sc = continuous lumpy deck → mid-to-high mean (most pixels are cloud,
            #      the texture is thickness variation not gaps).
            if thermal_mean_p >= 0.22:
                return "sc", "medium", \
                       f"daytime textured deck ({texture_note}) — mean≥0.22 → Sc (deck with thickness variation, not Cu gaps)"
            return "cu", "medium", \
                   f"daytime broken-cloud texture ({texture_note}) — low mean → Cu over blue gaps"
        if is_night:
            return "ac_as", "medium", \
                   f"night broken-cloud texture ({texture_note}) → Ac/broken Sc"

    # ---- Rule 0.5: Visual Puff detector (daytime only) ----
    if is_day and rgb_nrbr_p95 is not None and rgb_nrbr_p95 > 0.0:
        if thermal_mean_p is not None and thermal_mean_p < 0.30:
            return "cu", "low", f"visually stark white peak (nrbr_p95={rgb_nrbr_p95:+.2f}) → likely sparse Cu"

    # ---- Rule 1: active precipitation suggests ns_cb ----
    # 0.1 mm threshold (was 0.5): tipping-bucket gauges lag actual onset by
    # 10-15 minutes — the first wave of rain often reads 0.0-0.4 mm even
    # with visible droplets on the lens. Lowering catches the active-rain
    # regime earlier so the trailing-edge storm frames don't get routed to
    # sc/cs_cc when they're still ns_cb.
    if rain_mm is not None and rain_mm > 0.1:
        cb_signature = (goes_phase == "ice"
                        and goes_height is not None and goes_height > 7000
                        and humidity is not None and humidity > 90)
        if cb_signature:
            return "ns_cb", "high", \
                   (f"rain {rain_mm:.1f}mm + GOES ice top {goes_height:.0f}m + "
                    f"humidity {humidity:.0f}% → deep convective ns_cb")
        return "ns_cb", "medium", \
               f"rain_1h_mm={rain_mm:.2f} > 0.1 (visual genus may differ — verify)"

    # ---- Rule 2: targeted high-ice cs_cc (locals can't see thin cirrus) ----
    # GOES high-altitude ice phase is the most reliable cirriform indicator
    # we have. When that fires AND the local thermal patch reads near-clear
    # (the thin cirrus is below the thermal sensitivity floor — happens to
    # ~13/81 of hand-labeled cs_cc frames), trust GOES. Returns cs_cc rather
    # than ci because Cs/Cc are more common than isolated mare's-tails Ci at
    # our latitude (Calgary). Labeler overrides to ci when fibrous visible.
    # Note: METAR requirement dropped — most twilight/night frames have no
    # METAR update within the window, blocking the rule from firing on the
    # very frames it's designed to catch.
    if (goes_phase == "ice"
            and goes_height is not None and goes_height > 7000
            and goes_mask == 1
            and thermal_mean_p is not None and thermal_mean_p < 0.20):
        return "cs_cc", "medium", \
               (f"GOES high-ice cloud (top {goes_height:.0f}m, mask=1) + "
                f"thermal_p={thermal_mean_p:.2f} → thin Cs/Cc")

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

    # CSI is unreliable when the sun is very low (sun_alt < 10°): the clear-sky
    # model breaks down at long atmospheric path lengths, and tiny absolute
    # irradiance values give large relative errors. At sunrise (sun_alt ≈ 0°)
    # CSI routinely reads 0.2-0.5 even on cloudless mornings. Demote to weak
    # vote in this regime, abstain entirely if extremely low.
    csi_reliable = is_day and sun_alt is not None and sun_alt > 10.0
    if csi_reliable and csi is not None:
        # Widened cloud bands: CSI < 0.80 means >20% of expected irradiance is
        # missing — almost always partial cloud shading. The old < 0.70 floor
        # let moderate Cu/Sc shading slip into the "weak" bucket.
        # CSI > 1.15 captures cloud-edge reflection brightening near the sun.
        if csi < 0.80 or csi > 1.15:
            v = True
        elif csi < 0.92 or csi > 1.08:
            v = None
        else:
            v = False
        votes.append(("csi", v, f"csi={csi:.2f}", True))
    elif is_day and csi is not None and sun_alt is not None and 0 <= sun_alt <= 10:
        # Low-sun (sun_alt 0-10°): never vote strong-cloud or strong-clear here —
        # path-length attenuation makes CSI unreliable. Only an implausibly low
        # CSI (< 0.3) implies real cloud rather than attenuation, and then only
        # as a single weak vote. In the plausible-low-sun band (csi >= 0.3),
        # abstain entirely.
        if csi >= 0.3:
            pass  # don't append a vote — abstain
        else:
            votes.append(("csi", None, f"csi={csi:.2f} (low-sun, weak)", True))

    if metar_okta is not None:
        if metar_okta >= 5:
            v = True
        elif metar_okta >= 3:
            v = None
        else:
            v = False
        votes.append(("metar", v, f"metar_okta={metar_okta}", False))

    if is_day and rgb_nrbr_p95 is not None:
        if rgb_nrbr_p95 > -0.15:
            v = True
        elif rgb_nrbr_p95 < -0.50:
            v = False
        else:
            v = None
        votes.append(("rgb_nrbr_peak", v, f"nrbr_p95={rgb_nrbr_p95:+.2f}", True))

    if is_night and rgb_v_mean is not None and rgb_v_std is not None:
        # Brightness OR texture alone is unreliable — moonlight inflates v_mean,
        # and warm sensor noise inflates v_std. Require both for a cloud vote.
        #
        # The strong-CLEAR branch requires a light source (moon up + >10% phase
        # OR nautical twilight): at deep night with no moon, the patch is dark
        # whether it's clear OR thinly cloudy, so "dark = clear" is invalid.
        #
        # The strong-CLOUD branch requires sun to be well below horizon
        # (sun_alt < -6°): during civil twilight, sky brightening from
        # approaching/receding sun produces high v_mean + v_std identical to
        # cloud-reflected light. Without this guard, sunrise/sunset frames
        # false-positive as cloud and block the thermal_veto.
        light_available = (
            (sun_alt is not None and sun_alt > -12.0)
            or (moon_alt is not None and moon_alt > 0 and
                moon_phase is not None and moon_phase > 10)
        )
        sun_safely_down = sun_alt is not None and sun_alt < -6.0
        if rgb_v_std > 15.0 and rgb_v_mean > 80 and sun_safely_down:
            v = True
        elif rgb_v_mean < 50 and rgb_v_std < 5.0 and light_available:
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
    #    Texture guard: the scalar votes can't see spatial structure. If the
    #    thermal patch shows std > 0.05 (above the clear-sky ceiling of ~0.04),
    #    refuse to declare high-confidence clear — fall through to the rest of
    #    the cascade so the texture signal can route to cu/sc/ac_as.
    has_texture = thermal_std is not None and thermal_std > 0.05
    if n_scl >= 3 and n_w <= 1 and n_sc == 0 and not has_texture:
        # Cap confidence at medium any time the sun is near or below the horizon —
        # signal quality degrades smoothly from civil twilight through deep night.
        near_horizon = sun_alt is not None and sun_alt < 6.0
        sig = ", ".join(v[2] for v in strong_clear)
        weak_note = "" if n_w == 0 else f" (one boundary signal: {weak_cloud[0][2]})"
        conf = "medium" if near_horizon else "high"
        return "clear", conf, f"{n_scl} signals strongly clear ({sig}){weak_note}"

    # 2. Local clear majority - TRUST THE THERMAL PATCH
    #    Two ways in:
    #      (a) >= 2 local strong-clear signals beat local cloud evidence, OR
    #      (b) thermal_veto (thermal_mean_p < 0.02): the patch has essentially
    #          no warm pixels, so as long as no other LOCAL signal screams
    #          cloud and texture is absent, trust it. Regional disagreement
    #          (GOES/METAR) is a FOV mismatch, not a contradiction.
    #
    #    Comparison uses >= (not >) to avoid a weak vote tying and defeating
    #    the local clear majority.
    #
    #    RGB veto: if the peak whiteness (nrbr_p95) is above -0.15, there are
    #    pixels brighter than typical clear sky — fall through so cu/sc fire.
    #    Between -0.35 and -0.15, the frame is "RGB-suspicious" — still return
    #    clear, but at low confidence. Below -0.35 → medium confidence.
    # Threshold 0.05 (was 0.02): accommodates ~0.02-0.03 baseline bias from
    # residual sensor-edge contamination that survives manual cleaning. True
    # clouds read >0.20 mean so the wider veto can't false-trigger on cloud.
    thermal_veto = (
        thermal_mean_p is not None and thermal_mean_p < 0.05
        and (thermal_std is None or thermal_std < 0.05)
    )
    veto_path = thermal_veto and local_sc == 0
    majority_path = local_scl >= 2 and local_scl >= local_cloud_evidence
    if veto_path or majority_path:
        rgb_suspicious = rgb_nrbr_p95 is not None and rgb_nrbr_p95 > -0.15
        metar_overcast = metar_okta is not None and metar_okta >= 6
        # FOV-mismatch handling: METAR is regional (airport ~10 km away),
        # the thermal patch is a direct local measurement. When thermal_veto
        # fires (mean<0.02, std<0.05 — patch is unambiguously clear), only
        # RGB (also local) can block — METAR disagreement is interpreted as
        # FOV mismatch, not as evidence against clear. For majority_path
        # (multiple weaker local clears), METAR still blocks since the
        # local evidence is less rock-solid.
        # Thin-cirrus exception to the FOV-mismatch rule: if the thermal patch
        # is blind (veto) but BOTH GOES and METAR independently report a
        # frozen/mixed-phase cloud, it isn't a clear-overhead/cloudy-airport
        # mismatch — it's high thin cloud below the thermal sensitivity floor.
        # Water-phase regional cloud still falls through to the clear return
        # (preserving the deliberate "trust local thermal" design).
        # RGB whiteness band is the separator vs genuinely-clear twilight: a
        # truly clear sky stays strongly blue (nrbr_p95 ~ -0.47) even when the
        # distant airport reports overcast, whereas thin cirrus overhead lifts
        # nrbr_p95 into the -0.40..-0.15 band. Without the lower bound this rule
        # flips clear twilight frames (GOES+METAR see a regional deck) to cs_cc.
        thin_cirrus_whiteness = (
            rgb_nrbr_p95 is not None and -0.40 < rgb_nrbr_p95 <= -0.15
        )
        if (veto_path and thin_cirrus_whiteness
                and goes_mask == 1 and goes_phase in ("ice", "mixed")
                and metar_okta is not None and metar_okta >= 6):
            return "cs_cc", "low", (
                f"thermal blind (thermal_p={thermal_mean_p:.2f}) but GOES+METAR "
                f"agree {goes_phase} cloud (METAR {metar_okta}/8) + RGB whiteness "
                f"(nrbr_p95={rgb_nrbr_p95:+.2f}) → thin Cs/Cc invisible to thermal"
            )
        if veto_path:
            block = rgb_suspicious
        else:
            block = rgb_suspicious or metar_overcast
        if not block:
            sig = ", ".join(v[2] for v in strong_clear if v[3]) or f"thermal_p={thermal_mean_p:.2f}"
            if rgb_nrbr_p95 is not None and rgb_nrbr_p95 > -0.35:
                return "clear", "low", \
                       f"local says clear ({sig}); RGB nrbr_p95={rgb_nrbr_p95:+.2f} hints at cloud — verify"
            fov_note = " (METAR overcast = FOV mismatch from airport)" if metar_overcast else ""
            return "clear", "medium", \
                   f"local says clear ({sig}); regional/weak signals disagree{fov_note}"
        # else: fall through to cloud-evidence resolution — local RGB contradicts thermal.

    # 3. Cloud evidence dominates
    if n_sc >= 2 or (n_sc >= 1 and n_w >= 2):
        confident_cloud = (n_scl == 0)
        # Fall through to family resolution

    # 3b. Strong local thermal overrides signal-split punt.
    #     The thermal patch is the direct measurement of what's overhead.
    #     If it shows ≥50% coverage but the cascade would otherwise punt
    #     because of regional disagreement, trust the patch and let Rule 5's
    #     local-only-cloud fallback infer low family + Rule 6 pick the genus.
    elif thermal_mean_p is not None and thermal_mean_p > 0.5:
        confident_cloud = False  # regional signals disagree, hence low conf
        # Fall through to family resolution

    # 3c. Single cloud signal with no strong contradiction → trust it.
    #     When the only cloud evidence is one strong signal (typically GOES at
    #     night) and no strong-clear signal disagrees, route to family rather
    #     than punting. Avoids unknown for "thin sparse signal coverage" frames
    #     where the cascade only has 1-2 votes and one of them says cloud.
    elif n_sc >= 1 and n_scl == 0:
        confident_cloud = False  # only one cloud signal, low conf
        # Fall through to family resolution

    # 3d. Weak consensus: many weak cloud-leaning signals + zero strong clear.
    #     Thin uniform cloud (As, Cs, thin Ac) often shows as multiple weak
    #     votes rather than a single strong one — borderline thermal, GOES
    #     with phase-but-no-mask, METAR scattered, dim RGB at night. The
    #     cascade should treat this aggregate as evidence rather than punt.
    elif n_w >= 3 and n_scl == 0:
        confident_cloud = False  # thin cloud — low conf is honest
        # Fall through to family resolution

    # 4. Truly mixed (no strong cloud signal at all, OR cloud signal contradicted by clears)
    else:
        cl_src = [v[0] for v in strong_cloud] + [f"~{v[0]}" for v in weak_cloud]
        cr_src = [v[0] for v in strong_clear]
        return "unknown", "low", f"signals split cloud={cl_src} clear={cr_src}"

    # ---- Rule 4: deep convection (CB) via METAR genus ----
    cb_in_metar = metar_genus in ("CB", "TCU")
    cb_signals_strong = (cb_in_metar
                         and confident_cloud
                         and goes_phase in ("mixed", "ice")
                         and goes_height is not None and goes_height > 7000)
    if cb_signals_strong:
        return "ns_cb", "medium", \
               f"METAR {metar_genus} + GOES {goes_phase} top {goes_height:.0f}m"

    # ---- Rule 4.5: layered multi-cloud detection (cu/sc below ci/cs) ----
    # When GOES reports high-confidence cirriform cloud (ice phase at >8km)
    # AND the local thermal patch shows STRUCTURED warm regions (texture +
    # mid-to-high mean), the warm pixels can't be from the cold cirrus —
    # there must be a lower cloud layer visible below/between the cirrus.
    # Classic multi-cloud signature that the labeler often can't resolve
    # visually because the cirrus is too thin to see against twilight RGB.
    if (goes_phase == "ice" and goes_height is not None and goes_height > 8000
            and thermal_mean_p is not None and thermal_mean_p > 0.30
            and thermal_std is not None and thermal_std > 0.07):
        return "multi", "medium", (
            f"GOES high ice top {goes_height:.0f}m + textured warm thermal "
            f"(mean={thermal_mean_p:.2f}, std={thermal_std:.2f}) → "
            f"low/mid cloud layer below high cirriform = multi"
        )

    # ---- Rule 4.6: glaciated opaque deck → ns_cb (before family forces low) ----
    # A uniform opaque deck (thermal_mean_p>0.6, low std) whose tops are FROZEN
    # (GOES phase ice/mixed) is glaciated deep cloud (Ns/Cb), not warm low Sc —
    # true Sc is water-phase. The rain gauge often lags onset, so rain=0 doesn't
    # rule out Ns. Without this, Rule 5 below forces family="low" on high opacity
    # and Rule 6 returns Sc. Conservative guard: require uniformity (std<0.10) so
    # textured altostratus isn't swept in. Low confidence — labeler verifies.
    if (thermal_mean_p is not None and thermal_mean_p > 0.6
            and goes_phase in ("ice", "mixed")
            and metar_okta is not None and metar_okta >= 6
            and thermal_std is not None and thermal_std < 0.10):
        return "ns_cb", "low", (
            f"opaque uniform deck (thermal_p={thermal_mean_p:.2f}, "
            f"std={thermal_std:.2f}) + frozen tops (phase={goes_phase}) + "
            f"METAR {metar_okta}/8 → glaciated deep cloud (ns_cb)"
        )

    # ---- Rule 5: family from GOES height (preferred) or METAR ----
    family = None
    family_reason = ""
    if goes_height is not None and goes_height > 0:
        # The thermal-opacity overrides below are only trustworthy at NIGHT. In
        # daytime, solar heating of the ZnSe window/enclosure inflates
        # thermal_mean_p (a warm bias unrelated to cloud height), so a thin
        # mid/high layer reads "opaque" and would be wrongly demoted to low (Sc).
        # In daylight, defer to the GOES cloud-top height (the preferred signal).
        # ...and never for an ice-phase cloud: ice means cold/high (cirriform),
        # so a warm thermal under a GOES-confirmed ice top is a window-heating
        # artifact (the window stays warm into twilight/early night) or a layer
        # below high cirrus — never grounds to demote the family to low Sc.
        if (not is_day) and goes_phase != "ice" and thermal_mean_p is not None and thermal_mean_p > 0.60:
            family = "low"
            family_reason = f"GOES height {goes_height:.0f}m but high opacity (night, non-ice) suggests {family}"
        elif (not is_day) and goes_phase != "ice" and thermal_mean_p is not None and thermal_mean_p > 0.40 and goes_height > 2000:
            family = "mid"
            family_reason = f"GOES height {goes_height:.0f}m but moderate opacity (night, non-ice) suggests {family}"
        elif goes_height < 2000:
            family = "low"
            family_reason = f"GOES height {goes_height:.0f}m → {family}"
        elif goes_height < 7000 and goes_phase == "ice":
            # Phase-aware override for borderline 5-7 km: pure ice at this
            # height is much more often Cc/Cs (cirriform just below the 7 km
            # cutoff) than Ac (which is water/mixed phase at mid-latitudes).
            family = "high"
            family_reason = (
                f"GOES height {goes_height:.0f}m + ice phase → {family} "
                "(cirriform despite mid-range height)"
            )
        elif goes_height < 7000:
            family = "mid"
            family_reason = f"GOES height {goes_height:.0f}m → {family}"
        else:
            family = "high"
            family_reason = f"GOES height {goes_height:.0f}m → {family}"
    elif metar_bucket in ("low", "mid", "high"):
        family = metar_bucket
        family_reason = f"METAR bucket → {family}"

    if family is None:
        # Local-only cloud fallback: thermal patch screams cloud but every
        # regional signal (GOES, METAR) says clear sky → no cloud-top height
        # is available to set the family. By elimination this must be LOW
        # cloud — GOES and airport observers reliably see mid/high cloud at
        # this scale, so localized cu/sc is the only thing that fits.
        # Rule 6 then routes cu vs sc via thermal_std / METAR genus.
        if thermal_mean_p is not None and thermal_mean_p > 0.4:
            family = "low"
            family_reason = (
                f"local-only cloud (thermal_p={thermal_mean_p:.2f}, "
                "no regional height) → must be low cu/sc"
            )
        else:
            return "unknown", "low", "cloud present but altitude family unknown"

    base_conf = "medium" if confident_cloud else "low"
    reasoning_bits = [family_reason]
    if goes_phase:
        reasoning_bits.append(f"phase={goes_phase}")
    if cb_in_metar:
        reasoning_bits.append(f"METAR genus={metar_genus}")

    # ---- Rule 6: genus within family ----
    if family == "high":
        # Default to cs_cc (Cs/Cc) rather than ci: at mid-latitudes (e.g. Calgary)
        # cirrostratus sheets and cirrocumulus ripples are more common than
        # isolated mare's-tails cirrus. Rules can't distinguish these without
        # RGB texture; flipping the default minimizes systematic mislabeling
        # of Cc/Cs as Ci. Labeler overrides to `ci` when fibrous streaks visible.
        if is_night and (thermal_mean_p is None or thermal_mean_p < 0.3):
            return "cs_cc", "medium", "; ".join(
                reasoning_bits + ["nighttime high cloud, thermal_p<0.3 → thin Cc/Cs (override to ci if fibrous)"])
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
        # Opacity outranks CSI variability: a uniformly cloud-covered thermal
        # patch (mean > 0.50) is a deck (Sc), even if CSI fluctuates. The CSI
        # std → Cu inference only holds when the patch shows gaps (mean < 0.50);
        # otherwise the variability is thickness changes in a continuous deck.
        if thermal_mean_p is not None and thermal_mean_p > 0.50:
            return "sc", "medium", "; ".join(reasoning_bits + [
                f"low opaque deck (thermal_p={thermal_mean_p:.2f}) → Sc"])
        if is_day and csi_std is not None and csi_std > 0.10:
            return "cu", "medium", "; ".join(reasoning_bits + [
                f"day + CSI 10-min std={csi_std:.2f} (convective shading) → Cu"])
        if (is_day and metar_okta is not None and 1 <= metar_okta <= 4
                and csi is not None and 0.55 <= csi <= 1.05):
            return "cu", "low", "; ".join(reasoning_bits + [
                f"day + METAR {metar_okta}/8 scattered + CSI {csi:.2f} → Cu over Sc"])
        return "sc", "low", "; ".join(reasoning_bits + ["Cu vs Sc needs RGB texture"])

    return "unknown", "low", "fell through family rules"


if __name__ == "__main__":
    pass
