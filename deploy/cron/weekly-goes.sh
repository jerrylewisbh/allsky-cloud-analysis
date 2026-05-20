#!/usr/bin/env bash
# Weekly GOES-19 fetch (ACMC + ACTPC + ACHAC, no COD per Calgary-latitude limit).
# Downloads ~250 MB/day to goes_cache/ — set up rotation if disk-constrained.

source "$(dirname "$0")/_common.sh"

log "GOES-19 fetch — products: ${GOES_PRODUCTS}"
# shellcheck disable=SC2086
"${VENV}/bin/python" fetch_goes.py \
    --products ${GOES_PRODUCTS} \
    --site-lat "${SITE_LAT}" --site-lon "${SITE_LON}"

CACHE_MB=$(du -sm "${PROJECT_DIR}/goes_cache" 2>/dev/null | awk '{print $1}')
log "GOES fetch done — cache size: ${CACHE_MB} MB"
