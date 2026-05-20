#!/usr/bin/env bash
# Weekly METAR pull for both stations. Re-runs are idempotent (dedup by key).

source "$(dirname "$0")/_common.sh"

log "METAR fetch — primary ${METAR_STATION_PRIMARY}"
"${VENV}/bin/python" fetch_metar.py \
    --station "${METAR_STATION_PRIMARY}" \
    --site-lat "${SITE_LAT}" --site-lon "${SITE_LON}" \
    --station-lat "${METAR_STATION_PRIMARY_LAT}" \
    --station-lon "${METAR_STATION_PRIMARY_LON}"

log "METAR fetch — secondary ${METAR_STATION_SECONDARY}"
"${VENV}/bin/python" fetch_metar.py \
    --station "${METAR_STATION_SECONDARY}" \
    --site-lat "${SITE_LAT}" --site-lon "${SITE_LON}" \
    --station-lat "${METAR_STATION_SECONDARY_LAT}" \
    --station-lon "${METAR_STATION_SECONDARY_LON}"

log "METAR fetch done"
