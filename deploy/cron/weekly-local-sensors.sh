#!/usr/bin/env bash
# Weekly pull of AWNET weather + ESP sensors + ephemeris from PG.

source "$(dirname "$0")/_common.sh"

log "Local sensors fetch from PG ${PG_HOST}:${PG_PORT}/${PG_DB}"
"${VENV}/bin/python" fetch_local_sensors.py \
    --pg-host "${PG_HOST}" \
    --pg-port "${PG_PORT}" \
    --pg-db "${PG_DB}" \
    --pg-user "${PG_USER}" \
    --pg-pass "${PG_PASS}"

log "Local sensors fetch done"
