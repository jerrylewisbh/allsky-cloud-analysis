#!/usr/bin/env bash
# Catch-up script: find any day in the NAS that has frames but no (or
# incomplete) dataset_v2_ directory, and regenerate missing masks.
#
# Uses --skip-existing so frames already generated are preserved; only
# the gaps are filled. Safe to run any time, idempotent.
#
# Usage:
#   backfill-missing-days.sh                    # check all days, 4 jobs
#   backfill-missing-days.sh 8                  # check all days, 8 jobs
#   backfill-missing-days.sh 8 --force          # force regen ALL days

source "$(dirname "$0")/_common.sh"

JOBS="${1:-4}"
FORCE_FLAG="${2:-}"

log "=== Backfill scan starting ==="
log "  NAS:       ${NAS_ALLSKY_PATH}/images/"
log "  Local:     ${PROJECT_DIR}/dataset_v2_*"
log "  Jobs:      ${JOBS}"
[ "${FORCE_FLAG}" = "--force" ] && log "  Force:     yes (will regen all frames per day)"

# Find every day present on the NAS
NAS_DAYS=$(ls -1 "${NAS_ALLSKY_PATH}/images/" 2>/dev/null | grep -E '^[0-9]{8}$' | sort)
if [ -z "${NAS_DAYS}" ]; then
    log "No day directories found on NAS — nothing to do"
    exit 0
fi

TOTAL_DAYS=$(echo "${NAS_DAYS}" | wc -l)
log "Found ${TOTAL_DAYS} day(s) on NAS"
PROCESSED=0
SKIPPED=0

for DAY in ${NAS_DAYS}; do
    # Count raw frames on NAS for this day
    N_RAW=$(find "${NAS_ALLSKY_PATH}/images/${DAY}" -name "*.jpg" 2>/dev/null \
            | grep -v thumbnails | wc -l)
    # Count masks already generated locally
    N_OUT=$(ls "${PROJECT_DIR}/dataset_v2_${DAY}/masks/"*.png 2>/dev/null | wc -l)

    if [ "${N_RAW}" -eq 0 ]; then
        log "  ${DAY}: no raw frames on NAS, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [ "${FORCE_FLAG}" = "--force" ]; then
        log "  ${DAY}: force regen (${N_RAW} raw frames)"
    elif [ "${N_OUT}" -ge "${N_RAW}" ]; then
        log "  ${DAY}: complete (${N_OUT}/${N_RAW}), skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    else
        MISSING=$((N_RAW - N_OUT))
        log "  ${DAY}: ${N_OUT}/${N_RAW} done, ${MISSING} missing — generating"
    fi

    # Call daily-mask-gen.sh which handles --skip-existing logic
    "$(dirname "$0")/daily-mask-gen.sh" "${DAY}" "${JOBS}" "${FORCE_FLAG}"
    PROCESSED=$((PROCESSED + 1))
done

log "=== Backfill done: ${PROCESSED} day(s) processed, ${SKIPPED} skipped ==="
