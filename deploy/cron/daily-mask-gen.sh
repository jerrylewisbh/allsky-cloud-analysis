#!/usr/bin/env bash
# Generate cloud masks for yesterday's captures (or a specific day).
# Output: ${PROJECT_DIR}/dataset_v2_YYYYMMDD/
#
# Usage:
#   daily-mask-gen.sh                       # yesterday, skip-existing (cron default)
#   daily-mask-gen.sh 20260524              # specific day, skip-existing
#   daily-mask-gen.sh 20260524 8            # specific day, 8 jobs, skip-existing
#   daily-mask-gen.sh 20260524 8 --force    # specific day, force regen all frames

source "$(dirname "$0")/_common.sh"

DAY="${1:-$(date -u -d 'yesterday' +%Y%m%d)}"
JOBS="${2:-4}"
FORCE_FLAG="${3:-}"

SKIP_EXISTING="--skip-existing"
if [ "${FORCE_FLAG}" = "--force" ]; then
    SKIP_EXISTING=""
    log "Force mode: regenerating ALL frames for ${DAY} (skip-existing disabled)"
fi

log "Mask generation for day=${DAY} (jobs=${JOBS}${SKIP_EXISTING:+, skip-existing})"

"${VENV}/bin/python" make_masks_v2.py \
    --day "${DAY}" \
    --allsky-root "${NAS_ALLSKY_PATH}" \
    --thermal-root "${NAS_THERMAL_PATH}/${NAS_THERMAL_UUID}" \
    --output-root "${PROJECT_DIR}/dataset_v2_${DAY}" \
    --jobs "${JOBS}" \
    ${SKIP_EXISTING}

N_OUT=$(ls "${PROJECT_DIR}/dataset_v2_${DAY}/masks/" 2>/dev/null | wc -l)
log "Generated ${N_OUT} masks for ${DAY}"
