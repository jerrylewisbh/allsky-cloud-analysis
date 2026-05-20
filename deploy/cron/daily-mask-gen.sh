#!/usr/bin/env bash
# Generate cloud masks for yesterday's captures.
# Output: ${PROJECT_DIR}/dataset_v2_YYYYMMDD/

source "$(dirname "$0")/_common.sh"

DAY="${1:-$(date -u -d 'yesterday' +%Y%m%d)}"
log "Mask generation for day=${DAY}"

"${VENV}/bin/python" make_masks_v2.py \
    --day "${DAY}" \
    --allsky-root "${NAS_ALLSKY_PATH}" \
    --thermal-root "${NAS_THERMAL_PATH}/${NAS_THERMAL_UUID}" \
    --output-root "${PROJECT_DIR}/dataset_v2_${DAY}"

N_OUT=$(ls "${PROJECT_DIR}/dataset_v2_${DAY}/masks/" 2>/dev/null | wc -l)
log "Generated ${N_OUT} masks for ${DAY}"
