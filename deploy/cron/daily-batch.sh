#!/usr/bin/env bash
# Daily batch: orchestrates all data fetchers, mask generation, and auto-labeling.
# Safe to run manually anytime; it sources deploy/.env automatically.
#
# Usage:
#   daily-batch.sh                              # yesterday, 4 jobs
#   daily-batch.sh 20260524                     # specific date, 4 jobs
#   daily-batch.sh 20260524 8                   # specific date, 8 jobs
#   daily-batch.sh 20260524 8 --force           # force regen all frames (ignore skip-existing)
#   daily-batch.sh "" 8                         # yesterday, 8 jobs (pass empty date)
#   daily-batch.sh 20260524 8 --skip-fetch      # specific date, skip METAR/sensors/GOES re-fetch
#
# Notes:
#   - Date is YYYYMMDD; empty/missing = yesterday (UTC).
#   - METAR + local sensors + GOES fetchers are idempotent — re-running on
#     historical dates won't hurt, just wastes a few minutes. Use --skip-fetch
#     for fast historical mask-only runs.
#   - --force tells daily-mask-gen to regenerate every frame for the day,
#     overriding the default --skip-existing behavior.

set -e

# Jump to project root
cd "$(dirname "$0")/../.."
source "deploy/cron/_common.sh"

DAY="${1:-}"
JOBS="${2:-4}"

# Parse flags from $3 onwards (allows --force and --skip-fetch in any order)
FORCE_FLAG=""
SKIP_FETCH=0
for arg in "${@:3}"; do
    case "$arg" in
        --force) FORCE_FLAG="--force" ;;
        --skip-fetch) SKIP_FETCH=1 ;;
        *) log "WARNING: unknown flag '$arg' (expected --force or --skip-fetch)" ;;
    esac
done

DAY_DISPLAY="${DAY:-yesterday}"
log "=== Starting daily batch for ${DAY_DISPLAY} (jobs=${JOBS}${FORCE_FLAG:+, force}${SKIP_FETCH:+, skip-fetch}) ==="

# 1. Fetch metadata (Metar + Local Sensors)
if [ "${SKIP_FETCH}" = "0" ]; then
    log "Step 1/4: Fetching METAR and local sensors..."
    ./deploy/cron/daily-metar.sh
    ./deploy/cron/daily-local-sensors.sh
else
    log "Step 1/4: SKIPPED (--skip-fetch)"
fi

# 2. Parallel Processing:
#    - GOES download (slow, network bound)
#    - Mask generation (fast/CPU bound)
log "Step 2/4: Starting GOES fetch and Mask generation (jobs=${JOBS})..."
if [ "${SKIP_FETCH}" = "0" ]; then
    log "GOES fetch running in background..."
    ./deploy/cron/daily-goes.sh > /dev/null 2>&1 &
    GOES_PID=$!
else
    log "GOES fetch SKIPPED (--skip-fetch)"
    GOES_PID=""
fi

log "Mask generation starting for ${DAY_DISPLAY}..."
./deploy/cron/daily-mask-gen.sh "${DAY}" "${JOBS}" "${FORCE_FLAG}"

# 3. Wait for GOES to finish (if it was started)
if [ -n "${GOES_PID}" ]; then
    log "Step 3/4: Waiting for GOES fetch to complete (PID $GOES_PID)..."
    wait "$GOES_PID"
    log "GOES fetch finished."
else
    log "Step 3/4: GOES skipped (no PID to wait for)"
fi

# 4. Auto-classify
log "Step 4/4: Regenerating auto-labels..."
./deploy/cron/daily-auto-classify.sh

log "=== Daily batch complete for ${DAY_DISPLAY} ==="
