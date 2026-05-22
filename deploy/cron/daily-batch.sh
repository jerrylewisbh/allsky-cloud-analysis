#!/usr/bin/env bash
# Daily batch: Orchestrates all data fetchers, mask generation, and auto-labeling.
# Safe to run manually anytime; it sources deploy/.env automatically.

set -e

# Jump to project root
cd "$(dirname "$0")/../.."
source "deploy/cron/_common.sh"

log "=== Starting daily batch ==="

JOBS="${1:-4}"

# 1. Fetch metadata (Metar + Local Sensors)
log "Step 1/4: Fetching METAR and local sensors..."
./deploy/cron/daily-metar.sh
./deploy/cron/daily-local-sensors.sh

# 2. Parallel Processing:
#    - GOES download (slow, network bound)
#    - Mask generation (fast/CPU bound)
log "Step 2/4: Starting GOES fetch and Mask generation (jobs=${JOBS})..."
log "GOES fetch running in background..."
./deploy/cron/daily-goes.sh > /dev/null 2>&1 &
GOES_PID=$!

log "Mask generation starting..."
./deploy/cron/daily-mask-gen.sh "" "${JOBS}"

# 3. Wait for GOES to finish
log "Step 3/4: Waiting for GOES fetch to complete (PID $GOES_PID)..."
wait $GOES_PID
log "GOES fetch finished."

# 4. Auto-classify
log "Step 4/4: Regenerating auto-labels..."
./deploy/cron/daily-auto-classify.sh

log "=== Daily batch complete! ==="
