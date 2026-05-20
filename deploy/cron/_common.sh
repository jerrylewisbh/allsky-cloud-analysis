# Common setup sourced by every cron wrapper in this directory.
# Not executable by itself.

set -euo pipefail

# Resolve PROJECT_DIR even if cron invokes us from a weird cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load .env from the project root
ENV_FILE="${PROJECT_DIR_DEFAULT}/deploy/.env"
if [ -f "${ENV_FILE}" ]; then
    # shellcheck source=/dev/null
    set -a
    source "${ENV_FILE}"
    set +a
fi

PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
VENV="${PROJECT_DIR}/.venv"
LOG_DIR="${LOG_DIR:-/var/log/allsky-cloud-analysis}"

mkdir -p "${LOG_DIR}"

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$(basename "$0")] $*"
}

# Sanity: venv must exist
if [ ! -x "${VENV}/bin/python" ]; then
    log "ERROR: venv missing at ${VENV} — run deploy/server-setup.sh first"
    exit 1
fi

cd "${PROJECT_DIR}"
