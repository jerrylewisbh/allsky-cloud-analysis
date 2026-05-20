#!/usr/bin/env bash
# One-shot install/refresh for the allsky-cloud-analysis server deployment.
# Run on the server, from the project root:
#   ./deploy/server-setup.sh
#
# Idempotent — safe to re-run after a git pull to rebuild the container,
# refresh deps, and re-install the crontab.

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR_LOCAL="$(pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

bold "=== allsky-cloud-analysis server setup ==="
echo "Project dir: ${PROJECT_DIR_LOCAL}"

# ---------- 1. .env ----------
bold "--- Configuration ---"
if [ ! -f deploy/.env ]; then
    if [ -f deploy/.env.example ]; then
        cp deploy/.env.example deploy/.env
        echo "Created deploy/.env from example."
        warn "Edit deploy/.env to match your server (paths, PG credentials, site coords)"
        warn "Then re-run this script."
        exit 0
    else
        die "deploy/.env not found and no example to copy from"
    fi
fi

set -a
# shellcheck source=/dev/null
source deploy/.env
set +a
echo "Loaded deploy/.env"

# Sanity: PROJECT_DIR in .env should match where we are
if [ "${PROJECT_DIR}" != "${PROJECT_DIR_LOCAL}" ]; then
    warn "PROJECT_DIR in .env (${PROJECT_DIR}) != cwd (${PROJECT_DIR_LOCAL})"
    warn "Cron jobs will use the .env value. Adjust if that's wrong."
fi

# ---------- 2. Prerequisites ----------
bold "--- Prerequisites ---"
command -v docker >/dev/null || die "docker not installed"
docker compose version >/dev/null 2>&1 || command -v docker-compose >/dev/null || die "docker compose plugin not installed"
command -v python3 >/dev/null || die "python3 not installed"
command -v crontab >/dev/null || die "cron not installed"

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJMIN="$(echo "${PY_VER}" | awk -F. '{print ($1*100)+$2}')"
if [ "${PY_MAJMIN}" -lt 310 ]; then
    die "Python 3.10+ required (found ${PY_VER})"
fi

# On Debian/Ubuntu, python3-venv is a separate package and is needed for
# `python3 -m venv`. Check it BEFORE the venv step so the user gets a clean
# message instead of cryptic ensurepip errors halfway through setup.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    die "python3-venv module missing — install with: sudo apt install -y python${PY_VER}-venv"
fi
if ! python3 -c "import venv" >/dev/null 2>&1; then
    die "python3-venv module missing — install with: sudo apt install -y python${PY_VER}-venv"
fi

# pip availability is bundled with ensurepip but worth confirming
if ! python3 -m pip --version >/dev/null 2>&1; then
    warn "system pip not available — venv will bootstrap its own, should be fine"
fi

echo "Python ${PY_VER} ✓"
echo "Docker $(docker --version | awk '{print $3}' | tr -d ',') ✓"

# ---------- 3. NAS mounts ----------
# Day-dirs (YYYYMMDD) live one level deep under each mount:
#   allsky:  ${NAS_ALLSKY_PATH}/images/YYYYMMDD/
#   thermal: ${NAS_THERMAL_PATH}/${NAS_THERMAL_UUID}/exposures/YYYYMMDD/
bold "--- NAS mount check ---"

# --- allsky ---
if [ ! -d "${NAS_ALLSKY_PATH}" ]; then
    die "NAS path not mounted: ${NAS_ALLSKY_PATH}  (mount via fstab/smb first)"
fi
ALLSKY_DAY_DIR="${NAS_ALLSKY_PATH}/images"
if [ ! -d "${ALLSKY_DAY_DIR}" ]; then
    warn "${ALLSKY_DAY_DIR} not found — expected layout: ${NAS_ALLSKY_PATH}/images/YYYYMMDD/"
    warn "If the SMB share is mounted at a deeper level, adjust NAS_ALLSKY_PATH to the parent of 'images/'"
else
    N=$(ls "${ALLSKY_DAY_DIR}" 2>/dev/null | grep -cE '^[0-9]{8}$' || true)
    [ "${N}" -gt 0 ] \
        && echo "${NAS_ALLSKY_PATH} ✓  (${N} day-dirs in images/)" \
        || die "${ALLSKY_DAY_DIR} contains 0 YYYYMMDD subdirs — wrong mount or empty share?"
fi

# --- thermal ---
if [ ! -d "${NAS_THERMAL_PATH}" ]; then
    die "NAS path not mounted: ${NAS_THERMAL_PATH}  (mount via fstab/smb first)"
fi
THERMAL_DAY_DIR="${NAS_THERMAL_PATH}/${NAS_THERMAL_UUID}/exposures"
if [ ! -d "${THERMAL_DAY_DIR}" ]; then
    warn "${THERMAL_DAY_DIR} not found — check NAS_THERMAL_UUID in deploy/.env"
    warn "Available UUIDs under ${NAS_THERMAL_PATH}:"
    ls "${NAS_THERMAL_PATH}" 2>/dev/null | sed 's/^/    /'
else
    N=$(ls "${THERMAL_DAY_DIR}" 2>/dev/null | grep -cE '^[0-9]{8}$' || true)
    [ "${N}" -gt 0 ] \
        && echo "${NAS_THERMAL_PATH} ✓  (${N} day-dirs in ${NAS_THERMAL_UUID}/exposures/)" \
        || die "${THERMAL_DAY_DIR} contains 0 YYYYMMDD subdirs — wrong UUID or empty share?"
fi

# ---------- 4. PG port reachability ----------
# Deep PG check (psycopg2 + actual query) happens after venv setup below,
# because psycopg2 lives in the venv we're about to create.
bold "--- PG port check ---"
if command -v nc >/dev/null; then
    nc -z "${PG_HOST}" "${PG_PORT}" \
        && echo "PG port ${PG_HOST}:${PG_PORT} reachable" \
        || die "PG port ${PG_HOST}:${PG_PORT} unreachable (is the db container up?)"
else
    warn "nc not installed — skipping port check; psycopg2 will verify below"
fi

# ---------- 5. Directories ----------
bold "--- Creating directories ---"
mkdir -p "${PROJECT_DIR}/labels" "${PROJECT_DIR}/goes_cache" "${LOG_DIR}"
echo "Directories created/verified"

# ---------- 6. Host venv (for cron jobs) ----------
bold "--- Host venv ---"
# Detect a previous failed venv attempt (directory exists but pip is missing)
# and rebuild from scratch, otherwise we'd try to invoke .venv/bin/pip below
# and crash with "No such file or directory".
if [ -d .venv ] && [ ! -x .venv/bin/pip ]; then
    warn "Found broken .venv (no pip) — removing and recreating"
    rm -rf .venv
fi
if [ ! -d .venv ]; then
    python3 -m venv .venv || die "venv creation failed even though python3-venv is installed — try: rm -rf .venv && python3 -m venv .venv"
    echo "Created .venv"
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet \
    "streamlit>=1.30,<2" "pandas>=2,<3" "numpy>=1.24,<3" "pillow>=10,<12" \
    "opencv-python>=4.8,<5" "psycopg2-binary>=2.9,<3" "netCDF4>=1.6,<2"
echo "venv ready: $(.venv/bin/python --version | awk '{print $2}')"

# ---------- 6b. Deep PG check (psycopg2 + actual query) ----------
bold "--- PG connection (psycopg2) ---"
.venv/bin/python - <<PY || die "PG connection failed — check credentials in deploy/.env"
import psycopg2, sys
try:
    conn = psycopg2.connect(host="${PG_HOST}", port=${PG_PORT}, dbname="${PG_DB}",
                            user="${PG_USER}", password="${PG_PASS}", connect_timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM captures")
    print(f"PG ✓  ({cur.fetchone()[0]} captures rows)")
    conn.close()
except Exception as e:
    print(f"PG ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ---------- 7. Make cron wrappers executable ----------
bold "--- Cron wrappers ---"
chmod +x deploy/cron/*.sh
echo "Marked deploy/cron/*.sh executable"

# ---------- 8. Build & start labeling tool container ----------
bold "--- Labeling tool container ---"
docker compose --env-file deploy/.env -f deploy/docker-compose.labeling.yml build
docker compose --env-file deploy/.env -f deploy/docker-compose.labeling.yml up -d
sleep 3
if curl -fsS "http://${STREAMLIT_HOST}:${STREAMLIT_PORT}/_stcore/health" >/dev/null 2>&1; then
    echo "Labeling UI healthy at http://${STREAMLIT_HOST}:${STREAMLIT_PORT}"
else
    warn "Labeling UI not responding yet — check: docker logs labeling-tool"
fi

# ---------- 9. Install crontab (append-or-update, preserves other jobs) ----------
bold "--- Crontab ---"
./deploy/install-crontab.sh
echo "Active jobs:"
crontab -l | grep -E '^[0-9*]' | sed 's/^/  /'

# ---------- 10. Done ----------
bold "=== Setup complete ==="
echo
echo "Labeling UI:    http://${STREAMLIT_HOST}:${STREAMLIT_PORT}"
echo "Logs:           ${LOG_DIR}"
echo "Tail logs:      tail -f ${LOG_DIR}/*.log"
echo "Container logs: docker logs -f labeling-tool"
echo "Restart tool:   docker compose -f deploy/docker-compose.labeling.yml restart"
echo "Update:         git pull && ./deploy/server-setup.sh"
