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
echo "Python ${PY_VER} ✓"
echo "Docker $(docker --version | awk '{print $3}' | tr -d ',') ✓"

# ---------- 3. NAS mounts ----------
bold "--- NAS mount check ---"
for path in "${NAS_ALLSKY_PATH}" "${NAS_THERMAL_PATH}"; do
    if [ ! -d "${path}" ]; then
        die "NAS path not mounted: ${path}  (mount it via fstab/smb before re-running)"
    fi
    N_DAYS=$(ls "${path}" 2>/dev/null | grep -cE '^[0-9]{8}' || true)
    echo "${path} ✓  (${N_DAYS} day-dirs visible)"
done
if [ ! -d "${NAS_THERMAL_PATH}/${NAS_THERMAL_UUID}" ]; then
    warn "NAS_THERMAL_UUID dir not found inside ${NAS_THERMAL_PATH} — make sure NAS_THERMAL_UUID is correct"
fi

# ---------- 4. PG connection ----------
bold "--- PG connection check ---"
python3 - <<PY
import sys
try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 not installed yet — that's ok, venv setup happens below")
try:
    conn = psycopg2.connect(host="${PG_HOST}", port=${PG_PORT}, dbname="${PG_DB}",
                            user="${PG_USER}", password="${PG_PASS}", connect_timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM captures")
    print(f"PG ✓  ({cur.fetchone()[0]} captures rows)")
    conn.close()
except Exception as e:
    sys.exit(f"PG connection failed: {e}")
PY

# ---------- 5. Directories ----------
bold "--- Creating directories ---"
mkdir -p "${PROJECT_DIR}/labels" "${PROJECT_DIR}/goes_cache" "${LOG_DIR}"
echo "Directories created/verified"

# ---------- 6. Host venv (for cron jobs) ----------
bold "--- Host venv ---"
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "Created .venv"
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet \
    "streamlit>=1.30,<2" "pandas>=2,<3" "numpy>=1.24,<3" "pillow>=10,<12" \
    "opencv-python>=4.8,<5" "psycopg2-binary>=2.9,<3" "netCDF4>=1.6,<2"
echo "venv ready: $(.venv/bin/python --version | awk '{print $2}')"

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

# ---------- 9. Install crontab ----------
bold "--- Crontab ---"
# Substitute PROJECT_DIR_ROOT in the template, install
CRON_TMP="$(mktemp)"
sed "s|^PROJECT_DIR_ROOT=.*|PROJECT_DIR_ROOT=${PROJECT_DIR}|" \
    deploy/crontab.template > "${CRON_TMP}"
crontab "${CRON_TMP}"
rm -f "${CRON_TMP}"
echo "Crontab installed. Active jobs:"
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
