#!/usr/bin/env bash
# Append (or update in place) allsky-cloud-analysis cron jobs WITHOUT
# clobbering any other cron jobs the user already has.
#
# Idempotent: re-running this script replaces our block in place rather than
# duplicating it.
#
# Run from the project root:
#   ./deploy/install-crontab.sh
#
# To remove the block later:
#   ./deploy/install-crontab.sh --remove

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

BEGIN_MARK="# === BEGIN allsky-cloud-analysis (managed by install-crontab.sh) ==="
END_MARK="# === END allsky-cloud-analysis ==="

# Load .env so we can substitute PROJECT_DIR + LOG_DIR
if [ -f deploy/.env ]; then
    # shellcheck source=/dev/null
    set -a; source deploy/.env; set +a
fi
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
LOG_DIR="${LOG_DIR:-/var/log/allsky-cloud-analysis}"

# Get current crontab (empty if none installed)
CURRENT="$(crontab -l 2>/dev/null || true)"

# Strip any existing managed block (matches BEGIN_MARK through END_MARK inclusive)
WITHOUT_OURS="$(echo "${CURRENT}" | awk -v b="${BEGIN_MARK}" -v e="${END_MARK}" '
    $0 == b { in_block = 1; next }
    $0 == e { in_block = 0; next }
    !in_block { print }
')"

if [ "${1:-}" = "--remove" ]; then
    echo "${WITHOUT_OURS}" | crontab -
    echo "Removed allsky-cloud-analysis cron block. Current crontab:"
    crontab -l 2>/dev/null || echo "(empty)"
    exit 0
fi

# Build our block from the template, substituting PROJECT_DIR_ROOT and LOG_DIR
OUR_BLOCK="$(sed \
    -e "s|^PROJECT_DIR_ROOT=.*|PROJECT_DIR_ROOT=${PROJECT_DIR}|" \
    -e "s|^LOG_DIR=.*|LOG_DIR=${LOG_DIR}|" \
    deploy/crontab.template)"

# Compose final crontab: user's existing (minus old block) + our block
NEW_CRONTAB="$(printf '%s\n\n%s\n%s\n%s\n' \
    "${WITHOUT_OURS}" \
    "${BEGIN_MARK}" \
    "${OUR_BLOCK}" \
    "${END_MARK}")"

# Strip leading blank lines if user had no prior crontab
NEW_CRONTAB="$(echo "${NEW_CRONTAB}" | awk 'NF || seen { print; seen=1 }')"

# Install
echo "${NEW_CRONTAB}" | crontab -

echo "=== Installed. Current crontab: ==="
crontab -l
echo
echo "Logs will appear in: ${LOG_DIR}/"
echo "To remove our block later:  ./deploy/install-crontab.sh --remove"
