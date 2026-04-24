#!/usr/bin/env bash
# Start the per-project Redis for sensVeridian.
# - Binary comes from the py310 conda env (no system install).
# - All state lives under /data3/ssharma8 (never under /home/ssharma8).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

REDIS_BIN="${REDIS_BIN:-/data3/ssharma8/py310/bin/redis-server}"
REDIS_CLI="${REDIS_CLI:-/data3/ssharma8/py310/bin/redis-cli}"
REDIS_CONF="${PROJECT_ROOT}/cache/redis/redis.conf"
PID_FILE="${PROJECT_ROOT}/cache/redis/redis.pid"
HOST="${SV_REDIS_HOST:-127.0.0.1}"
PORT="${SV_REDIS_PORT:-6379}"

if [[ ! -x "${REDIS_BIN}" ]]; then
  echo "redis-server binary not found at ${REDIS_BIN}" >&2
  echo "install with: conda install -p /data3/ssharma8/py310 -c conda-forge --override-channels -y redis-server" >&2
  exit 1
fi

if [[ ! -f "${REDIS_CONF}" ]]; then
  echo "redis config not found at ${REDIS_CONF}" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "redis-server already running (pid $(cat "${PID_FILE}"))"
  exit 0
fi

if "${REDIS_CLI}" -h "${HOST}" -p "${PORT}" PING >/dev/null 2>&1; then
  echo "redis already responding on ${HOST}:${PORT} (external instance)"
  exit 0
fi

mkdir -p "${PROJECT_ROOT}/cache/redis/data"

echo "starting redis-server with ${REDIS_CONF}"
"${REDIS_BIN}" "${REDIS_CONF}"

for i in {1..20}; do
  if "${REDIS_CLI}" -h "${HOST}" -p "${PORT}" PING >/dev/null 2>&1; then
    echo "redis-server up on ${HOST}:${PORT} (pid $(cat "${PID_FILE}" 2>/dev/null || echo '?'))"
    exit 0
  fi
  sleep 0.25
done

echo "redis-server failed to respond within timeout; check ${PROJECT_ROOT}/cache/redis/redis.log" >&2
exit 1
