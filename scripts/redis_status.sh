#!/usr/bin/env bash
# Report the status of the per-project Redis for sensVeridian.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

REDIS_CLI="${REDIS_CLI:-/data3/ssharma8/py310/bin/redis-cli}"
PID_FILE="${PROJECT_ROOT}/cache/redis/redis.pid"
HOST="${SV_REDIS_HOST:-127.0.0.1}"
PORT="${SV_REDIS_PORT:-6379}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "pid file: ${PID_FILE} (pid $(cat "${PID_FILE}"))"
fi

if [[ -x "${REDIS_CLI}" ]] && "${REDIS_CLI}" -h "${HOST}" -p "${PORT}" PING >/dev/null 2>&1; then
  echo "state: running on ${HOST}:${PORT}"
  "${REDIS_CLI}" -h "${HOST}" -p "${PORT}" INFO server 2>/dev/null | \
    grep -E "^(redis_version|process_id|uptime_in_seconds|tcp_port):" || true
  echo "keys(face:registered:*): $("${REDIS_CLI}" -h "${HOST}" -p "${PORT}" --scan --pattern 'face:registered:*' 2>/dev/null | wc -l)"
  exit 0
fi

echo "state: not running on ${HOST}:${PORT}"
exit 1
