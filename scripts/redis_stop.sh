#!/usr/bin/env bash
# Stop the per-project Redis for sensVeridian.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

REDIS_CLI="${REDIS_CLI:-/data3/ssharma8/py310/bin/redis-cli}"
PID_FILE="${PROJECT_ROOT}/cache/redis/redis.pid"
HOST="${SV_REDIS_HOST:-127.0.0.1}"
PORT="${SV_REDIS_PORT:-6379}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  PID="$(cat "${PID_FILE}")"
  if [[ -x "${REDIS_CLI}" ]]; then
    "${REDIS_CLI}" -h "${HOST}" -p "${PORT}" SHUTDOWN NOSAVE 2>/dev/null || true
  fi
  for i in {1..20}; do
    if ! kill -0 "${PID}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "redis-server stopped (pid ${PID})"
      exit 0
    fi
    sleep 0.25
  done
  echo "redis-server did not stop cleanly; sending SIGTERM"
  kill -TERM "${PID}" 2>/dev/null || true
  sleep 1
  if kill -0 "${PID}" 2>/dev/null; then
    echo "forcing SIGKILL"
    kill -KILL "${PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  exit 0
fi

echo "no running redis-server found (pid file missing or stale)"
rm -f "${PID_FILE}"
exit 0
