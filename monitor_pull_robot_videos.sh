#!/usr/bin/env bash
# Poll EC2 for render_robot_videos.py; every 15 minutes pull output2/robot_videos and log status.
# Run from repo root: ./monitor_pull_robot_videos.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
INTERVAL="${MONITOR_INTERVAL_SEC:-900}"
LOG="${ROBOT_VIDEOS_MONITOR_LOG:-$HERE/output2/robot_videos/_monitor.log}"
mkdir -p "$(dirname "$LOG")"

HOST=""
if [[ -f "$HERE/.ec2_sync_host" ]]; then
  HOST="$(
    grep -v '^[[:space:]]*#' "$HERE/.ec2_sync_host" | grep -v '^[[:space:]]*$' | head -1 |
      sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
  )"
fi
[[ -n "${EC2_SYNC_TARGET:-}" ]] && HOST="$EC2_SYNC_TARGET"
if [[ -z "$HOST" ]]; then
  echo "Set .ec2_sync_host or EC2_SYNC_TARGET" >&2
  exit 1
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
if [[ -f "$HERE/omnigibson-key.pem" ]]; then
  SSH_BASE=(ssh -i "$HERE/omnigibson-key.pem" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
fi

remote_pids() {
  # Match only the Python worker (not bash nohup wrappers).
  "${SSH_BASE[@]}" -o BatchMode=yes "$HOST" "pgrep -f 'python.*render_robot_videos\\.py' || true" 2>/dev/null | tr '\n' ' '
}

log_line() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

log_line "Monitor start; host=$HOST interval=${INTERVAL}s"

while true; do
  pids="$(remote_pids | xargs)"
  if [[ -z "${pids// /}" ]]; then
    log_line "No render_robot_videos.py on EC2; final pull and exit."
    ./pull_robot_videos.sh >>"$LOG" 2>&1 || log_line "pull_robot_videos.sh failed (exit $?)"
    log_line "Done."
    exit 0
  fi
  log_line "Render still running (PIDs: $pids); pulling partial outputs..."
  ./pull_robot_videos.sh >>"$LOG" 2>&1 || log_line "pull_robot_videos.sh failed (exit $?)"
  log_line "Sleeping ${INTERVAL}s until next check."
  sleep "$INTERVAL"
done
