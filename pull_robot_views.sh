#!/usr/bin/env bash
# Pull output2/robot_views/ from EC2 into this repo (reverse of sync_to_ec2 for images only).
#
# From repo root on your Mac:
#   chmod +x pull_robot_views.sh
#   ./pull_robot_views.sh -i ~/.ssh/key.pem ubuntu@EC2_HOST
#
# Same host resolution as sync_to_ec2.sh:
#   export EC2_SYNC_TARGET=ubuntu@...
#   export EC2_SSH_KEY=~/.ssh/key.pem
#   echo 'ubuntu@HOST' > .ec2_sync_host
#
# Optional: REMOTE_DIR=MyTakeHome ./pull_robot_views.sh ...
#
set -euo pipefail

SSH_KEY="${EC2_SSH_KEY:-}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i)
      SSH_KEY="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ${#ARGS[@]} -lt 1 && -n "${EC2_SYNC_TARGET:-}" ]]; then
  ARGS+=("$EC2_SYNC_TARGET")
fi

if [[ ${#ARGS[@]} -lt 1 && -f "$HERE/.ec2_sync_host" ]]; then
  _host="$(
    grep -v '^[[:space:]]*#' "$HERE/.ec2_sync_host" | grep -v '^[[:space:]]*$' | head -1 |
      sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
  )"
  [[ -n "$_host" ]] && ARGS+=("$_host")
fi

if [[ ${#ARGS[@]} -lt 1 ]]; then
  echo "Usage: $0 [-i KEY.pem] ubuntu@EC2_HOST" >&2
  echo "Or: export EC2_SYNC_TARGET=ubuntu@EC2_HOST" >&2
  echo "Or: .ec2_sync_host with one line USER@HOST" >&2
  exit 1
fi

REMOTE="${ARGS[0]}"
REMOTE_NAME="${REMOTE_DIR:-OmniGibson_TakeHomeTest}"

if [[ -z "$SSH_KEY" && -f "$HERE/omnigibson-key.pem" ]]; then
  SSH_KEY="$HERE/omnigibson-key.pem"
fi

if [[ -n "$SSH_KEY" && "$SSH_KEY" == ~* ]]; then
  SSH_KEY="${SSH_KEY/#\~/$HOME}"
fi

RSYNC_E="ssh -o StrictHostKeyChecking=accept-new"
[[ -n "$SSH_KEY" ]] && RSYNC_E="ssh -i ${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

REMOTE_PATH="${REMOTE}:~/${REMOTE_NAME}/output2/robot_views/"
LOCAL_PATH="$HERE/output2/robot_views/"

mkdir -p "$LOCAL_PATH"

echo "Pulling: ${REMOTE_PATH} -> ${LOCAL_PATH}"
rsync -avz -e "$RSYNC_E" "$REMOTE_PATH" "$LOCAL_PATH"
echo "Done."
