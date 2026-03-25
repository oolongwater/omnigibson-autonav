#!/usr/bin/env bash
# Mirror this repo to EC2 so the instance matches your local OmniGibson_TakeHomeTest layout.
#
# From repo root on your Mac:
#   chmod +x sync_to_ec2.sh
#   ./sync_to_ec2.sh -i ~/.ssh/your-key.pem ubuntu@EC2_PUBLIC_IP
#
# Optional:
#   REMOTE_DIR=MyTakeHome ./sync_to_ec2.sh ...   # default remote folder: OmniGibson_TakeHomeTest
#   ./sync_to_ec2.sh --skip-behavior ...         # omit BEHAVIOR-1K; symlink ~/BEHAVIOR-1K if it exists
#
set -euo pipefail

SKIP_BEHAVIOR=0
SSH_KEY=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-behavior)
      SKIP_BEHAVIOR=1
      shift
      ;;
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

if [[ ${#ARGS[@]} -lt 1 ]]; then
  echo "Usage: $0 [-i KEY.pem] [--skip-behavior] ubuntu@EC2_HOST" >&2
  exit 1
fi

REMOTE="${ARGS[0]}"
REMOTE_NAME="${REMOTE_DIR:-OmniGibson_TakeHomeTest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && SSH_BASE=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
RSYNC_E="ssh -o StrictHostKeyChecking=accept-new"
[[ -n "$SSH_KEY" ]] && RSYNC_E="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"

RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.cursor/'
  --exclude='omnigibson-key.pem'
  --exclude='*.pem'
)

if [[ "$SKIP_BEHAVIOR" -eq 1 ]]; then
  RSYNC_EXCLUDES+=(--exclude='BEHAVIOR-1K/')
fi

echo "Syncing: $HERE/ -> ${REMOTE}:~/${REMOTE_NAME}/"
rsync -avz --delete "${RSYNC_EXCLUDES[@]}" -e "$RSYNC_E" \
  "$HERE/" "${REMOTE}:~/${REMOTE_NAME}/"

if [[ "$SKIP_BEHAVIOR" -eq 1 ]]; then
  echo "Ensuring BEHAVIOR-1K is available under ~/${REMOTE_NAME}..."
  "${SSH_BASE[@]}" "$REMOTE" "set -e
    if [[ -d \"\$HOME/BEHAVIOR-1K\" ]]; then
      ln -sfn \"\$HOME/BEHAVIOR-1K\" \"\$HOME/${REMOTE_NAME}/BEHAVIOR-1K\"
      echo 'Symlinked ~/BEHAVIOR-1K -> ~/${REMOTE_NAME}/BEHAVIOR-1K'
    elif [[ ! -d \"\$HOME/${REMOTE_NAME}/BEHAVIOR-1K\" ]]; then
      echo 'Warning: no ~/BEHAVIOR-1K on EC2. Run sync without --skip-behavior or clone BEHAVIOR-1K there.'
    fi"
fi

echo "Done. On EC2: install ~/run_partB_dcv.sh once (or sync includes it); then ~/run_partB_dcv.sh"
