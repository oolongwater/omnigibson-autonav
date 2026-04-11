#!/usr/bin/env bash
# Mirror this repo to EC2 so the instance matches your local OmniGibson_TakeHomeTest layout.
#
# From repo root on your Mac:
#   chmod +x sync_to_ec2.sh
#   ./sync_to_ec2.sh ubuntu@EC2_PUBLIC_IP
#   # Uses ./omnigibson-key.pem automatically if present (still excluded from rsync — never uploaded).
#   ./sync_to_ec2.sh -i ~/.ssh/other.pem ubuntu@EC2_PUBLIC_IP
#
# Optional:
#   REMOTE_DIR=MyTakeHome ./sync_to_ec2.sh ...   # default remote folder: OmniGibson_TakeHomeTest
#   ./sync_to_ec2.sh --skip-behavior ...         # omit BEHAVIOR-1K; symlink ~/BEHAVIOR-1K if it exists
#
# Or set defaults once (e.g. in ~/.zshrc):
#   export EC2_SYNC_TARGET=ubuntu@ec2-xxx.compute.amazonaws.com
#   export EC2_SSH_KEY=~/.ssh/your-key.pem   # optional if ssh(1) already knows the key
#
# Or create ./.ec2_sync_host (gitignored) with one line: USER@HOST
#   ubuntu@...  — Ubuntu / many GPU AMIs   |   ec2-user@...  — stock Amazon Linux
#
set -euo pipefail

SKIP_BEHAVIOR=0
SSH_KEY="${EC2_SSH_KEY:-}"
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
  echo "Usage: $0 [-i KEY.pem] [--skip-behavior] ubuntu@EC2_HOST" >&2
  echo "Or: export EC2_SYNC_TARGET=ubuntu@EC2_HOST  (and optionally EC2_SSH_KEY=~/.ssh/key.pem)" >&2
  echo "Or: echo 'ubuntu@YOUR_PUBLIC_DNS' > \"$HERE/.ec2_sync_host\"" >&2
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

# IdentitiesOnly avoids ssh-agent keys being tried first (can cause spurious failures on some servers).
SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && SSH_BASE=(
  ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
)
RSYNC_E="ssh -o StrictHostKeyChecking=accept-new"
[[ -n "$SSH_KEY" ]] && RSYNC_E="ssh -i ${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.cursor/'
  --exclude='omnigibson-key.pem'
  --exclude='*.pem'
  # Remote Kit/Omniverse cache; often root-owned or unreadable over rsync — not part of the repo source tree.
  --exclude='BEHAVIOR-1K/OmniGibson/appdata/'
)

if [[ "$SKIP_BEHAVIOR" -eq 1 ]]; then
  RSYNC_EXCLUDES+=(--exclude='BEHAVIOR-1K/')
fi

echo "Syncing: $HERE/ -> ${REMOTE}:~/${REMOTE_NAME}/"
rsync -avz --delete "${RSYNC_EXCLUDES[@]}" -e "$RSYNC_E" \
  "$HERE/" "${REMOTE}:~/${REMOTE_NAME}/"

# So DCV users see it next to ~/run_partB_dcv.sh when they ls ~
if [[ -f "$HERE/cycle_scenes.sh" ]]; then
  echo "Linking ~/cycle_scenes.sh -> ~/${REMOTE_NAME}/cycle_scenes.sh"
  "${SSH_BASE[@]}" "$REMOTE" "set -e
    if [[ -f \"\$HOME/${REMOTE_NAME}/cycle_scenes.sh\" ]]; then
      ln -sfn \"\$HOME/${REMOTE_NAME}/cycle_scenes.sh\" \"\$HOME/cycle_scenes.sh\"
      echo 'OK: ~/cycle_scenes.sh'
    fi"
fi

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

echo "Done. On EC2: ~/run_partB_dcv.sh and ~/cycle_scenes.sh (symlink to ~/${REMOTE_NAME}/cycle_scenes.sh)"
