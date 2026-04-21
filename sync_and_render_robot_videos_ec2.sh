#!/usr/bin/env bash
# Sync this repo to EC2 (see sync_to_ec2.sh) and start headless robot video render in background.
# Run from repo root when your instance is running: ./sync_and_render_robot_videos_ec2.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
./sync_to_ec2.sh --skip-behavior
HOST="$(
  grep -v '^[[:space:]]*#' "$HERE/.ec2_sync_host" | grep -v '^[[:space:]]*$' | head -1 |
    sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
)"
if [[ -z "$HOST" ]]; then
  echo "No host in .ec2_sync_host or EC2_SYNC_TARGET" >&2
  exit 1
fi
SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
if [[ -f "$HERE/omnigibson-key.pem" ]]; then
  SSH_BASE=(ssh -i "$HERE/omnigibson-key.pem" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
fi
REMOTE_DIR="${REMOTE_DIR:-OmniGibson_TakeHomeTest}"
"${SSH_BASE[@]}" "$HOST" "set -e
cd \"\$HOME/${REMOTE_DIR}\"
mkdir -p output2/robot_videos
pkill -f 'python.*render_robot_videos\\.py' 2>/dev/null || true
sleep 1
nohup ./render_robot_videos.sh --headless >> output2/robot_videos/ec2_render.nohup.out 2>&1 &
echo \"Started render (remote log: ~/${REMOTE_DIR}/output2/robot_videos/ec2_render.nohup.out)\"
echo \"Tail on EC2: tail -f ~/${REMOTE_DIR}/output2/robot_videos/ec2_render.nohup.out\""
