#!/bin/bash
# First-person robot navigation videos (render_robot_videos.py).
# Run on EC2 DCV desktop: ./render_robot_videos.sh
# Extra args pass through: ./render_robot_videos.sh --scenes Rs_int --max-seconds 30
# Headless over SSH: ./render_robot_videos.sh --headless
set -euo pipefail
ROOT="${RENDER_ROBOT_VIDEOS_ROOT:-$HOME/OmniGibson_TakeHomeTest}"
if [[ -f "$ROOT/render_robot_videos.py" ]]; then
  cd "$ROOT"
elif [[ -f "$HOME/render_robot_videos.py" ]]; then
  cd "$HOME"
else
  echo "render_robot_videos.py not found. Sync from your Mac: ./sync_to_ec2.sh" >&2
  exit 1
fi
source ~/miniconda/etc/profile.d/conda.sh
conda activate behavior
# QuickTime-friendly post-process in render_robot_videos.py needs ffmpeg; conda env often omits it.
export PATH="/usr/bin:/usr/local/bin:${PATH}"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg not on PATH after conda — MP4s stay mp4v and may not open in QuickTime." >&2
  echo "       On Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg" >&2
  echo "       After pull on Mac: ./transcode_robot_videos_h264.sh" >&2
fi
export TMPDIR=/opt/dlami/nvme/tmp
export OMNIGIBSON_APPDATA_PATH=/opt/dlami/nvme/og-appdata
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$HOME/BEHAVIOR-1K/datasets}"
unset OMNIGIBSON_HEADLESS OMNIGIBSON_REMOTE_STREAMING
export OMNIGIBSON_DCV_COMPAT=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/dcv/og-demo.xauth}"
export OMNIGIBSON_DCV_WINDOW_WIDTH="${OMNIGIBSON_DCV_WINDOW_WIDTH:-1280}"
export OMNIGIBSON_DCV_WINDOW_HEIGHT="${OMNIGIBSON_DCV_WINDOW_HEIGHT:-800}"
exec python render_robot_videos.py --accept-license --output-dir output2/robot_videos \
  --scenes Beechwood_0_int,Rs_int,Benevolence_1_int "$@"
