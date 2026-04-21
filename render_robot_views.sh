#!/bin/bash
# Save robot-perspective RGB/depth images per scene (render_robot_views.py).
# Run on EC2 DCV desktop: ./render_robot_views.sh
# Extra args pass through: ./render_robot_views.sh --scenes Rs_int --num-views 2
# Headless over SSH: ./render_robot_views.sh --headless
set -euo pipefail
ROOT="${RENDER_ROBOT_VIEWS_ROOT:-$HOME/OmniGibson_TakeHomeTest}"
if [[ -f "$ROOT/render_robot_views.py" ]]; then
  cd "$ROOT"
elif [[ -f "$HOME/render_robot_views.py" ]]; then
  cd "$HOME"
else
  echo "render_robot_views.py not found. Sync from your Mac: ./sync_to_ec2.sh" >&2
  exit 1
fi
source ~/miniconda/etc/profile.d/conda.sh
conda activate behavior
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
exec python render_robot_views.py --accept-license --output-dir output2/robot_views "$@"
