#!/bin/bash
# Cycle 10 hardcoded BEHAVIOR-1K scenes in OmniGibson (~5 s sim each by default).
# Run from DCV desktop on EC2: ./cycle_scenes.sh   or   ./cycle_scenes.sh --skip-download
set -euo pipefail
ROOT="${CYCLE_SCENES_ROOT:-$HOME/OmniGibson_TakeHomeTest}"
if [[ -f "$ROOT/cycle_scenes.py" ]]; then
  cd "$ROOT"
elif [[ -f "$HOME/cycle_scenes.py" ]]; then
  cd "$HOME"
else
  echo "cycle_scenes.py not found. Sync from your Mac: ./sync_to_ec2.sh" >&2
  exit 1
fi
source ~/miniconda/etc/profile.d/conda.sh
conda activate behavior
export TMPDIR=/opt/dlami/nvme/tmp
export OMNIGIBSON_APPDATA_PATH=/opt/dlami/nvme/og-appdata
unset OMNIGIBSON_HEADLESS OMNIGIBSON_REMOTE_STREAMING
export OMNIGIBSON_DCV_COMPAT=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/dcv/og-demo.xauth}"
export OMNIGIBSON_DCV_WINDOW_WIDTH="${OMNIGIBSON_DCV_WINDOW_WIDTH:-1280}"
export OMNIGIBSON_DCV_WINDOW_HEIGHT="${OMNIGIBSON_DCV_WINDOW_HEIGHT:-800}"
exec python cycle_scenes.py --accept-license "$@"
