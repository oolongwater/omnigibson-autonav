#!/bin/bash
# Part B: 180 s autonomous navigation via DCV.
# Kept at repo root for easy discovery and scp to EC2.
# On EC2: keep a copy in ~/ (or symlink from repo). Prefers same layout as local: ~/OmniGibson_TakeHomeTest/
set -euo pipefail
PARTB_ROOT="${PARTB_ROOT:-$HOME/OmniGibson_TakeHomeTest}"
if [[ -f "$PARTB_ROOT/autonomous_nav_60s.py" ]]; then
  cd "$PARTB_ROOT"
elif [[ -f "$HOME/autonomous_nav_60s.py" ]]; then
  cd "$HOME"
else
  echo "autonomous_nav_60s.py not found. Sync from your Mac: ./sync_to_ec2.sh -i KEY ubuntu@HOST" >&2
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
exec python autonomous_nav_60s.py "$@"
