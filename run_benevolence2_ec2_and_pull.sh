#!/usr/bin/env bash
# Sync repo, run autonomous_nav_benevolence2.py --record (default SIM_SECONDS in script, e.g. 180s) on EC2, pull FPV MP4 + diagnostics locally.
# Requires SSH access to the same host as sync_to_ec2.sh (agent key or -i / omnigibson-key.pem).
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
  echo "Usage: $0 [-i KEY.pem] [ubuntu@EC2_HOST]" >&2
  echo "Or: EC2_SYNC_TARGET / .ec2_sync_host (see sync_to_ec2.sh)" >&2
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

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && SSH_BASE=(ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)

RSYNC_E="ssh -o StrictHostKeyChecking=accept-new"
[[ -n "$SSH_KEY" ]] && RSYNC_E="ssh -i ${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

echo "==> Sync (skip BEHAVIOR-1K)"
if [[ -n "$SSH_KEY" ]]; then
  EC2_SSH_KEY="$SSH_KEY" "$HERE/sync_to_ec2.sh" --skip-behavior "$REMOTE"
else
  "$HERE/sync_to_ec2.sh" --skip-behavior "$REMOTE"
fi

OUT_NAME="autonomous_nav_benevolence2_fpv.mp4"
REMOTE_MP4="~/${REMOTE_NAME}/output/${OUT_NAME}"
LOCAL_OUT="$HERE/output/${OUT_NAME}"

echo "==> Run on EC2 (this may take many minutes)"
"${SSH_BASE[@]}" -tt "$REMOTE" bash -s <<REMOTE
set -euo pipefail
cd "\$HOME/${REMOTE_NAME}"
mkdir -p output
source ~/miniconda/etc/profile.d/conda.sh
conda activate behavior
export TMPDIR=/opt/dlami/nvme/tmp
export OMNIGIBSON_APPDATA_PATH=/opt/dlami/nvme/og-appdata
unset OMNIGIBSON_HEADLESS OMNIGIBSON_REMOTE_STREAMING
export OMNIGIBSON_DCV_COMPAT=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1
export DISPLAY="\${DISPLAY:-:0}"
export XAUTHORITY="\${XAUTHORITY:-/run/user/\$(id -u)/dcv/og-demo.xauth}"
exec python3 autonomous_nav_benevolence2.py --record --no-teleop-camera --once
REMOTE

echo "==> Pull ${OUT_NAME} and diagnostics"
mkdir -p "$HERE/output"
REMOTE_NAV="~/${REMOTE_NAME}/output/autonomous_nav_benevolence2_nav.csv"
REMOTE_EVT="~/${REMOTE_NAME}/output/autonomous_nav_benevolence2_events.log"
REMOTE_SUM="~/${REMOTE_NAME}/output/autonomous_nav_benevolence2_summary.json"
rsync -avz -e "$RSYNC_E" \
  "${REMOTE}:${REMOTE_MP4}" \
  "${REMOTE}:${REMOTE_NAV}" \
  "${REMOTE}:${REMOTE_EVT}" \
  "${REMOTE}:${REMOTE_SUM}" \
  "$HERE/output/"
ls -la "$LOCAL_OUT" "$HERE/output/autonomous_nav_benevolence2_nav.csv" \
  "$HERE/output/autonomous_nav_benevolence2_events.log" \
  "$HERE/output/autonomous_nav_benevolence2_summary.json" 2>/dev/null || true

# Re-render the floor-plan path overlay locally now that we have the actual nav CSV.
# Uses the cached floor_trav_0.png (pulled once into .cache/Benevolence_2_int/).
if [[ -f "$HERE/.cache/Benevolence_2_int/floor_trav_0.png" ]]; then
  echo "==> Re-render output/benevolence2_marked_path.png with actual trajectory"
  ( cd "$HERE" && python3 plot_marked_waypoints_path_benevolence2.py ) || \
    echo "[WARN] overlay re-render failed (continuing)"
else
  echo "[INFO] no cached floor_trav_0.png; skipping overlay re-render. Run:" >&2
  echo "  rsync -avz -e \"$RSYNC_E\" ${REMOTE}:~/${REMOTE_NAME}/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/Benevolence_2_int/layout/floor_trav_0.png $HERE/.cache/Benevolence_2_int/" >&2
fi

# EC2 often has no ffmpeg — raw file is mp4v and won't play in QuickTime / Cursor preview.
# Re-encode locally to H.264 + silent AAC + min 480px side (same idea as transcode_robot_videos_h264.sh).
H264_OUT="$HERE/output/autonomous_nav_benevolence2_fpv_h264.mp4"
if [[ -f "$LOCAL_OUT" ]] && command -v ffmpeg >/dev/null 2>&1; then
  echo "==> Transcode for preview -> ${H264_OUT##*/}"
  USE_VTB=0
  if [[ "$(uname -s)" == Darwin ]] && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'h264_videotoolbox'; then
    USE_VTB=1
  fi
  W="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$LOCAL_OUT" 2>/dev/null || echo 0)"
  H="$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$LOCAL_OUT" 2>/dev/null || echo 0)"
  VF=()
  if [[ "${W:-0}" =~ ^[0-9]+$ && "${H:-0}" =~ ^[0-9]+$ ]]; then
    minwh="$W"
    [[ "$H" -lt "$minwh" ]] && minwh="$H"
    if [[ "$minwh" -lt 480 ]]; then
      if [[ "$W" -ge "$H" ]]; then
        VF=( -vf "scale=480:-2:flags=neighbor" )
      else
        VF=( -vf "scale=-2:480:flags=neighbor" )
      fi
    fi
  fi
  tmp="${H264_OUT%.mp4}.tmp.mp4"
  if [[ "$USE_VTB" -eq 1 ]]; then
    ffmpeg -nostdin -y -hide_banner -loglevel error \
      -f lavfi -i anullsrc=r=48000:cl=mono \
      -i "$LOCAL_OUT" -shortest \
      -map 1:v:0 -map 0:a:0 \
      "${VF[@]}" \
      -c:v h264_videotoolbox -profile:v baseline -b:v 4M -allow_sw 1 -pix_fmt yuv420p \
      -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
      -movflags +faststart \
      -c:a aac -b:a 48k -ac 1 \
      "$tmp"
  else
    ffmpeg -nostdin -y -hide_banner -loglevel error \
      -f lavfi -i anullsrc=r=48000:cl=mono \
      -i "$LOCAL_OUT" -shortest \
      -map 1:v:0 -map 0:a:0 \
      "${VF[@]}" \
      -c:v libx264 -profile:v baseline -level 3.0 -crf 18 -pix_fmt yuv420p \
      -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
      -movflags +faststart \
      -c:a aac -b:a 48k -ac 1 \
      "$tmp"
  fi
  mv "$tmp" "$H264_OUT"
  if [[ "$(uname -s)" == Darwin ]]; then
    xattr -c "$H264_OUT" 2>/dev/null </dev/null || true
  fi
  ls -la "$H264_OUT"
  echo "Preview this file in Finder / QuickTime / Cursor: $H264_OUT"
elif [[ -f "$LOCAL_OUT" ]]; then
  echo "[WARN] ffmpeg not in PATH; install ffmpeg (brew install ffmpeg) then run:" >&2
  echo "  ffmpeg -y -i \"$LOCAL_OUT\" -c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart -an \"$H264_OUT\"" >&2
fi

echo "Done: $LOCAL_OUT (+ nav CSV, events log, summary JSON)"
