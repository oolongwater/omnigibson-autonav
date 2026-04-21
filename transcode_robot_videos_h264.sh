#!/usr/bin/env bash
# Re-encode fpv_*.mp4 from OpenCV (mp4v) to H.264 Baseline for QuickTime / Safari.
# Run after pull: ./transcode_robot_videos_h264.sh [dir]
# Default dir: output2/robot_videos
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${1:-$ROOT/output2/robot_videos}"
if ! command -v ffmpeg >/dev/null; then
  echo "Install ffmpeg (e.g. brew install ffmpeg)" >&2
  exit 1
fi
# Apple decoders often behave best with VideoToolbox-encoded H.264 on macOS.
USE_VTB=0
if [[ "$(uname -s)" == Darwin ]] && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'h264_videotoolbox'; then
  USE_VTB=1
fi
# macOS Bash 3.2 — avoid mapfile and <<< multi-line edge cases; use process substitution.
any=0
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  any=1
  tmp="${f%.mp4}.apple_h264.tmp.mp4"
  W="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$f" 2>/dev/null || echo 0)"
  H="$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$f" 2>/dev/null || echo 0)"
  VF=()
  # Tiny FPV (e.g. 256x256) often fails in Cursor preview / some QuickTime paths; upscale to min side 480.
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
  # ffmpeg reads stdin by default; would consume the find|while pipe.
  # Silent AAC + BT.709 tags: some QuickTime / Finder paths reject video-only or untagged H.264.
  if [[ "$USE_VTB" -eq 1 ]]; then
    ffmpeg -nostdin -y -hide_banner -loglevel error \
      -f lavfi -i anullsrc=r=48000:cl=mono \
      -i "$f" \
      -shortest \
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
      -i "$f" \
      -shortest \
      -map 1:v:0 -map 0:a:0 \
      "${VF[@]}" \
      -c:v libx264 -profile:v baseline -level 3.0 -crf 18 -pix_fmt yuv420p \
      -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
      -movflags +faststart \
      -c:a aac -b:a 48k -ac 1 \
      "$tmp"
  fi
  mv "$tmp" "$f"
  if [[ "$(uname -s)" == Darwin ]]; then
    # xattr may read stdin; do not steal from the find|while pipe.
    xattr -c "$f" 2>/dev/null </dev/null || true
  fi
  echo "OK: $f"
done < <(find "$DIR" -name 'fpv_*.mp4' -type f 2>/dev/null | sort)
if [[ "$any" -eq 0 ]]; then
  echo "No fpv_*.mp4 under $DIR" >&2
  exit 1
fi
