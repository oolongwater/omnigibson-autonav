#!/usr/bin/env bash
# Forward Mac localhost:8443 -> EC2 DCV (127.0.0.1:8443).
# Leave this running in a terminal, then open DCV Viewer to https://localhost:8443
#
# Usage:
#   chmod +x dcv_tunnel.sh
#   ./dcv_tunnel.sh -i omnigibson-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
#
set -euo pipefail
KEY=""
REMOTE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i)
      KEY="$2"
      shift 2
      ;;
    *)
      REMOTE="$1"
      shift
      ;;
  esac
done

if [[ -z "$KEY" || -z "$REMOTE" ]]; then
  echo "Usage: $0 -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>" >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh not found in PATH" >&2
  exit 1
fi

echo "Tunnel: localhost:8443 -> ${REMOTE}:8443 (leave this window open)"
echo "Then DCV Viewer -> https://localhost:8443  user: ubuntu"
exec ssh -N \
  -i "$KEY" \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -L 8443:127.0.0.1:8443 \
  "$REMOTE"
