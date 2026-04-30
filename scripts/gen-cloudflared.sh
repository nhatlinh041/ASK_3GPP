#!/usr/bin/env bash
# Sinh cloudflared-demo.yml từ template + .env (KHÔNG commit file output).
set -e

# Đi tới repo root
cd "$(dirname "$0")/.."

# Load .env
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
else
  echo "✗ .env not found at $(pwd)/.env" >&2
  exit 1
fi

# Kiểm tra bắt buộc
: "${CLOUDFLARED_TUNNEL_ID:?CLOUDFLARED_TUNNEL_ID chưa set trong .env}"
: "${CLOUDFLARED_CREDENTIALS_FILE:?CLOUDFLARED_CREDENTIALS_FILE chưa set trong .env}"
: "${CLOUDFLARED_HOSTNAME:?CLOUDFLARED_HOSTNAME chưa set trong .env}"

TEMPLATE="cloudflared-demo.template.yml"
OUTPUT="cloudflared-demo.yml"

if [ ! -f "$TEMPLATE" ]; then
  echo "✗ Template not found: $TEMPLATE" >&2
  exit 1
fi

# Chỉ substitute đúng 3 biến whitelist, không đụng ${} khác trong YAML
envsubst '${CLOUDFLARED_TUNNEL_ID} ${CLOUDFLARED_CREDENTIALS_FILE} ${CLOUDFLARED_HOSTNAME}' \
  < "$TEMPLATE" > "$OUTPUT"

echo "✓ Wrote $OUTPUT"
