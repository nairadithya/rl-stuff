#!/usr/bin/env bash
set -euo pipefail

if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "RUNPOD_POD_ID or RUNPOD_API_KEY not set — not a RunPod environment, skipping shutdown."
  exit 0
fi

echo "Shutting down RunPod pod $RUNPOD_POD_ID ..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  "https://api.runpod.ai/v2/pods/${RUNPOD_POD_ID}/stop" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)

echo "HTTP $HTTP_CODE: $BODY"
[ "$HTTP_CODE" = "200" ] && echo "Stop request accepted." || echo "Warning: unexpected response."
