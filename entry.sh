#!/bin/sh
# RUNPOD_ENDPOINT_ID is only set on RunPod serverless workers; dedicated pods
# and local runs fall through to the normal cog HTTP server on :5000.
if [ -n "$RUNPOD_ENDPOINT_ID" ]; then
  python /src/rp_handler.py
  code=$?
  python /src/ship_log.py "$code"
  exit $code
fi
exec python -m cog.server.http
