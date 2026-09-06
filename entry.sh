#!/bin/sh
# RUNPOD_ENDPOINT_ID is only set on RunPod serverless workers; dedicated pods
# and local runs fall through to the normal cog HTTP server on :5000.
# The handler's stdout/stderr are tee'd at the fd level so native libraries'
# messages (CUDA, triton, flex_gemm) land in the shipped worker log too.
if [ -n "$RUNPOD_ENDPOINT_ID" ]; then
  (python -u /src/rp_handler.py 2>&1; echo "$?" > /tmp/pixal3d-worker.exit) | tee -a /tmp/pixal3d-worker.log
  code=$(cat /tmp/pixal3d-worker.exit 2>/dev/null || echo 1)
  python /src/ship_log.py "$code"
  exit "$code"
fi
exec python -m cog.server.http
