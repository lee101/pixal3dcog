#!/usr/bin/env bash
# Build both image tags, push them, save a new RunPod template and point the
# endpoint at it. Runs on the build box (see README: a box with fast upload).
#   DOCKER_HOST=unix:///var/run/docker-build.sock RUNPOD_API_KEY=... \
#   RUNPOD_REGISTRY_AUTH_ID=... S3_ENDPOINT_URL=... S3_ACCESS_KEY_ID=... S3_SECRET_ACCESS_KEY=... \
#   scripts/build_push_deploy.sh <endpoint-id>
set -euo pipefail
ENDPOINT_ID="${1:?endpoint id}"
IMAGE="${IMAGE:-ghcr.io/lee101/pixal3dcog}"
cd "$(dirname "$0")/.."
DOCKER_BUILDKIT=1 docker build --network host -f Dockerfile.cog -t "$IMAGE:latest" .
docker build --network host -f Dockerfile.sls -t "$IMAGE:sls" .
docker push "$IMAGE:latest" | tail -1
docker push "$IMAGE:sls" | tail -1
OUT=$(python3 scripts/runpod_deploy.py --endpoint-id "$ENDPOINT_ID" --image "$IMAGE:sls" --name pixal3d \
  --gpu-ids "${GPU_IDS:-ADA_24,AMPERE_24}" --workers-max "${WORKERS_MAX:-2}" --idle "${IDLE_SECONDS:-120}" --disk-gb "${DISK_GB:-80}" \
  --env S3_BUCKET="${S3_BUCKET:-simplexstatic}" --env S3_ENDPOINT_URL="$S3_ENDPOINT_URL" \
  --env S3_ACCESS_KEY_ID="$S3_ACCESS_KEY_ID" --env S3_SECRET_ACCESS_KEY="$S3_SECRET_ACCESS_KEY" \
  --env S3_PUBLIC_BASE_URL="${S3_PUBLIC_BASE_URL:-https://simplexstatic.simplexgen.com}" \
  --env S3_PREFIX="${S3_PREFIX:-static/generated-3d/pixal3d}" --env S3_REGION=auto \
  ${NETWORK_VOLUME_ID:+--network-volume-id "$NETWORK_VOLUME_ID"})
echo "$OUT"
TEMPLATE_ID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['template_id'])" "$OUT")
# saveEndpoint does not always switch the template; the REST endpoint does.
curl -sf -X PATCH "https://rest.runpod.io/v1/endpoints/$ENDPOINT_ID" -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: pixal3dcog/1.0" -d "{\"templateId\":\"$TEMPLATE_ID\"}" >/dev/null
echo "endpoint $ENDPOINT_ID now on template $TEMPLATE_ID"
