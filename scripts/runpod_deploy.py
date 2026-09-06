#!/usr/bin/env python3
"""Create or update the RunPod serverless endpoint for this image.

Usage:
  RUNPOD_API_KEY=... scripts/runpod_deploy.py \
      --image ghcr.io/lee101/pixal3dcog:sls --name pixal3d \
      --gpu-ids "ADA_24,AMPERE_24" --workers-max 3 --idle 120 \
      --env S3_BUCKET=simplexstatic --env S3_ENDPOINT_URL=https://... \
      [--endpoint-id existing] [--registry-auth-id id]

Prints the endpoint id. Re-running with --endpoint-id re-saves the template
(new image tag or env) and points the existing endpoint at it.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

GRAPHQL = "https://api.runpod.io/graphql"


def gql(api_key, query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        GRAPHQL + "?api_key=" + api_key,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key, "User-Agent": "pixal3dcog/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    if payload.get("errors"):
        raise SystemExit("runpod graphql error: " + json.dumps(payload["errors"])[:800])
    return payload["data"]


def save_template(api_key, name, image, env, disk_gb, registry_auth_id):
    inp = {
        "name": name,
        "imageName": image,
        "isServerless": True,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "dockerArgs": "",
        "env": [{"key": k, "value": v} for k, v in env.items()],
    }
    if registry_auth_id:
        inp["containerRegistryAuthId"] = registry_auth_id
    data = gql(api_key, "mutation saveTemplate($input: SaveTemplateInput) { saveTemplate(input: $input) { id } }", {"input": inp})
    return data["saveTemplate"]["id"]


def create_network_volume(api_key, name, size_gb, data_center_id):
    data = gql(api_key, "mutation createNetworkVolume($input: CreateNetworkVolumeInput!) { createNetworkVolume(input: $input) { id dataCenterId } }",
               {"input": {"name": name, "size": size_gb, "dataCenterId": data_center_id}})
    return data["createNetworkVolume"]["id"]


def save_endpoint(api_key, name, template_id, gpu_ids, workers_max, idle, endpoint_id=None, network_volume_id=None):
    inp = {
        "name": name,
        "templateId": template_id,
        "gpuIds": gpu_ids,
        "workersMin": 0,
        "workersMax": workers_max,
        "idleTimeout": idle,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "locations": None,
        "networkVolumeId": network_volume_id or None,
    }
    if endpoint_id:
        inp["id"] = endpoint_id
    data = gql(api_key, "mutation saveEndpoint($input: EndpointInput!) { saveEndpoint(input: $input) { id } }", {"input": inp})
    return data["saveEndpoint"]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default="pixal3d")
    ap.add_argument("--gpu-ids", default="ADA_24,AMPERE_24")
    ap.add_argument("--workers-max", type=int, default=3)
    ap.add_argument("--idle", type=int, default=120)
    ap.add_argument("--disk-gb", type=int, default=50)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--endpoint-id", default="")
    ap.add_argument("--registry-auth-id", default=os.environ.get("RUNPOD_REGISTRY_AUTH_ID", ""))
    ap.add_argument("--network-volume-id", default="", help="Existing network volume (mounted at /runpod-volume) that caches weights")
    ap.add_argument("--create-volume-gb", type=int, default=0, help="Create a network volume of this size first")
    ap.add_argument("--data-center", default="US-TX-3", help="Data center for --create-volume-gb")
    args = ap.parse_args()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        sys.exit("RUNPOD_API_KEY is required")
    env = {}
    for item in args.env:
        k, _, v = item.partition("=")
        if k:
            env[k] = v
    volume_id = args.network_volume_id
    if args.create_volume_gb and not volume_id:
        volume_id = create_network_volume(api_key, args.name + "-weights", args.create_volume_gb, args.data_center)
        print("created network volume", volume_id, file=sys.stderr)
    # Template names must be unique per account, so every deploy makes a new one.
    template_id = save_template(api_key, f"{args.name}-template-{int(time.time())}", args.image, env, args.disk_gb, args.registry_auth_id)
    endpoint_id = save_endpoint(api_key, args.name, template_id, args.gpu_ids, args.workers_max, args.idle, args.endpoint_id or None, volume_id or None)
    print(json.dumps({"template_id": template_id, "endpoint_id": endpoint_id, "network_volume_id": volume_id or None}))


if __name__ == "__main__":
    main()
