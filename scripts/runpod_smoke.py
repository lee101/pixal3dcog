#!/usr/bin/env python3
"""Submit one job to the RunPod endpoint and print timings.

  RUNPOD_API_KEY=... scripts/runpod_smoke.py --endpoint <id> --image-url https://... [--resolution 1536] [--texture-size 2048]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

BASE = "https://api.runpod.ai/v2"


def call(api_key, method, url, body=None, retries=8):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key, "User-Agent": "pixal3dcog/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # DNS blips and 5xx/409 while the endpoint updates are transient.
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1}: {exc}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--image-url", required=True)
    ap.add_argument("--resolution", type=int, default=1536)
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--decimation-target", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        sys.exit("RUNPOD_API_KEY is required")
    payload = {"input": {"image_url": args.image_url, "resolution": args.resolution, "texture_size": args.texture_size,
                         "decimation_target": args.decimation_target, "seed": args.seed},
               "policy": {"executionTimeout": 1800000}}
    t0 = time.time()
    job = call(api_key, "POST", f"{BASE}/{args.endpoint}/run", payload)
    job_id = job["id"]
    print("job", job_id, job.get("status"), flush=True)
    last = None
    while True:
        time.sleep(3)
        st = call(api_key, "GET", f"{BASE}/{args.endpoint}/status/{job_id}")
        if st.get("status") != last:
            print(f"{time.time() - t0:6.1f}s {st.get('status')}", flush=True)
            last = st.get("status")
        if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            out = st.get("output") or {}
            if isinstance(out, dict) and "glb_base64" in out:
                out = {**out, "glb_base64": f"<{len(out['glb_base64'])} chars>"}
            print(json.dumps({"status": st.get("status"), "error": st.get("error"), "delayTime_ms": st.get("delayTime"),
                              "executionTime_ms": st.get("executionTime"), "wall_s": round(time.time() - t0, 1), "output": out}, indent=1))
            sys.exit(0 if st.get("status") == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
