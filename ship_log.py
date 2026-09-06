"""Upload the worker log after the handler process exits (crash diagnostics).

entry.sh runs this with the handler's exit code; RunPod itself exposes no
worker logs, so a hard crash (137 OOM-kill, 139 segfault, 134 abort) would
otherwise leave nothing behind.
"""

import os
import socket
import sys

LOG_PATH = "/tmp/pixal3d-worker.log"

code = sys.argv[1] if len(sys.argv) > 1 else "?"
meaning = {"137": "SIGKILL (host OOM killer or RunPod kill)", "139": "segmentation fault", "134": "abort (CUDA/native assertion)"}.get(code, "")
with open(LOG_PATH, "a") as f:
    f.write(f"\n[worker] handler process exited with code {code} {meaning}\n")
try:
    import boto3

    bucket = os.environ.get("S3_BUCKET", "")
    key_id = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if bucket and key_id and secret:
        worker = os.environ.get("RUNPOD_POD_ID") or socket.gethostname()
        prefix = os.environ.get("S3_PREFIX", "pixal3d").strip("/")
        client = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None, aws_access_key_id=key_id,
                              aws_secret_access_key=secret, region_name=os.environ.get("S3_REGION", "auto"))
        with open(LOG_PATH, "rb") as f:
            client.put_object(Bucket=bucket, Key=f"{prefix}/logs/{worker}.log", Body=f.read()[-2_000_000:], ContentType="text/plain")
except Exception as exc:
    print(f"[ship_log] {exc}", file=sys.stderr)
