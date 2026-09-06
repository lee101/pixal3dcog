"""RunPod serverless handler. Calls the cog Predictor in-process (no HTTP hop).

Input: {"image_url": ..., or "image_base64": ..., plus any predict() kwargs}
Output: {"glb_url": str} when S3 upload is configured, else
        {"glb_base64": str, "file_name": "model.glb"}, plus seed/resolution/timings.

The predictor loads lazily inside the first job so the worker registers with
RunPod immediately; submit jobs with a generous executionTimeout (the first
one on a fresh worker also fetches ~26GB of weights unless a network volume
already holds them). Worker stdout/stderr is mirrored to
S3_PREFIX/logs/<worker>.log when S3 is configured, since RunPod exposes no
worker logs through its API.
"""

import base64
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.request

import runpod

LOG_PATH = "/tmp/pixal3d-worker.log"
_predictor = None
_predictor_lock = threading.Lock()

PREDICT_FIELDS = {
    "resolution", "seed", "texture_size", "decimation_target", "remesh",
    "ss_guidance_strength", "ss_guidance_rescale", "ss_sampling_steps", "ss_rescale_t",
    "shape_slat_guidance_strength", "shape_slat_guidance_rescale", "shape_slat_sampling_steps", "shape_slat_rescale_t",
    "tex_slat_guidance_strength", "tex_slat_guidance_rescale", "tex_slat_sampling_steps", "tex_slat_rescale_t",
    "mesh_scale", "max_num_tokens", "fov", "upload",
}


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


sys.stdout = _Tee(sys.stdout, LOG_PATH)
sys.stderr = _Tee(sys.stderr, LOG_PATH)


def _log_key():
    worker = os.environ.get("RUNPOD_POD_ID") or socket.gethostname()
    prefix = os.environ.get("S3_PREFIX", "pixal3d").strip().strip("/")
    return f"{prefix}/logs/{worker}.log" if prefix else f"logs/{worker}.log"


def _ship_log():
    bucket = os.environ.get("S3_BUCKET", "").strip()
    key_id = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not (bucket and key_id and secret):
        return
    try:
        import boto3

        client = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
                              aws_access_key_id=key_id, aws_secret_access_key=secret,
                              region_name=os.environ.get("S3_REGION", "auto"))
        with open(LOG_PATH, "rb") as f:
            client.put_object(Bucket=bucket, Key=_log_key(), Body=f.read()[-2_000_000:], ContentType="text/plain")
    except Exception as exc:  # logging must never take the worker down
        sys.__stderr__.write(f"[logship] {exc}\n")


def _log_shipper():
    while True:
        time.sleep(5)
        _ship_log()


def _get_predictor():
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            from predict import Predictor

            p = Predictor()
            p.setup()
            _predictor = p
    return _predictor


def _fetch_image(inp):
    if inp.get("image_base64"):
        fd, path = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(inp["image_base64"]))
        return path
    url = inp.get("image_url") or inp.get("image")
    if not url:
        raise ValueError("image_url is required")
    suffix = os.path.splitext(url.split("?")[0])[1] or ".png"
    req = urllib.request.Request(url, headers={"User-Agent": "pixal3dcog/1.0"})
    fd, path = tempfile.mkstemp(suffix=suffix)
    with urllib.request.urlopen(req, timeout=120) as resp, os.fdopen(fd, "wb") as f:
        f.write(resp.read())
    return path


def handler(job):
    payload = job.get("input") or {}
    inp = payload.get("input", payload)
    start = time.time()
    try:
        predictor = _get_predictor()
    except Exception:
        traceback.print_exc()
        _ship_log()
        return {"error": "worker setup failed: " + traceback.format_exc()[-1500:]}
    setup_seconds = round(time.time() - start, 2)
    try:
        image_path = _fetch_image(inp)
    except Exception as exc:
        return {"error": f"image fetch failed: {exc}"}
    from cog import Path

    from predict import predict_defaults

    # Calling predict() directly bypasses cog's input resolution, so unset
    # inputs would arrive as pydantic FieldInfo objects; fill real defaults.
    kwargs = predict_defaults()
    kwargs.update({k: v for k, v in inp.items() if k in PREDICT_FIELDS and v is not None})
    try:
        out = predictor.predict(image=Path(image_path), **kwargs)
    except Exception as exc:
        traceback.print_exc()
        _ship_log()
        return {"error": str(exc)}
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass
    result = {
        "seed": out.seed,
        "resolution": out.resolution,
        "timings": {k: round(v, 2) for k, v in out.timings.items()},
        "setup_seconds": setup_seconds,
        "gpu_seconds": round(time.time() - start, 2),
    }
    if out.glb_url:
        result["glb_url"] = out.glb_url
    elif out.glb:
        with open(str(out.glb), "rb") as f:
            result["glb_base64"] = base64.b64encode(f.read()).decode()
        result["file_name"] = "model.glb"
        os.remove(str(out.glb))
    return result


if __name__ == "__main__":
    print(f"[worker] starting pod={os.environ.get('RUNPOD_POD_ID')} endpoint={os.environ.get('RUNPOD_ENDPOINT_ID')}", flush=True)
    threading.Thread(target=_log_shipper, daemon=True).start()
    if os.environ.get("PIXAL3D_EAGER_SETUP", "").lower() in ("1", "true"):
        _get_predictor()
    runpod.serverless.start({"handler": handler})
