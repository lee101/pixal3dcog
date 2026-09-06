"""RunPod serverless handler. Calls the cog Predictor in-process (no HTTP hop).

Input: {"image_url": ..., or "image_base64": ..., plus any predict() kwargs}
Output: {"glb_url": str} when S3 upload is configured, else
        {"glb_base64": str, "file_name": "model.glb"}, plus seed/resolution/timings.
"""

import base64
import os
import tempfile
import time
import urllib.request

import runpod
from cog import Path

from predict import Predictor

_predictor = None

PREDICT_FIELDS = {
    "resolution", "seed", "texture_size", "decimation_target", "remesh",
    "ss_guidance_strength", "ss_guidance_rescale", "ss_sampling_steps", "ss_rescale_t",
    "shape_slat_guidance_strength", "shape_slat_guidance_rescale", "shape_slat_sampling_steps", "shape_slat_rescale_t",
    "tex_slat_guidance_strength", "tex_slat_guidance_rescale", "tex_slat_sampling_steps", "tex_slat_rescale_t",
    "mesh_scale", "max_num_tokens", "fov", "upload",
}


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
        _predictor.setup()
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
        image_path = _fetch_image(inp)
    except Exception as exc:
        return {"error": f"image fetch failed: {exc}"}
    kwargs = {k: v for k, v in inp.items() if k in PREDICT_FIELDS and v is not None}
    try:
        out = _get_predictor().predict(image=Path(image_path), **kwargs)
    except Exception as exc:
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
    _get_predictor()
    runpod.serverless.start({"handler": handler})
