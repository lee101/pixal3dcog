"""Cog predictor for TencentARC/Pixal3D (TRELLIS.2 backbone) image-to-3D.

Inputs mirror fal's pixal3d endpoint so callers can swap providers without
re-mapping parameters. Output is a textured PBR GLB, returned as a file or,
when S3-compatible storage is configured, uploaded and returned as a URL.
"""

import math
import os
import random
import sys
import time
import uuid
from typing import Dict, Optional

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Weights live in the Hugging Face cache. On RunPod a network volume mounted
# at /runpod-volume keeps them across cold starts; otherwise they are fetched
# into the container on first start (weights.py lists the repos).
_HF_HOME = os.environ.get("PIXAL3D_HF_HOME") or ("/runpod-volume/hf" if os.path.isdir("/runpod-volume") else "")
if _HF_HOME:
    os.makedirs(_HF_HOME, exist_ok=True)
    os.environ["HF_HOME"] = _HF_HOME

PIXAL3D_ROOT = os.environ.get("PIXAL3D_ROOT", "/opt/pixal3d")
sys.path.insert(0, PIXAL3D_ROOT)

AUTOTUNE_CACHE = os.environ.get(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    "/src/autotune_cache.json" if os.path.exists("/src/autotune_cache.json") else "/tmp/flex_gemm_autotune_cache.json",
)
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = AUTOTUNE_CACHE


def _pick_attn_backend() -> str:
    forced = os.environ.get("ATTN_BACKEND")
    if forced:
        return forced
    try:
        import flash_attn  # noqa: F401

        return "flash_attn"
    except Exception:
        return "sdpa"


os.environ["ATTN_BACKEND"] = _pick_attn_backend()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cog import BaseModel, BasePredictor, Input, Path  # noqa: E402
from PIL import Image  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import inference as p3d  # noqa: E402  (from /opt/pixal3d)
import o_voxel  # noqa: E402
import pixal3d.pipelines.rembg as _rembg  # noqa: E402

# pipeline.json names briaai/RMBG-2.0 for matting, a gated, non-commercial
# repo. BiRefNet (MIT) is the same architecture and loads through the same
# wrapper, so swap the model id before the pipeline instantiates it.
REMBG_MODEL = os.environ.get("PIXAL3D_REMBG_MODEL", "ZhengPeng7/BiRefNet")
_OrigBiRefNet = _rembg.BiRefNet


class _BiRefNet(_OrigBiRefNet):
    def __init__(self, model_name: str = REMBG_MODEL):
        super().__init__(REMBG_MODEL)


_rembg.BiRefNet = _BiRefNet

GLB_ROTATION = np.array(
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


class Output(BaseModel):
    glb: Optional[Path]
    glb_url: Optional[str]
    seed: int
    resolution: int
    timings: Dict[str, float]


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class S3Uploader:
    """Optional upload of results to any S3-compatible bucket (R2, S3, MinIO)."""

    def __init__(self):
        self.bucket = os.environ.get("S3_BUCKET", "").strip()
        self.endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip()
        self.key_id = os.environ.get("S3_ACCESS_KEY_ID", "").strip() or os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
        self.secret = os.environ.get("S3_SECRET_ACCESS_KEY", "").strip() or os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        self.public_base = os.environ.get("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")
        self.prefix = os.environ.get("S3_PREFIX", "pixal3d").strip().strip("/")
        self.client = None
        if self.bucket and self.key_id and self.secret:
            import boto3

            self.client = boto3.client(
                "s3",
                endpoint_url=self.endpoint or None,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.secret,
                region_name=os.environ.get("S3_REGION", "auto"),
            )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def upload(self, local_path: str, content_type: str = "model/gltf-binary") -> str:
        key = f"{self.prefix}/{uuid.uuid4().hex}.glb" if self.prefix else f"{uuid.uuid4().hex}.glb"
        with open(local_path, "rb") as f:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=f, ContentType=content_type)
        if self.public_base:
            return f"{self.public_base}/{key}"
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=7 * 24 * 3600
        )


def ensure_weights():
    """Download the model repos unless the cache already has them, then go offline."""
    from huggingface_hub import snapshot_download

    from weights import MODELS

    t0 = time.time()
    for repo, kwargs in MODELS:
        try:
            snapshot_download(repo, local_files_only=True, **kwargs)
        except Exception:
            print(f"[weights] fetching {repo}", flush=True)
            snapshot_download(repo, **kwargs)
    naf = os.path.join(torch.hub.get_dir(), "checkpoints", "naf_release.pth")
    if not os.path.exists(naf):
        os.makedirs(os.path.dirname(naf), exist_ok=True)
        torch.hub.download_url_to_file("https://github.com/valeoai/NAF/releases/download/model/naf_release.pth", naf)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"[weights] ready in {time.time() - t0:.1f}s at {os.environ.get('HF_HOME', '~/.cache/huggingface')}", flush=True)


def predict_defaults() -> Dict[str, object]:
    """Plain default values of predict() inputs, for callers that bypass the cog server."""
    import inspect

    out = {}
    for name, param in inspect.signature(Predictor.predict).parameters.items():
        if name in ("self", "image"):
            continue
        default = param.default
        # cog.input.FieldInfo is cog's own class (not pydantic's), so duck-type it.
        if type(default).__name__ == "FieldInfo" and hasattr(default, "default"):
            default = default.default
        out[name] = default
    return out


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        ensure_weights()
        vram = _vram_gb()
        # Standard mode keeps all flow models resident (~18GB). Anything under
        # 20GB (A10 24GB is fine, 16GB cards are not) falls back to low-VRAM.
        self.low_vram = _env_flag("PIXAL3D_LOW_VRAM", vram < 20)
        self.default_resolution = int(os.environ.get("PIXAL3D_DEFAULT_RESOLUTION", "1024" if self.low_vram else "1536"))
        print(f"[setup] vram={vram:.1f}GB low_vram={self.low_vram} attn={os.environ['ATTN_BACKEND']} autotune_cache={AUTOTUNE_CACHE}")
        self.pipeline = p3d.init_pipeline(os.environ.get("PIXAL3D_MODEL_PATH", p3d.MODEL_PATH), low_vram=self.low_vram)
        # MoGe-2 (camera estimation) is small; keep it resident unless memory is tight.
        self.moge = p3d.load_moge_model(device="cuda" if not self.low_vram else "cpu")
        self.moge_device = "cuda" if not self.low_vram else "cpu"
        self.uploader = S3Uploader()
        self.workdir = "/tmp/pixal3d"
        os.makedirs(self.workdir, exist_ok=True)
        print(f"[setup] ready in {time.time() - t0:.1f}s upload={'s3' if self.uploader.enabled else 'file'}")
        if _env_flag("PIXAL3D_WARMUP", False):
            self._warmup()

    def _warmup(self):
        sample = os.path.join(PIXAL3D_ROOT, "assets", "images", "0_img.png")
        if not os.path.exists(sample):
            return
        t0 = time.time()
        try:
            self.predict(image=Path(sample), **{**predict_defaults(), "resolution": 1024, "seed": 1, "texture_size": 1024, "decimation_target": 100000, "upload": False})
            print(f"[setup] warmup done in {time.time() - t0:.1f}s")
        except Exception as exc:  # warmup is best effort
            print(f"[setup] warmup failed: {exc}")

    def _camera_params(self, image: Image.Image, fov: float, mesh_scale: float):
        if fov > 0:
            distance = p3d.distance_from_fov(
                fov, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0, 511]), mesh_scale, 512
            )["distance_from_x"]
            return {"camera_angle_x": float(fov), "distance": float(distance), "mesh_scale": mesh_scale}
        tmp = os.path.join(self.workdir, f"pre_{uuid.uuid4().hex}.png")
        image.save(tmp)
        try:
            if self.moge_device == "cpu":
                self.moge.cuda()
            params = p3d.get_camera_params_wild_moge(tmp, self.moge, device="cuda", mesh_scale=mesh_scale, image_resolution=512)
            if self.moge_device == "cpu":
                self.moge.cpu()
                torch.cuda.empty_cache()
        finally:
            os.remove(tmp)
        return params

    def predict(
        self,
        image: Path = Input(description="Reference image. RGBA alpha is used as the mask; RGB is matted automatically."),
        resolution: int = Input(description="Pipeline resolution. 1536 is best quality; 1024 is ~2x faster.", default=1536, choices=[1024, 1536]),
        seed: int = Input(description="Random seed. -1 picks one.", default=-1),
        texture_size: int = Input(description="Baked texture size.", default=2048, choices=[1024, 2048, 4096]),
        decimation_target: int = Input(description="Target face count after decimation.", default=500000, ge=10000, le=2000000),
        remesh: bool = Input(description="Remesh before baking for cleaner topology.", default=True),
        ss_guidance_strength: float = Input(default=7.5, ge=0, le=20),
        ss_guidance_rescale: float = Input(default=0.7, ge=0, le=1),
        ss_sampling_steps: int = Input(default=12, ge=1, le=50),
        ss_rescale_t: float = Input(default=5.0, ge=0, le=10),
        shape_slat_guidance_strength: float = Input(default=7.5, ge=0, le=20),
        shape_slat_guidance_rescale: float = Input(default=0.5, ge=0, le=1),
        shape_slat_sampling_steps: int = Input(default=12, ge=1, le=50),
        shape_slat_rescale_t: float = Input(default=3.0, ge=0, le=10),
        tex_slat_guidance_strength: float = Input(default=1.0, ge=0, le=20),
        tex_slat_guidance_rescale: float = Input(default=0.0, ge=0, le=1),
        tex_slat_sampling_steps: int = Input(default=12, ge=1, le=50),
        tex_slat_rescale_t: float = Input(default=3.0, ge=0, le=10),
        mesh_scale: float = Input(default=1.0, ge=0.1, le=10),
        max_num_tokens: int = Input(default=49152, ge=4096, le=200000),
        fov: float = Input(description="Manual horizontal FOV in radians; <= 0 estimates it with MoGe-2.", default=-1.0),
        upload: bool = Input(description="Upload the GLB to configured S3 storage and return a URL instead of a file.", default=True),
    ) -> Output:
        timings: Dict[str, float] = {}
        t0 = time.time()
        if seed is None or seed < 0:
            seed = random.randint(0, 2**31 - 1)
        if resolution not in (1024, 1536):
            resolution = self.default_resolution
        if self.low_vram and resolution == 1536 and not _env_flag("PIXAL3D_ALLOW_1536_LOW_VRAM", True):
            resolution = 1024

        img = Image.open(str(image))
        pre = self.pipeline.preprocess_image(img)
        timings["preprocess"] = time.time() - t0

        t1 = time.time()
        camera = self._camera_params(pre, fov, mesh_scale)
        timings["camera"] = time.time() - t1

        t2 = time.time()
        torch.manual_seed(seed)
        mesh_list, (_, _, res) = self.pipeline.run(
            pre,
            camera_params=camera,
            seed=seed,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
            },
            preprocess_image=False,
            return_latent=True,
            pipeline_type=f"{resolution}_cascade",
            max_num_tokens=max_num_tokens,
        )
        timings["generate"] = time.time() - t2

        t3 = time.time()
        mesh = mesh_list[0]
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=self.pipeline.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target, texture_size=texture_size,
            remesh=remesh, remesh_band=1, remesh_project=0, use_tqdm=False,
        )
        glb.apply_transform(GLB_ROTATION)
        out_path = os.path.join(self.workdir, f"pixal3d_{uuid.uuid4().hex}.glb")
        glb.export(out_path, extension_webp=True)
        timings["export"] = time.time() - t3
        timings["total"] = time.time() - t0
        torch.cuda.empty_cache()
        print(f"[predict] seed={seed} res={resolution} faces={len(glb.faces) if hasattr(glb, 'faces') else -1} timings={ {k: round(v, 1) for k, v in timings.items()} }")

        if upload and self.uploader.enabled:
            url = self.uploader.upload(out_path)
            os.remove(out_path)
            return Output(glb=None, glb_url=url, seed=seed, resolution=resolution, timings=timings)
        return Output(glb=Path(out_path), glb_url=None, seed=seed, resolution=resolution, timings=timings)
