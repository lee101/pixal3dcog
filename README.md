# pixal3dcog

[Cog](https://github.com/replicate/cog) packaging of [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D)
(SIGGRAPH 2026, TRELLIS.2 backbone): single image to a textured PBR GLB.
One image runs as a cog HTTP server on a dedicated GPU pod or as a RunPod
serverless worker, and is the self-hosted image-to-3D backend for
[simplexgen.com](https://simplexgen.com) and [app.nz](https://app.nz).

Weights (Pixal3D, DINOv3 mirror, MoGe-2, BiRefNet; ~26GB) are fetched from the
Hub on first start rather than baked in, because GHCR rejects image layers
over 10GB. On RunPod attach a network volume: it is mounted at
`/runpod-volume` and the cache lands in `/runpod-volume/hf`, so only the first
worker ever downloads. `scripts/fetch_weights.py` fills any other cache
(`HF_HOME`) ahead of time. NAF's small checkpoint is baked. Everything is MIT or similarly
permissive; see `LICENSE.pixal3d` and the upstream `NOTICE`.

## Inputs

Same names as fal's `fal-ai/pixal3d` endpoint so callers can swap providers:

| name | default | notes |
|---|---|---|
| `image` | required | RGBA alpha is used as the mask, RGB is matted with BiRefNet |
| `resolution` | 1536 | 1024 is about 2x faster, lower detail |
| `seed` | -1 | random when negative |
| `texture_size` | 2048 | 1024 / 2048 / 4096 |
| `decimation_target` | 500000 | target faces |
| `remesh` | true | remesh before baking |
| `ss_*`, `shape_slat_*`, `tex_slat_*` | upstream defaults | steps, guidance, rescale per stage |
| `mesh_scale`, `max_num_tokens`, `fov` | 1.0, 49152, auto | `fov` in radians; MoGe-2 estimates it when <= 0 |
| `upload` | true | upload to S3 when configured, else return the file |

Output: `glb` (file) or `glb_url` (when `S3_BUCKET` is configured), `seed`,
`resolution`, `timings` (preprocess, camera, generate, export, total seconds).

## Runtime env

- `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`,
  `S3_PUBLIC_BASE_URL`, `S3_PREFIX`: upload results (R2, S3, MinIO). Without a
  public base the handler returns a 7-day presigned URL.
- `PIXAL3D_HF_HOME`: where to cache weights (defaults to `/runpod-volume/hf`
  when a RunPod volume is mounted, else the standard HF cache).
- `PIXAL3D_LOW_VRAM=1`: stage models on demand (auto when VRAM < 20GB).
- `ATTN_BACKEND=sdpa|flash_attn`: defaults to flash-attn 2 when importable.
- `PIXAL3D_WARMUP=1`: run one small generation at startup to fill the
  FlexGEMM autotune cache. A committed `autotune_cache.json` is used when present.

## Build

```
cog build -t ghcr.io/lee101/pixal3dcog:latest          # or, without cog:
docker build --network host -f Dockerfile.cog -t ghcr.io/lee101/pixal3dcog:latest .
docker build -f Dockerfile.sls -t ghcr.io/lee101/pixal3dcog:sls .
docker push ghcr.io/lee101/pixal3dcog:latest && docker push ghcr.io/lee101/pixal3dcog:sls
```

`Dockerfile.cog` is `cog debug` output made self-contained; regenerate it after
editing `cog.yaml`. The image is about 27GB (CUDA 12.4 base, torch 2.6, the
compiled extensions); a 72-core box builds it in ~25 minutes.

Native extensions (nvdiffrast, nvdiffrec renderutils, CuMesh, FlexGEMM,
o-voxel) are compiled for sm_80/86/89/90, so the image runs on A10, 3090,
A40, 4090, L40S, A100 and H100 workers. Blackwell needs a cu128 rebuild.

## Run

```
cog predict -i image=@assets/chair.png -i resolution=1536 -i texture_size=2048
```

RunPod serverless (`Dockerfile.sls` image, `RUNPOD_ENDPOINT_ID` set by RunPod;
`scripts/runpod_deploy.py` creates the template, endpoint and weight volume):

```
POST https://api.runpod.ai/v2/<endpoint>/run
{"input": {"image_url": "https://.../ref.png", "resolution": 1536, "texture_size": 2048}}
```

Serverless output: `{"glb_url": ...}` or `{"glb_base64": ..., "file_name": "model.glb"}`
plus `seed`, `resolution`, `timings`, `gpu_seconds`.

## Speed

Standard mode keeps all flow models resident (~18GB) and MoGe-2 on the GPU.
flash-attn 2, TF32 and cuDNN autotune are on; the FlexGEMM autotune cache
avoids re-tuning sparse GEMM kernels on every cold start.
