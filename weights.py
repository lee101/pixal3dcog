"""Model repos the predictor needs in the Hugging Face cache."""

MODELS = [
    ("TencentARC/Pixal3D", {"allow_patterns": ["pipeline.json", "ckpts/*"], "ignore_patterns": ["*_mv.*"]}),
    ("camenduru/dinov3-vitl16-pretrain-lvd1689m", {}),
    ("Ruicheng/moge-2-vitl", {}),
    ("ZhengPeng7/BiRefNet", {}),
]
