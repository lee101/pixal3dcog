#!/usr/bin/env python3
"""Populate the Hugging Face cache with everything the predictor needs.

The Docker build pulls the same files as pre-packed tarballs from our public
mirror (see cog.yaml); use this script to fill a pod or dev box straight from
the Hub instead, or to rebuild the mirror tarballs.

  python scripts/fetch_weights.py            # into the default HF cache
  HF_HOME=/models python scripts/fetch_weights.py
"""

import os
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weights import MODELS  # noqa: E402

if __name__ == "__main__":
    for repo, kwargs in MODELS:
        path = snapshot_download(repo, **kwargs)
        print(repo, "->", path)
