"""Read optional Hugging Face configuration from the caller's environment.

This module never modifies environment variables. Public checkpoints do not
require a token; authenticated access uses HF_TOKEN or HUGGING_FACE_HUB_TOKEN.
"""

import os
from pathlib import Path

HF_HOME = os.environ.get("HF_HOME", str(Path("~/.cache/huggingface").expanduser()))
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
