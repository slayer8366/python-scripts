"""Locate and download the ONNX models the swapper needs.

Both models come from the official InsightFace ``model-zoo`` GitHub release.
InsightFace publishes its pretrained weights for non-commercial research use
only, so the first download prints that notice.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/"
SWAPPER_FILENAME = "inswapper_128.onnx"
SWAPPER_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
ANALYSIS_PACK = "buffalo_l"

LICENSE_NOTICE = (
    "InsightFace pretrained models (buffalo_l, inswapper_128) are licensed for "
    "non-commercial research use only. See https://github.com/deepinsight/insightface"
)


def default_model_root() -> Path:
    """Directory holding the models. Override with FACESWAP_MODEL_DIR."""
    return Path(os.environ.get("FACESWAP_MODEL_DIR", "~/.insightface")).expanduser()


def _download(url: str, dest: Path, sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}", file=sys.stderr)
    print(LICENSE_NOTICE, file=sys.stderr)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if sha256 and digest.hexdigest() != sha256:
        partial.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}; download discarded.")
    partial.replace(dest)


def swapper_path(model_root: Path | None = None, download: bool = True) -> Path:
    """Return the path to inswapper_128.onnx, downloading it if allowed."""
    root = model_root or default_model_root()
    path = root / "models" / SWAPPER_FILENAME
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} is missing. Run `faceswap-video --download-models` or place "
                f"{SWAPPER_FILENAME} there manually."
            )
        _download(RELEASE_URL + SWAPPER_FILENAME, path, SWAPPER_SHA256)
    return path
