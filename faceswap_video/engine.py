"""Face detection, identity matching, and per-frame swapping."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from . import models

MODES = ("all", "largest", "reference")


class NoFaceError(ValueError):
    """Raised when an image that must contain a face has none."""


def _area(face) -> float:
    x1, y1, x2, y2 = face.bbox[:4]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denom if denom else 0.0


def select_targets(
    faces: Sequence,
    mode: str = "all",
    reference_embedding: np.ndarray | None = None,
    threshold: float = 0.35,
) -> list:
    """Pick which detected faces in a frame get replaced.

    ``all`` swaps every face, ``largest`` swaps only the biggest one, and
    ``reference`` swaps faces whose ArcFace embedding is at least
    ``threshold`` cosine-similar to ``reference_embedding``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not faces:
        return []
    if mode == "all":
        return list(faces)
    if mode == "largest":
        return [max(faces, key=_area)]
    if reference_embedding is None:
        raise ValueError("reference mode needs a reference_embedding")
    return [
        f
        for f in faces
        if getattr(f, "embedding", None) is not None
        and cosine_similarity(f.embedding, reference_embedding) >= threshold
    ]


class FaceSwapper:
    """Wraps InsightFace's buffalo_l analyser and the inswapper_128 model."""

    def __init__(
        self,
        model_root: Path | None = None,
        providers: Sequence[str] | None = None,
        det_size: int = 640,
        download: bool = True,
    ):
        # Imported here so the pure helpers above stay usable without the
        # heavy runtime dependencies (and so tests can import this module).
        import insightface
        from insightface.app import FaceAnalysis

        root = Path(model_root) if model_root else models.default_model_root()
        if not download and not (root / "models" / models.ANALYSIS_PACK).is_dir():
            raise FileNotFoundError(f"{root / 'models' / models.ANALYSIS_PACK} is missing.")
        kwargs = {"providers": list(providers)} if providers else {}
        self.analyser = FaceAnalysis(
            name=models.ANALYSIS_PACK,
            root=str(root),
            allowed_modules=["detection", "recognition"],
            **kwargs,
        )
        self.analyser.prepare(ctx_id=0, det_size=(det_size, det_size))
        self.swapper = insightface.model_zoo.get_model(
            str(models.swapper_path(root, download=download)), **kwargs
        )

    def detect(self, image: np.ndarray) -> list:
        return self.analyser.get(image)

    def face_from_image(self, image: np.ndarray, what: str = "image"):
        """Return the largest face in a still image."""
        faces = self.detect(image)
        if not faces:
            # SCRFD often misses a face that fills the whole frame (tight
            # headshots, pre-aligned crops). Padding gives it context. Only the
            # embedding is used from still images, so coordinates don't matter.
            pad = max(image.shape[:2]) // 2
            padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            faces = self.detect(padded)
        if not faces:
            raise NoFaceError(f"No face found in the {what}.")
        return max(faces, key=_area)

    def swap_frame(
        self,
        frame: np.ndarray,
        source_face,
        mode: str = "all",
        reference_embedding: np.ndarray | None = None,
        threshold: float = 0.35,
    ) -> tuple[np.ndarray, int]:
        """Swap ``source_face`` onto the selected faces in one BGR frame.

        Returns the new frame and how many faces were replaced.
        """
        targets = select_targets(self.detect(frame), mode, reference_embedding, threshold)
        out = frame
        for target in targets:
            out = self.swapper.get(out, target, source_face, paste_back=True)
        return out, len(targets)
