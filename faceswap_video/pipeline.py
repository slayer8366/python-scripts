"""End-to-end video processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .engine import FaceSwapper
from .label import DEFAULT_LABEL, draw_label
from .video import VideoWriter, probe, read_frames

Progress = Callable[[int, int], None]


@dataclass
class SwapJob:
    source_image: Path
    target_video: Path
    output: Path
    mode: str = "all"
    reference_image: Path | None = None
    threshold: float = 0.35
    label: str | None = DEFAULT_LABEL
    max_frames: int | None = None
    crf: int = 18


@dataclass
class SwapStats:
    frames: int = 0
    frames_with_swap: int = 0
    faces_swapped: int = 0


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def run(job: SwapJob, swapper: FaceSwapper, progress: Progress | None = None) -> SwapStats:
    source_face = swapper.face_from_image(load_image(job.source_image), "source image")

    reference_embedding = None
    if job.mode == "reference":
        if job.reference_image is None:
            raise ValueError("--mode reference needs a reference image of the person to replace.")
        ref = swapper.face_from_image(load_image(job.reference_image), "reference image")
        reference_embedding = ref.embedding

    info = probe(job.target_video)
    total = info.frame_count
    if job.max_frames:
        total = min(total, job.max_frames) if total else job.max_frames

    stats = SwapStats()
    comment = "AI-generated face swap" + (f" ({job.label})" if job.label else "")
    Path(job.output).parent.mkdir(parents=True, exist_ok=True)
    with VideoWriter(
        job.output, info.width, info.height, info.fps,
        audio_source=job.target_video, crf=job.crf, comment=comment,
    ) as writer:
        for frame in read_frames(job.target_video):
            if job.max_frames and stats.frames >= job.max_frames:
                break
            out, n = swapper.swap_frame(
                frame, source_face, job.mode, reference_embedding, job.threshold
            )
            if job.label:
                out = draw_label(out, job.label)
            writer.write(out)
            stats.frames += 1
            stats.faces_swapped += n
            stats.frames_with_swap += bool(n)
            if progress:
                progress(stats.frames, total)
    if stats.frames == 0:
        raise ValueError(f"No frames could be decoded from {job.target_video}")
    return stats
