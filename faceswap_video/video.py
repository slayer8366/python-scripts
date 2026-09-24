"""Frame reading with OpenCV and H.264 writing with ffmpeg.

ffmpeg comes from the ``imageio-ffmpeg`` wheel, so no system install is needed.
The original audio track, if any, is copied into the output.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # Some containers report 0 or nonsense; 30 is a safe fallback.
            fps=fps if 0 < fps < 1000 else 30.0,
            frame_count=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
    finally:
        cap.release()


def read_frames(path: str | Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoWriter:
    """Pipe BGR frames into ffmpeg, muxing audio from ``audio_source``."""

    def __init__(
        self,
        output: str | Path,
        width: int,
        height: int,
        fps: float,
        audio_source: str | Path | None = None,
        crf: int = 18,
        comment: str | None = None,
    ):
        cmd = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
        ]
        if audio_source is not None:
            cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac"]
        cmd += [
            # yuv420p needs even dimensions.
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-shortest",
        ]
        if comment:
            cmd += ["-metadata", f"comment={comment}"]
        cmd.append(str(output))
        self._size = (height, width)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != self._size:
            raise ValueError(f"Frame is {frame.shape[:2]}, expected {self._size}")
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode(errors="replace")
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._proc.kill()
            self._proc.wait()
