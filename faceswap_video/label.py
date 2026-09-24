"""Visible disclosure label burned into every output frame."""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_LABEL = "AI-GENERATED: face swapped"


def draw_label(frame: np.ndarray, text: str = DEFAULT_LABEL) -> np.ndarray:
    """Draw ``text`` on a dark translucent box in the bottom-left corner."""
    h, w = frame.shape[:2]
    scale = max(0.4, h / 900)
    thickness = max(1, round(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(4, round(th * 0.5))
    x0, y1 = pad, h - pad
    x1 = min(w, x0 + tw + 2 * pad)
    y0 = max(0, y1 - th - baseline - 2 * pad)

    out = frame.copy()
    box = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (box * 0.35).astype(frame.dtype)
    cv2.putText(
        out,
        text,
        (x0 + pad, y1 - pad - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out
