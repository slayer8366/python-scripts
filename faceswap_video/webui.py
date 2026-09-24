"""Gradio web interface: ``faceswap-video-ui``."""

from __future__ import annotations

import argparse
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr

from .cli import CONSENT_TEXT
from .engine import MODES, FaceSwapper
from .label import DEFAULT_LABEL
from .pipeline import SwapJob, run


@lru_cache(maxsize=1)
def _swapper() -> FaceSwapper:
    return FaceSwapper()


def swap(source, target, mode, reference, threshold, keep_label, max_frames, consent,
         progress=gr.Progress()):
    if not consent:
        raise gr.Error("Confirm that everyone involved has consented.")
    if not source or not target:
        raise gr.Error("Upload a source image and a target video.")
    if mode == "reference" and not reference:
        raise gr.Error("Reference mode needs an image of the person to replace.")

    output = Path(tempfile.mkdtemp(prefix="faceswap_")) / "swapped.mp4"
    job = SwapJob(
        source_image=Path(source),
        target_video=Path(target),
        output=output,
        mode=mode,
        reference_image=Path(reference) if reference else None,
        threshold=float(threshold),
        label=DEFAULT_LABEL if keep_label else None,
        max_frames=int(max_frames) or None,
    )

    def report(done: int, total: int) -> None:
        progress((done, total or None), unit="frames")

    try:
        stats = run(job, _swapper(), report)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    summary = (f"{stats.frames} frames processed, {stats.faces_swapped} faces swapped "
               f"in {stats.frames_with_swap} frames.")
    return str(output), summary


def build() -> gr.Blocks:
    with gr.Blocks(title="Video Face Swap") as demo:
        gr.Markdown("# Video face swap\n" + CONSENT_TEXT)
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="Source face", type="filepath")
                target = gr.Video(label="Target video")
                mode = gr.Radio(list(MODES), value="all", label="Faces to replace")
                reference = gr.Image(label="Reference: person to replace (reference mode)",
                                     type="filepath")
                threshold = gr.Slider(0.1, 0.9, value=0.35, step=0.01,
                                      label="Reference match threshold")
                keep_label = gr.Checkbox(value=True, label="Burn in 'AI-generated' label")
                max_frames = gr.Number(value=0, precision=0,
                                       label="Max frames (0 = whole video)")
                consent = gr.Checkbox(label="Everyone whose face is used or replaced has consented")
                button = gr.Button("Swap", variant="primary")
            with gr.Column():
                result = gr.Video(label="Result")
                summary = gr.Textbox(label="Summary", interactive=False)
        button.click(
            swap,
            [source, target, mode, reference, threshold, keep_label, max_frames, consent],
            [result, summary],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(prog="faceswap-video-ui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build().queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
