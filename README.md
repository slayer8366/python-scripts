# faceswap-video

Takes the face from one image and puts it onto the faces in a video. Uses
InsightFace for detection and identity embeddings (`buffalo_l`) and the
`inswapper_128` model for the swap itself. Includes a command-line tool and a
small Gradio web UI.

An on-device Android version lives in [`android/`](android/README.md).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ui,test]"
faceswap-video --download-models      # ~840 MB, into ~/.insightface/models
```

ffmpeg ships with the `imageio-ffmpeg` wheel, so no system install is needed.
For an NVIDIA GPU, replace `onnxruntime` with `onnxruntime-gpu` and pass
`--providers CUDAExecutionProvider CPUExecutionProvider`.

## Use

```bash
# Replace every face in the video
faceswap-video -s me.jpg -t clip.mp4 -o out.mp4 --consent

# Replace only one person, identified by a photo of them
faceswap-video -s me.jpg -t clip.mp4 -o out.mp4 --consent \
    --mode reference --reference friend.jpg

# Quick preview of the first 60 frames
faceswap-video -s me.jpg -t clip.mp4 -o preview.mp4 --consent --max-frames 60

# Web UI at http://127.0.0.1:7860
faceswap-video-ui
```

`--mode` options:

| mode        | swaps                                                              |
|-------------|--------------------------------------------------------------------|
| `all`       | every detected face (default)                                      |
| `largest`   | the biggest face in each frame                                     |
| `reference` | faces whose embedding matches `--reference` at `--threshold` or more |

The `reference` threshold is a cosine similarity between ArcFace embeddings.
0.35 is a starting point, not a calibrated value. Raise it if the wrong people
get swapped, lower it if the right person is missed in some frames (profile
views and motion blur lower the score).

## Consent and labelling

Every run requires `--consent` (a checkbox in the UI). Output carries a
visible "AI-GENERATED" label in the corner and an `AI-generated face swap`
comment in the MP4 metadata. `--no-label` removes the visible label only.

## Limits

- Each frame is processed on its own, so there is no temporal smoothing. Expect
  some flicker, especially on small or fast-moving faces.
- `inswapper_128` works at 128×128, so faces that are large in frame look
  softer than the rest of the image.
- Output frame rate is taken from the container's average. Variable-frame-rate
  sources (many phone recordings) can drift slightly against the audio.
- CPU speed is roughly a few seconds per frame with several faces. Use a GPU for
  anything longer than a short clip.

## Model licence

The code here is yours to use. The InsightFace pretrained models are
licensed for **non-commercial research only**. Models are fetched from the
official `deepinsight/insightface` `model-zoo` GitHub release, and the
swapper's SHA-256 is checked after download.

## Tests

```bash
pytest            # the end-to-end test runs only if the models are downloaded
```
