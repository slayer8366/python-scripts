# Face Swap Video for Android

On-device port of the Python `faceswap-video` tool. Pick a photo of the face
to insert and a video; the phone detects faces in every frame, swaps them,
and saves an MP4 to `Movies/FaceSwap`. Nothing is uploaded anywhere.

## Getting the APK

- **CI**: every push touching `android/` runs `.github/workflows/android.yml`,
  which runs the core tests and uploads a debug APK as a build artifact.
- **Locally**: open `android/` in Android Studio, or run
  `./gradlew :app:assembleDebug` with the Android SDK installed
  (compileSdk 36). Install with `adb install app/build/outputs/apk/debug/app-debug.apk`.

On first launch, pick a swap model and tap **Download models** (Wi-Fi advised):

| model | download | notes |
|-------|----------|-------|
| **Fast (int8)**, default | ~466 MB | int8 build of inswapper_128, published on this repo's `swap-model-int8-v1` release |
| Full precision | ~843 MB | the original InsightFace model |

The int8 build comes from `tools/quantize_swapper.py`, run by the
`swap-model-int8` workflow on the official download. Two builds on different
machines produced the same bytes (`cfbfd851…`), and the app checks that hash.
Measured on a desktop CPU against full precision, over 120 swap pairs from 9
photos not used for calibration:

| | full precision | int8 |
|--|--|--|
| swap time per face | 733 ms | 267 ms (2.7x faster) |
| identity similarity to source (ArcFace cosine) | 0.902 | 0.897 |
| mean pixel difference | | 2.4 / 255 |

Before a long run, use **Preview one frame**: pick a moment with the slider
and the app swaps just that frame, showing before and after. It decodes the
frame through the same pipeline as the full run, so what you see is what
the video will get.

**Tracking.** Faces are linked across frames, so in one-person mode each
face's match is re-checked every 6 frames instead of every frame (6 instead
of 36 recognitions on a 12-frame, 3-face test clip). The optional **Track faces
between detections** setting also replaces most full-frame detections with
cheap detections in a crop around each face: about 11% faster per frame with
one face and the int8 model, no gain with three faces, landmarks within 1.7 px
of full detection (the detector's own jitter is 1.1 px), and a face entering
the shot can stay unswapped for up to 5 frames. It is off by default.

## Requirements

- Android 10 (API 29) or newer, 64-bit.
- Roughly 1 GB of free RAM while swapping with the full-precision model (the
  swap model alone is 554 MB); less with the int8 model (177 MB).
- SDR video. HDR / 10-bit recordings (a default on some recent phones) are
  rejected with a message. Convert them or record in SDR.

## Layout

| module | what |
|--------|------|
| `core/` | Pure Kotlin/JVM: SCRFD detection, ArcFace embeddings, inswapper, paste-back blending, YUV conversion, ONNX initializer reader. Runs on ONNX Runtime; testable on a desktop JVM. |
| `app/`  | Android framework only (no AndroidX): UI, model download, foreground service, MediaCodec decode/encode, audio passthrough, gallery save. |

The app uses no AndroidX libraries so every line could be compile-checked
against the Android framework in an environment without Google's Maven
repository. A Compose UI is a straightforward later swap.

## How faithful is the port?

`core/` reimplements insightface 2.0's pipeline and is tested against the
Python output on the same images (`core/tools/make_golden.py`):

| check | result |
|-------|--------|
| ArcFace embeddings, 3 faces | cosine 0.99999994 |
| Source embedding (padded-retry path) | cosine 0.9998 |
| 112 px alignment crop vs OpenCV | mean abs diff 0.00005 / 255 |
| 128 px generated face | mean abs diff 0.0009 / 255 |
| Full frame, all faces swapped | mean abs diff 0.012 / 255 |

Those tests run only when models and golden files are present:

```bash
FACESWAP_MODEL_DIR=~/.insightface python core/tools/make_golden.py /tmp/golden
FACESWAP_MODEL_DIR=~/.insightface FACESWAP_GOLDEN_DIR=/tmp/golden ./gradlew -PcoreOnly :core:test
```

`-PcoreOnly` skips the Android module, for machines without the Android SDK.

## Differences from the Python app

- Timestamps come from the decoder, so variable-frame-rate phone video keeps
  audio sync (the Python app assumes constant frame rate).
- Frames are rotated upright before detection. The output has no rotation flag.
- The "AI-generated" disclosure is the burned-in label plus a gallery
  description. `MediaMuxer` can't write a comment tag into the MP4 itself.
- If the encoder can't handle the source resolution, output is scaled down.

## Tests

- `core/`: JVM unit tests plus the Python comparison tests above, tracker
  logic tests, and opt-in model tests for the int8 swapper
  (`FACESWAP_INT8_DIR`), tracking accuracy, and per-stage timing
  (`FACESWAP_BENCH=1`).
- `app/src/androidTest`: instrumented tests of the video pipeline, run in CI
  on API 29 and API 35 emulators. They generate their own clips with
  `MediaCodec` (no fixtures) and check the output with
  `MediaMetadataRetriever`, an independent decoder, covering frame count,
  colours, audio passthrough and duration, upright rotation, frame seeking for
  the preview, and the 5-second limit. The per-frame step is a painted
  stand-in, so no models are needed. The face engine itself is covered by
  the core tests.
- Android Lint runs in CI and fails the build on errors such as API-level misuse.

## Not verified yet

These need a real device:

- Speed per frame and memory headroom on actual phones. All timings above are
  from a desktop x86 CPU with AVX-512 VNNI. Phone CPUs have comparable int8
  instructions, so int8 should also help there, but by an unmeasured amount.
- int8 quality beyond the 9 test photos, especially unusual lighting and
  skin tones the small calibration set (6 faces) may not cover.
- `MediaCodec` behaviour on vendor codecs. The emulators use Google's
  software codecs; Qualcomm, Exynos and MediaTek hardware codecs can differ.
- Audio passthrough for non-AAC tracks (falls back to silent output with a note).

## Model licence

InsightFace's pretrained models are for **non-commercial research only**.
They are downloaded from the official `deepinsight/insightface` model-zoo
release and checked against SHA-256 hashes before use.
