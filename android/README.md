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

On first launch, tap **Download models** (about 840 MB, Wi-Fi advised).

## Requirements

- Android 10 (API 29) or newer, 64-bit.
- Roughly 1 GB of free RAM while swapping. The swap model alone is 554 MB.
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

## Not verified yet

These need a real device:

- Speed per frame and memory headroom on actual phones.
- `MediaCodec` behaviour across vendors (plane layouts, encoder input sizes).
- Audio passthrough for non-AAC tracks (falls back to silent output with a note).

## Model licence

InsightFace's pretrained models are for **non-commercial research only**.
They are downloaded from the official `deepinsight/insightface` model-zoo
release and checked against SHA-256 hashes before use.
