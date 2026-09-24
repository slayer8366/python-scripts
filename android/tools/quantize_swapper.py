"""Build the int8 inswapper_128 used by the Android app's fast mode.

    python quantize_swapper.py MODEL_DIR OUT_DIR

MODEL_DIR holds models/inswapper_128.onnx and models/buffalo_l/ (the official
InsightFace downloads). Writes OUT_DIR/inswapper_128_int8.onnx and
OUT_DIR/inswapper_emap.bin (the 512x512 embedding projection, which
quantization drops from the graph), then prints their SHA-256 hashes.

Recipe, chosen by measuring speed and identity preservation on held-out
faces against the fp32 model:
  - static QDQ quantization of Conv layers only (92% of run time),
  - per-channel int8 weights, uint8 activations,
  - percentile (99.99) calibration on 16 swap pairs built from the faces in
    insightface's bundled t1.jpg,
  - first and last Conv left in fp32.
Calibration inputs are fixed, so with the pinned package versions in
.github/workflows/swap-model-int8.yml the output should be reproducible.
"""

import hashlib
import os
import sys
from pathlib import Path

import cv2
import insightface
import numpy as np
import onnx
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from onnx import numpy_helper, version_converter
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quant_pre_process, quantize_static)

EXPECTED_FP32_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
CALIBRATION_PAIRS = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main(model_dir: Path, out_dir: Path) -> None:
    fp32 = model_dir / "models" / "inswapper_128.onnx"
    if sha256(fp32) != EXPECTED_FP32_SHA256:
        sys.exit(f"{fp32} is not the official inswapper_128.onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(fp32)
    emap = numpy_helper.to_array(model.graph.initializer[-1]).astype("<f4")
    assert emap.shape == (512, 512), emap.shape
    # Big-endian floats, matching Java's DataInputStream.readFloat in the app.
    emap_path = out_dir / "inswapper_emap.bin"
    emap.astype(">f4").tofile(emap_path)

    app = FaceAnalysis("buffalo_l", root=str(model_dir), allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(0, det_size=(640, 640))
    img = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    faces = sorted(app.get(img), key=lambda f: float(f.bbox[0]))  # fixed order
    pairs = []
    for i, target in enumerate(faces):
        aimg, _ = face_align.norm_crop2(img, target.kps, 128)
        blob = cv2.dnn.blobFromImage(aimg, 1 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        for j, source in enumerate(faces):
            if i != j:
                lat = source.normed_embedding.reshape(1, -1) @ emap.astype(np.float32)
                pairs.append({"target": blob, "source": (lat / np.linalg.norm(lat)).astype(np.float32)})
    pairs = pairs[:CALIBRATION_PAIRS]
    print(f"{len(faces)} calibration faces, {len(pairs)} pairs")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(pairs)

        def get_next(self):
            return next(self.it, None)

    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    # Per-channel DequantizeLinear needs opset 13; the model ships as opset 11.
    op13 = work / "op13.onnx"
    onnx.save(version_converter.convert_version(model, 13), op13)
    pre = work / "pre.onnx"
    quant_pre_process(str(op13), str(pre), skip_symbolic_shape=True)
    convs = [n.name for n in onnx.load(pre, load_external_data=False).graph.node if n.op_type == "Conv"]

    out = out_dir / "inswapper_128_int8.onnx"
    quantize_static(
        str(pre), str(out), Reader(),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"], nodes_to_exclude=[convs[0], convs[-1]],
        calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.99},
    )
    for p in work.iterdir():
        p.unlink()
    work.rmdir()

    for p in (out, emap_path):
        print(f"{sha256(p)}  {p.name}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
