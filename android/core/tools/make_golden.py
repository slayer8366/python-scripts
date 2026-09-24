"""Write reference outputs from the Python pipeline for the Kotlin tests.

    FACESWAP_MODEL_DIR=~/.insightface python make_golden.py OUT_DIR

Uses sample images bundled with the insightface wheel. Nothing written here
is committed; point FACESWAP_GOLDEN_DIR at OUT_DIR when running the tests.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import insightface
import numpy as np
from insightface.utils import face_align

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from faceswap_video.engine import FaceSwapper  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
root = Path(os.environ.get("FACESWAP_MODEL_DIR", "~/.insightface")).expanduser()
images = Path(insightface.__file__).parent / "data" / "images"

frame = cv2.imread(str(images / "t1.jpg"))[:480, :640].copy()
source = cv2.imread(str(images / "Tom_Hanks_54745.png"))
cv2.imwrite(str(out / "frame.png"), frame)
cv2.imwrite(str(out / "source.png"), source)

swapper = FaceSwapper(root, download=False, det_size=640)
faces = swapper.detect(frame)
src = swapper.face_from_image(source, "source")

swapped = frame
for f in faces:
    swapped = swapper.swapper.get(swapped, f, src, paste_back=True)
cv2.imwrite(str(out / "swapped.png"), swapped)

f0 = faces[0]
M112 = face_align.estimate_norm(f0.kps, 112)
M128 = face_align.estimate_norm(f0.kps, 128)
cv2.imwrite(str(out / "crop112.png"), face_align.norm_crop(frame, f0.kps, 112))

# Single-face paste-back, isolating the blend from later faces.
one = swapper.swapper.get(frame, f0, src, paste_back=True)
cv2.imwrite(str(out / "swapped_one.png"), one)
fake, _ = swapper.swapper.get(frame, f0, src, paste_back=False)
cv2.imwrite(str(out / "fake_one.png"), fake)

emap = swapper.swapper.emap
json.dump(
    {
        "faces": [
            {
                "bbox": f.bbox.tolist(),
                "score": float(f.det_score),
                "kps": f.kps.ravel().tolist(),
                "embedding": f.normed_embedding.tolist(),
            }
            for f in faces
        ],
        "source_embedding": src.normed_embedding.tolist(),
        "M112": M112.ravel().tolist(),
        "M128": M128.ravel().tolist(),
        "emap_shape": list(emap.shape),
        "emap_first": emap.ravel()[:8].tolist(),
        "emap_sum": float(emap.astype(np.float64).sum()),
    },
    open(out / "golden.json", "w"),
)
print(f"{len(faces)} faces; wrote {out}")
