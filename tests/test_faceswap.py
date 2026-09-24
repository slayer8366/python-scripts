import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from faceswap_video import cli
from faceswap_video.engine import cosine_similarity, select_targets
from faceswap_video.label import draw_label
from faceswap_video.video import VideoWriter, ffmpeg_exe, probe, read_frames


def face(x1, y1, x2, y2, embedding=None):
    return SimpleNamespace(bbox=np.array([x1, y1, x2, y2], float), embedding=embedding)


def test_select_all_and_largest():
    small, big = face(0, 0, 10, 10), face(0, 0, 50, 40)
    assert select_targets([small, big], "all") == [small, big]
    assert select_targets([small, big], "largest") == [big]
    assert select_targets([], "largest") == []


def test_select_reference_uses_threshold():
    ref = np.array([1.0, 0.0])
    same = face(0, 0, 1, 1, np.array([0.9, 0.1]))
    other = face(0, 0, 1, 1, np.array([0.0, 1.0]))
    no_embedding = face(0, 0, 1, 1)
    picked = select_targets([same, other, no_embedding], "reference", ref, 0.35)
    assert picked == [same]


def test_select_rejects_bad_input():
    with pytest.raises(ValueError):
        select_targets([face(0, 0, 1, 1)], "nope")
    with pytest.raises(ValueError):
        select_targets([face(0, 0, 1, 1)], "reference")


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_label_changes_only_bottom_left():
    frame = np.full((360, 640, 3), 200, np.uint8)
    out = draw_label(frame)
    assert out.shape == frame.shape
    assert not np.array_equal(out[-60:, :320], frame[-60:, :320])
    assert np.array_equal(out[:200], frame[:200])
    assert np.array_equal(frame, np.full_like(frame, 200))  # input untouched


def _make_video(path: Path, w=64, h=48, n=10, fps=10, audio=True):
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={n / fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=duration={n / fps}", "-c:a", "aac"]
    cmd += ["-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)


def _streams(path: Path) -> str:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True).stderr


def test_writer_roundtrip_keeps_audio_and_tags(tmp_path):
    src, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_video(src)
    info = probe(src)
    with VideoWriter(out, info.width, info.height, info.fps, audio_source=src,
                     comment="AI-generated face swap") as writer:
        for frame in read_frames(src):
            writer.write(frame)
    meta = _streams(out)
    assert "Audio" in meta and "AI-generated face swap" in meta
    assert probe(out).frame_count == 10


def test_writer_handles_odd_size_and_no_audio(tmp_path):
    src, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_video(src, audio=False)
    with VideoWriter(out, 63, 47, 10, audio_source=src) as writer:
        writer.write(np.zeros((47, 63, 3), np.uint8))
    assert "Audio" not in _streams(out)
    assert (probe(out).width, probe(out).height) == (64, 48)


def test_writer_rejects_wrong_frame_size(tmp_path):
    with pytest.raises(ValueError):
        with VideoWriter(tmp_path / "o.mp4", 64, 48, 10) as writer:
            writer.write(np.zeros((10, 10, 3), np.uint8))


def test_cli_requires_consent(tmp_path, capsys):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"")
    with pytest.raises(SystemExit):
        cli.main(["-s", str(src), "-t", str(src), "-o", str(tmp_path / "o.mp4")])
    assert "--consent" in capsys.readouterr().err


def test_cli_reference_mode_needs_reference(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["-s", "a", "-t", "b", "-o", "c", "--mode", "reference", "--consent"])
    assert "--reference" in capsys.readouterr().err


MODEL_ROOT = Path(os.environ.get("FACESWAP_MODEL_DIR", "~/.insightface")).expanduser()
HAVE_MODELS = (MODEL_ROOT / "models" / "inswapper_128.onnx").exists() and (
    MODEL_ROOT / "models" / "buffalo_l").is_dir()


@pytest.mark.skipif(not HAVE_MODELS, reason="models not downloaded")
def test_end_to_end_swaps_faces(tmp_path):
    import insightface

    from faceswap_video.engine import FaceSwapper
    from faceswap_video.pipeline import SwapJob, run

    data = Path(insightface.__file__).parent / "data" / "images"
    group = cv2.imread(str(data / "t1.jpg"))[:480, :640]
    video = tmp_path / "group.mp4"
    with VideoWriter(video, 640, 480, 10) as writer:
        for _ in range(3):
            writer.write(group)

    swapper = FaceSwapper(MODEL_ROOT, download=False)
    stats = run(SwapJob(data / "Tom_Hanks_54745.png", video, tmp_path / "out.mp4"), swapper)
    assert stats.frames == 3
    assert stats.frames_with_swap == 3
    assert stats.faces_swapped >= 3
