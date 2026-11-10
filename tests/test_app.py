"""Tes aplikasi Streamlit (AppTest). Tes yang butuh best.pth di-skip bila tidak ada."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from app import inference as inf  # noqa: E402

APP = str(ROOT / "app" / "streamlit_app.py")


def _runs():
    inf._scan.clear()
    return inf.list_runs()


def _need_run():
    runs = _runs()
    if not runs:
        pytest.skip("tidak ada run dengan best.pth")
    return runs[0]["name"]


@pytest.fixture(autouse=True)
def _stub_webrtc(monkeypatch):
    """AppTest memalsukan runtime Streamlit; komponen webrtc butuh sesi sungguhan, jadi di-stub."""
    try:
        import streamlit_webrtc
    except ImportError:
        return
    from types import SimpleNamespace
    monkeypatch.setattr(streamlit_webrtc, "webrtc_streamer", lambda **kw: SimpleNamespace(video_processor=None))


def test_app_loads_with_run():
    _need_run()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_app_no_runs_shows_error(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "RESULTS_DIR", tmp_path)
    inf._scan.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Belum ada model" in e.value for e in at.sidebar.error)
    inf._scan.clear()


def test_prepare_image_rejects_bad_input():
    with pytest.raises(inf.InputImageError):
        inf.prepare_image(b"bukan gambar")
    with pytest.raises(inf.InputImageError):
        inf.prepare_image(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, "PNG")
    with pytest.raises(inf.InputImageError):
        inf.prepare_image(buf.getvalue())


def test_predict_matches_demo():
    run = _need_run()
    cfg = inf.load_config(run)
    root = ROOT / cfg["data_dir"]
    split = ROOT / cfg["split_path"]
    if not root.exists() or not split.exists():
        pytest.skip("dataset/split tidak ada")
    with open(split, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"][:3]
    device = torch.device("cpu")
    model, tf, _ = inf.get_model(run, "cpu")
    for r in rows:
        p = root / r["path"]
        got = inf.predict(model, tf, inf.prepare_image(p.read_bytes(), exif=False), device, threshold=0.0)
        top, _ = inf.demo.predict_topk(model, tf, inf.demo.load_rgb(p, 448), device, k=3)
        assert got["letter"] == inf.CLASSES[top[0][0]]
        assert round(got["conf"], 3) == round(top[0][1], 3)
