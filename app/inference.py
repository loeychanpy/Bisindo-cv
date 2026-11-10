"""inference.py - pembungkus tipis di atas src/demo.py untuk aplikasi Streamlit (hanya lokal)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import demo  # noqa: E402
from dataset import CLASSES, get_transforms  # noqa: E402

RESULTS_DIR = ROOT / "results" / "transfer"   # dapat di-monkeypatch pada tes
MIN_SIDE = 64


class InputImageError(ValueError):
    """Gambar tidak bisa dipakai; pesan sudah dalam Bahasa Indonesia."""


class CropUnavailable(RuntimeError):
    """Modul crop / mediapipe tidak tersedia; pesan sudah dalam Bahasa Indonesia."""


# --------------------------------------------------------------------------- #
# Daftar run
# --------------------------------------------------------------------------- #
def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def _scan(results_dir: str) -> list[dict]:
    runs = []
    for d in Path(results_dir).glob("*/"):
        if not ((d / "best.pth").exists() and (d / "config_used.json").exists()):
            continue
        m = _read_json(d / "metrics.json") or {}
        test, val = m.get("test") or {}, m.get("val") or {}
        runs.append(dict(
            name=d.name,
            test_f1=test.get("macro_f1"), test_acc=test.get("accuracy"),
            val_f1=val.get("macro_f1", m.get("best_val_macro_f1")),
        ))
    runs.sort(key=lambda r: (r["test_f1"] if r["test_f1"] is not None else -1,
                             r["val_f1"] if r["val_f1"] is not None else -1), reverse=True)
    return runs


def list_runs() -> list[dict]:
    """Run yang punya best.pth DAN config_used.json, terurut test macro-F1 lalu val macro-F1."""
    return _scan(str(RESULTS_DIR))


def run_label(r: dict) -> str:
    if r["test_f1"] is not None:
        return f"{r['name']} — test F1 {r['test_f1']:.2f}"
    if r["val_f1"] is not None:
        return f"{r['name']} — val F1 {r['val_f1']:.2f}"
    return r["name"]


def run_dir(run_name: str) -> Path:
    return RESULTS_DIR / run_name


def load_config(run_name: str) -> dict:
    return _read_json(run_dir(run_name) / "config_used.json") or {}


def device_options() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# --------------------------------------------------------------------------- #
# Model & crop
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat model ...")
def get_model(run_name: str, device_name: str):
    device = torch.device(device_name)
    model, img_size, ckpt = demo.load_model(run_dir(run_name), device)
    cfg = load_config(run_name)
    cfg["_best_epoch"] = ckpt.get("epoch")
    return model, get_transforms(train=False, img_size=img_size), cfg


def crop_available() -> tuple[bool, str]:
    if not (SRC / "hand_bbox.py").exists():
        return False, "Modul src/hand_bbox.py tidak ditemukan."
    if importlib.util.find_spec("mediapipe") is None:
        return False, "Paket mediapipe belum terpasang (pip install mediapipe)."
    return True, ""


@st.cache_resource(show_spinner="Memuat detektor tangan ...")
def get_cropper():
    ok, why = crop_available()
    if not ok:
        raise CropUnavailable(why)
    try:
        return demo.HandCropper()
    except Exception as e:  # unduhan model gagal, mediapipe rusak, dst.
        raise CropUnavailable(f"Detektor tangan gagal dimuat ({type(e).__name__}: {e}).") from e


# --------------------------------------------------------------------------- #
# Gambar & prediksi
# --------------------------------------------------------------------------- #
def prepare_image(file_or_bytes, mirror: bool = False, draft: int = 448, exif: bool = True) -> Image.Image:
    """Buka unggahan sebagai RGB. exif=True memutar foto HP; set False untuk meniru training/demo.py."""
    if hasattr(file_or_bytes, "getvalue"):
        data = file_or_bytes.getvalue()
    elif hasattr(file_or_bytes, "read"):
        data = file_or_bytes.read()
    else:
        data = bytes(file_or_bytes)
    if not data:
        raise InputImageError("File kosong.")
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypheim", b"ftypmsf1", b"ftyphevc"):
        raise InputImageError("Format HEIC tidak didukung. Ekspor ulang ke JPG atau PNG.")
    try:
        img = Image.open(io.BytesIO(data))
        if draft:
            img.draft("RGB", (draft, draft))
        if exif:
            img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise InputImageError(f"File rusak atau bukan gambar yang valid ({type(e).__name__}).") from e
    if min(img.size) < MIN_SIDE:
        raise InputImageError(f"Gambar terlalu kecil ({img.width}×{img.height}); minimal {MIN_SIDE} piksel.")
    return ImageOps.mirror(img) if mirror else img


def predict(model, tf, img: Image.Image, device, k: int = 3, threshold: float = 0.5, crop: bool = False) -> dict:
    """Klasifikasi satu gambar. Lihat PLAN_streamlit.md Fase 1 untuk arti kunci hasil."""
    box, hand_found, cropped = None, None, img
    if crop:
        cropper = get_cropper()  # bisa raise CropUnavailable
        cropped, box, _ = cropper.crop(img)
        hand_found = cropped is not None
        if not hand_found:
            return dict(letter="—", conf=0.0, topk=[], latency_ms=0.0, crop_box=None, hand_found=False)
    top, ms = demo.predict_topk(model, tf, cropped, device, k=k)
    best_i, best_p = top[0]
    return dict(
        letter=CLASSES[best_i] if best_p >= threshold else "?",
        conf=best_p,
        topk=[(CLASSES[i], p) for i, p in top],
        latency_ms=ms, crop_box=box, hand_found=hand_found,
    )
