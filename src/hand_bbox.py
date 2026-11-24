"""
hand_bbox.py - Hitung kotak (bbox) tangan dengan MediaPipe untuk tiap gambar di split.

Dijalankan di environment terpisah .venv-mp (mediapipe tidak dibutuhkan untuk training):
  .venv-mp\\Scripts\\python src/hand_bbox.py --config configs/config_v3.yaml

Langkah per gambar: decode ~1024 px (cepat), HandLandmarker (maks 2 tangan, confidence 0.3), ambil gabungan
semua landmark, tambah margin 20%, jadikan persegi (dalam piksel) dan jepit ke dalam gambar.

Keluaran:
  results/crops/bboxes_v3.csv   : path,detected,x0,y0,x1,y1 (dinormalisasi 0-1 terhadap gambar asli)
  results/crops/crop_grid.png   : ~40 crop acak untuk pemeriksaan visual
  results/crops/detection_rate.csv : tingkat deteksi keseluruhan / per huruf / per latar
Model: models/hand_landmarker.task (diunduh otomatis dari storage.googleapis.com/mediapipe-models).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")
DECODE_PX = 1024
MARGIN = 0.20


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def make_detector(model_path: Path):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Mengunduh model tangan ke {model_path} ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
    )
    return mp, vision.HandLandmarker.create_from_options(opts)


def square_box(xs, ys, w: int, h: int, margin: float = MARGIN):
    """Gabungan landmark (0-1) -> kotak persegi (dalam piksel) + margin, dijepit ke gambar. Return 0-1."""
    x0, x1, y0, y1 = min(xs) * w, max(xs) * w, min(ys) * h, max(ys) * h
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    side = min(max(bw, bh), w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    return left / w, top / h, (left + side) / w, (top + side) / h


def detect_one(mp, det, path: Path):
    img = Image.open(path)
    img.draft("RGB", (DECODE_PX, DECODE_PX))
    img = img.convert("RGB")
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
    if not res.hand_landmarks:
        return None
    xs = [lm.x for hand in res.hand_landmarks for lm in hand]
    ys = [lm.y for hand in res.hand_landmarks for lm in hand]
    return square_box(xs, ys, *img.size)


def main():
    ap = argparse.ArgumentParser(description="Bbox tangan (MediaPipe) untuk semua gambar di split")
    ap.add_argument("--config", default="configs/config_v3.yaml")
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--out", default="results/crops/bboxes_v3.csv")
    ap.add_argument("--limit", type=int, default=0, help="hanya N gambar pertama (uji cepat)")
    a = ap.parse_args()

    cfg = yaml.safe_load(_abs(a.config).read_text(encoding="utf-8"))
    root = _abs(cfg["data_dir"])
    with open(_abs(cfg["split_path"]), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[: a.limit]
    mp, det = make_detector(_abs(a.model))

    out = _abs(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    boxes = {}
    for i, r in enumerate(rows, 1):
        try:
            boxes[r["path"]] = detect_one(mp, det, root / r["path"])
        except Exception as e:  # file rusak: anggap tidak terdeteksi, training memakai gambar penuh
            print(f"[WARN] {r['path']}: {e}")
            boxes[r["path"]] = None
        if i % 250 == 0:
            print(f"  {i}/{len(rows)} ...", flush=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "detected", "x0", "y0", "x1", "y1"])
        for r in sorted(rows, key=lambda r: r["path"]):
            b = boxes[r["path"]]
            w.writerow([r["path"], int(b is not None)] + ([f"{v:.5f}" for v in b] if b else ["", "", "", ""]))

    # tingkat deteksi: keseluruhan, per huruf, per latar
    rate = defaultdict(lambda: [0, 0])
    for r in rows:
        ok = boxes[r["path"]] is not None
        for key in ("all", f"letter:{r['label']}", f"background:{r.get('background', '-')}"):
            rate[key][0] += ok
            rate[key][1] += 1
    with open(out.with_name("detection_rate.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "detected", "total", "rate"])
        for k in sorted(rate):
            d, n = rate[k]
            w.writerow([k, d, n, f"{d / n:.4f}"])
    d, n = rate["all"]
    print(f"\nTingkat deteksi: {d}/{n} = {d / n:.1%}")
    low = sorted(((v[0] / v[1], k) for k, v in rate.items() if k != "all"))[:6]
    print("Terendah:", ", ".join(f"{k} {r:.0%}" for r, k in low))

    # grid ~40 crop acak (hanya yang terdeteksi)
    found = [r for r in rows if boxes[r["path"]] is not None]
    random.Random(0).shuffle(found)
    tiles = []
    for r in found[:40]:
        img = Image.open(root / r["path"])
        img.draft("RGB", (DECODE_PX, DECODE_PX))
        img = img.convert("RGB")
        b = boxes[r["path"]]
        W, H = img.size
        tile = img.crop((round(b[0] * W), round(b[1] * H), round(b[2] * W), round(b[3] * H))).resize((160, 160))
        ImageDraw.Draw(tile).text((4, 4), r["label"], fill=(255, 0, 0))
        tiles.append(tile)
    cols = 8
    grid = Image.new("RGB", (cols * 160, ((len(tiles) + cols - 1) // cols) * 160), "white")
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % cols) * 160, (k // cols) * 160))
    grid.save(out.with_name("crop_grid.png"))
    print(f"[OK] {out}\n[OK] {out.with_name('crop_grid.png')}")


if __name__ == "__main__":
    sys.exit(main())
