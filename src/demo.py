"""
demo.py - Demo inferensi BISINDO A-Z dari model hasil src/train_transfer.py.

  python src/demo.py --run mnv3s_partial3 --image foto1.jpg foto2.jpg     # demo offline (utama)
  python src/demo.py --run mnv3s_partial3 --webcam                        # bonus; perlu: pip install opencv-python
  python src/demo.py --run <run_crop> --crop --image foto1.jpg            # dengan deteksi tangan (MediaPipe)

Model dibaca dari results/transfer/<run>/best.pth (tidak di-commit; ada di laptop yang melatihnya).

Catatan penting:
  - Tanpa --crop model TIDAK punya deteksi tangan: tanpa tangan di bingkai ia tetap menebak satu huruf.
    Karena itu prediksi di bawah --threshold ditampilkan sebagai "?" dan tidak dikunci.
  - Dengan --crop, MediaPipe mencari tangan dan gambar dipotong ke tangan (sama dengan training crop=bbox).
    Bila tidak ada tangan, demo menampilkan "tangan tidak terdeteksi" dan tidak menebak.
    Butuh `pip install mediapipe` di environment demo (model tangan diunduh otomatis ke models/).
  - Mode webcam memotong tengah bingkai menjadi persegi (kotak hijau): letakkan tangan di dalam kotak.
    Foto latih berbentuk persegi, jadi bingkai 4:3 / 16:9 yang dipaksa jadi persegi akan terdistorsi.
  - Jalankan di terminal, bukan di dalam Jupyter (cv2.imshow di notebook membuat kernel macet).
    Tombol: q = keluar, c = hapus kata. Jendela video harus dalam fokus agar tombol terbaca.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CLASSES, get_transforms  # noqa: E402
from train_transfer import ROOT, build_model, load_rgb  # noqa: E402


class HandCropper:
    """Deteksi tangan MediaPipe -> crop persegi (aturan sama dengan src/hand_bbox.py). Dipakai oleh --crop."""

    def __init__(self, model_path: Path | None = None):
        from hand_bbox import make_detector  # impor di sini agar demo tanpa --crop tidak butuh mediapipe
        self.mp, self.det = make_detector(model_path or ROOT / "models" / "hand_landmarker.task")

    def crop(self, img: Image.Image):
        """Return (gambar_crop atau None bila tidak ada tangan, kotak piksel atau None, ms detektor)."""
        from hand_bbox import square_box
        t0 = time.perf_counter()
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.asarray(img)))
        ms = (time.perf_counter() - t0) * 1000
        if not res.hand_landmarks:
            return None, None, ms
        xs = [lm.x for h in res.hand_landmarks for lm in h]
        ys = [lm.y for h in res.hand_landmarks for lm in h]
        w, h = img.size
        b = square_box(xs, ys, w, h)
        box = (round(b[0] * w), round(b[1] * h), round(b[2] * w), round(b[3] * h))
        return img.crop(box), box, ms


def load_model(run_dir: Path, device: torch.device):
    cfg = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    ckpt = torch.load(run_dir / "best.pth", map_location=device, weights_only=True)
    model = build_model(cfg["mode"], cfg.get("unfreeze_blocks", 2), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg["img_size"], ckpt


@torch.no_grad()
def predict_topk(model, tf, img: Image.Image, device, k: int = 3):
    """Return [(idx, prob), ...] top-k dan latency (ms) untuk preprocess + forward."""
    t0 = time.perf_counter()
    x = tf(img).unsqueeze(0).to(device)
    probs = model(x).softmax(1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    p, i = probs.topk(k)
    return [(int(a), float(b)) for a, b in zip(i, p)], ms


class LetterStabilizer:
    """Kunci satu huruf bila prediksinya stabil di `window` frame terakhir dan cukup yakin."""

    def __init__(self, window: int = 15, min_ratio: float = 0.8, min_conf: float = 0.8, cooldown: float = 1.0):
        self.window, self.min_ratio, self.min_conf, self.cooldown = window, min_ratio, min_conf, cooldown
        self.preds: deque = deque(maxlen=window)
        self.confs: deque = deque(maxlen=window)
        self.last_lock = -1e9

    def update(self, idx: int, conf: float, now: float):
        self.preds.append(idx)
        self.confs.append(conf)
        if len(self.preds) < self.window:
            return None
        top, count = Counter(self.preds).most_common(1)[0]
        avg_conf = float(np.mean([c for p, c in zip(self.preds, self.confs) if p == top]))
        if count >= self.min_ratio * self.window and avg_conf >= self.min_conf and now - self.last_lock >= self.cooldown:
            self.last_lock = now
            self.preds.clear()
            self.confs.clear()
            return top
        return None


def run_images(model, tf, device, paths, threshold, cropper=None):
    for p in paths:
        try:
            img = load_rgb(p, 1024 if cropper else 448)
        except (FileNotFoundError, OSError) as e:
            print(f"{Path(p).name}: gagal dibuka ({type(e).__name__}), dilewati")
            continue
        det_ms = 0.0
        if cropper:
            img, _, det_ms = cropper.crop(img)
            if img is None:
                print(f"{Path(p).name}: tangan tidak terdeteksi (tidak menebak) | detektor {det_ms:.1f} ms")
                continue
        top, ms = predict_topk(model, tf, img, device)
        best_i, best_p = top[0]
        flag = "" if best_p >= threshold else "  (di bawah threshold)"
        alts = ", ".join(f"{CLASSES[i]} {pr:.2f}" for i, pr in top)
        lat = (f"{ms:.1f} ms" if not cropper
               else f"{ms:.1f} ms tanpa detektor | {ms + det_ms:.1f} ms dengan detektor ({det_ms:.1f} ms)")
        print(f"{Path(p).name}: {CLASSES[best_i]} ({best_p:.2f}){flag} | top-3: {alts} | {lat}")


def run_webcam(model, tf, device, args, cropper=None):
    import cv2  # impor di sini agar mode --image tidak butuh OpenCV

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Coba --camera 1 dan tutup aplikasi lain yang memakai kamera.")
    stab = LetterStabilizer(args.window, 0.8, args.threshold, 1.0)
    word, t_prev, fps = "", time.perf_counter(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame gagal dibaca, berhenti.")
                break
            h, w = frame.shape[:2]
            side = min(h, w)
            x0, y0 = (w - side) // 2, (h - side) // 2
            if cropper:
                hand, box, _ = cropper.crop(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if hand is None:
                    cv2.putText(frame, "tangan tidak terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                    cv2.imshow("BISINDO demo (q=keluar, c=hapus)", frame)
                    stab.preds.clear()
                    stab.confs.clear()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("c"):
                        word = ""
                    continue
                x0, y0, side = box[0], box[1], box[2] - box[0]
                crop_img = hand
            else:
                crop_img = Image.fromarray(cv2.cvtColor(frame[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB))
            top, ms = predict_topk(model, tf, crop_img, device, k=1)
            idx, conf = top[0]
            locked = stab.update(idx, conf, time.perf_counter())
            if locked is not None:
                word += CLASSES[locked]

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            shown = CLASSES[idx] if conf >= args.threshold else "?"
            cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
            cv2.putText(frame, f"Huruf: {shown} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Kata: {word}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"{fps:.1f} FPS | {ms:.0f} ms", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.imshow("BISINDO demo (q=keluar, c=hapus)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                word = ""
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return word


def main(argv=None):
    ap = argparse.ArgumentParser(description="Demo BISINDO A-Z")
    ap.add_argument("--run", required=True, help="nama run di results/transfer/")
    ap.add_argument("--image", nargs="+", help="satu atau lebih gambar")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--crop", action="store_true", help="deteksi tangan (MediaPipe) lalu crop sebelum klasifikasi")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    if not a.image and not a.webcam:
        ap.error("pilih --image <file> atau --webcam")

    run_dir = (ROOT / a.results_dir / "transfer" / a.run) if not Path(a.results_dir).is_absolute() \
        else Path(a.results_dir) / "transfer" / a.run
    if not (run_dir / "best.pth").exists():
        raise SystemExit(f"{run_dir / 'best.pth'} tidak ada. Latih dulu dengan src/train_transfer.py "
                         "(best.pth tidak di-commit, jadi hanya ada di laptop yang melatihnya).")
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if a.device == "auto" else a.device))
    model, img_size, ckpt = load_model(run_dir, device)
    tf = get_transforms(train=False, img_size=img_size)
    print(f"Model: {a.run} (epoch {ckpt.get('epoch')}, val macro-F1 {ckpt.get('val_f1', float('nan')):.3f}) | {device}")
    train_crop = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8")).get("crop", "none")
    if train_crop != "none" and not a.crop:
        print(f"[PERINGATAN] run ini dilatih dengan crop={train_crop}; tanpa --crop akurasi bisa turun.")
    if a.crop and train_crop == "none":
        print("[PERINGATAN] run ini dilatih tanpa crop; --crop mengubah distribusi masukan.")
    cropper = HandCropper() if a.crop else None
    if a.image:
        run_images(model, tf, device, a.image, a.threshold, cropper)
    if a.webcam:
        word = run_webcam(model, tf, device, a, cropper)
        print(f"Kata terakhir: {word!r}")


if __name__ == "__main__":
    main()
