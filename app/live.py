"""live.py - prediksi video langsung (streamlit-webrtc) memakai model yang sama dengan demo.py --webcam.

Callback video berjalan di thread terpisah: jangan memakai st.* atau session_state di sini.
"""
from __future__ import annotations

import importlib.util
import threading
import time

import numpy as np
import torch
from PIL import Image

from app import inference as inf

CLASSES = inf.CLASSES

# Video yang tampil di tab Live adalah salinan yang di-encode ulang oleh aiortc (VP8) di Python.
# Bawaan aiortc: mulai 500 kbps, maks 1.5 Mbps -> buram di 720p; encoder juga dibuat ulang setiap
# bitrate berubah >10% (keyframe baru -> patah-patah saat bitrate naik). Aplikasi ini hanya lokal
# (localhost, tanpa STUN), jadi bandwidth bukan masalah: kunci bitrate di satu nilai tinggi.
LIVE_BITRATE = 4_000_000  # bps


def fix_encoder_bitrate(bps: int = LIVE_BITRATE) -> None:
    from aiortc.codecs import h264, vpx
    for mod in (vpx, h264):
        mod.MIN_BITRATE = mod.DEFAULT_BITRATE = mod.MAX_BITRATE = bps


def live_available() -> tuple[bool, str]:
    for mod, pip in (("streamlit_webrtc", "streamlit-webrtc"), ("av", "av"), ("cv2", "opencv-python")):
        if importlib.util.find_spec(mod) is None:
            return False, f"Paket {pip} belum terpasang (pip install -r requirements-app.txt)."
    return True, ""


def make_processor_class():
    """Dibuat lazy agar aplikasi tetap jalan bila streamlit-webrtc tidak terpasang."""
    import av
    import cv2
    from streamlit_webrtc import VideoProcessorBase

    fix_encoder_bitrate()

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = self.tf = self.device = self.cropper = None
            self.threshold, self.mirror = 0.5, False
            self.window = 15
            self.stab = inf.demo.LetterStabilizer(self.window, 0.8, 0.8, 1.0)
            self.lock = threading.Lock()
            self.word = ""
            self.letter, self.conf, self.ms = "?", 0.0, 0.0
            self.box = None
            self._latest, self._thread, self._last_recv = None, None, 0.0

        # ---- dipanggil dari thread utama Streamlit ----
        def configure(self, model, tf, device, cropper, threshold, mirror):
            self.model, self.tf, self.device, self.cropper = model, tf, device, cropper
            self.threshold, self.mirror = threshold, mirror
            self.stab.min_conf = max(threshold, 0.8)

        def edit_word(self, action: str):
            with self.lock:
                self.word = {"space": self.word + " ", "back": self.word[:-1], "reset": ""}[action]

        def state(self):
            with self.lock:
                return self.word, self.letter, self.conf, self.ms

        # ---- inferensi di thread latar: video tidak menunggu model ----
        def _worker(self):
            while time.time() - self._last_recv < 5:           # berhenti sendiri bila stream mati
                with self.lock:
                    frame, self._latest = self._latest, None
                if frame is None or self.model is None:
                    time.sleep(0.01)
                    continue
                try:
                    # konversi warna dilakukan di sini (hanya untuk frame yang benar-benar diprediksi),
                    # bukan di recv() untuk setiap frame video
                    rgb = frame.to_ndarray(format="rgb24")
                    if self.mirror:
                        rgb = rgb[:, ::-1]
                    h, w = rgb.shape[:2]
                    side = min(h, w)
                    x0, y0 = (w - side) // 2, (h - side) // 2
                    found = True
                    if self.cropper is not None:
                        crop_img, box, _ = self.cropper.crop(Image.fromarray(np.ascontiguousarray(rgb)))
                        found = crop_img is not None
                        if found:
                            x0, y0, side = box[0], box[1], box[2] - box[0]
                    else:
                        crop_img = Image.fromarray(np.ascontiguousarray(rgb[y0:y0 + side, x0:x0 + side]))
                    if not found:
                        self.stab.preds.clear()
                        self.stab.confs.clear()
                        with self.lock:
                            self.letter, self.conf, self.box = "—", 0.0, None
                        continue
                    top, ms = inf.demo.predict_topk(self.model, self.tf, crop_img, self.device, k=1)
                    idx, conf = top[0]
                    locked = self.stab.update(idx, conf, time.perf_counter())
                    with self.lock:
                        if locked is not None:
                            self.word += CLASSES[locked]
                        self.letter = CLASSES[idx] if conf >= self.threshold else "?"
                        self.conf, self.ms, self.box = conf, ms, (x0, y0, side)
                except Exception:
                    time.sleep(0.05)

        # ---- thread video: hanya menyimpan frame terbaru + menggambar hasil terakhir ----
        def recv(self, frame):
            bgr = frame.to_ndarray(format="bgr24")
            if self.mirror:
                bgr = cv2.flip(bgr, 1)
            self._last_recv = time.time()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()
            with self.lock:
                self._latest = frame   # frame lama dibuang (selalu terbaru); dikonversi di worker
                letter, conf, box = self.letter, self.conf, self.box
            h, w = bgr.shape[:2]
            if self.cropper is None or box is not None:
                x0, y0, side = box or ((w - min(h, w)) // 2, (h - min(h, w)) // 2, min(h, w))
                cv2.rectangle(bgr, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
                cv2.putText(bgr, f"Huruf: {letter} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
            else:
                cv2.putText(bgr, "tangan tidak terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    return LiveProcessor
