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

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = self.tf = self.device = self.cropper = None
            self.threshold, self.mirror = 0.5, False
            self.window = 15
            self.stab = inf.demo.LetterStabilizer(self.window, 0.8, 0.8, 1.0)
            self.lock = threading.Lock()
            self.word = ""
            self.letter, self.conf, self.ms = "?", 0.0, 0.0

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

        # ---- thread video ----
        def recv(self, frame):
            bgr = frame.to_ndarray(format="bgr24")
            if self.model is None:
                return frame
            if self.mirror:
                bgr = np.ascontiguousarray(bgr[:, ::-1])
            h, w = bgr.shape[:2]
            side = min(h, w)
            x0, y0 = (w - side) // 2, (h - side) // 2
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            try:
                if self.cropper is not None:
                    crop_img, box, _ = self.cropper.crop(rgb)
                    if crop_img is None:
                        self.stab.preds.clear()
                        self.stab.confs.clear()
                        with self.lock:
                            self.letter, self.conf = "—", 0.0
                        cv2.putText(bgr, "tangan tidak terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 2)
                        return av.VideoFrame.from_ndarray(bgr, format="bgr24")
                    x0, y0, side = box[0], box[1], box[2] - box[0]
                else:
                    crop_img = rgb.crop((x0, y0, x0 + side, y0 + side))
                top, ms = inf.demo.predict_topk(self.model, self.tf, crop_img, self.device, k=1)
            except Exception as e:  # jangan matikan stream karena satu frame gagal
                cv2.putText(bgr, f"error: {type(e).__name__}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(bgr, format="bgr24")
            idx, conf = top[0]
            locked = self.stab.update(idx, conf, time.perf_counter())
            shown = CLASSES[idx] if conf >= self.threshold else "?"
            with self.lock:
                if locked is not None:
                    self.word += CLASSES[locked]
                self.letter, self.conf, self.ms = shown, conf, ms
            cv2.rectangle(bgr, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
            cv2.putText(bgr, f"Huruf: {shown} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    return LiveProcessor
