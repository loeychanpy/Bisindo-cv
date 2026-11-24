"""diagnostics.py - log, info lingkungan, dan panduan untuk kegagalan koneksi live (WebRTC).

Tidak mengimpor streamlit/torch di tingkat modul agar mudah diuji dan aman dipakai dari thread video.

- Log ke `logs/app_live.log` (berputar, maks ~3 MB). Berisi versi paket, frame video pertama (tanda koneksi
  WebRTC terbentuk), kesalahan inferensi, dan status ICE dari aioice/aiortc. Kirim file ini bila minta bantuan.
- `ConnectionWatch`: mendeteksi "klik START tapi tidak ada video" dan memberi petunjuk, bukan spinner tanpa akhir.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
import threading
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "app_live.log"
LOG_REL = "logs/app_live.log"
LOGGER_NAME = "bisindo.app"
logger = logging.getLogger(LOGGER_NAME)

# Hanya modul ICE/DTLS yang dibuat rinci (DEBUG). Modul RTP/codec tetap senyap supaya tidak memberatkan CPU.
_ICE_LOGGERS = ("aioice", "aiortc.rtcicetransport", "aiortc.rtcdtlstransport", "aiortc.rtcpeerconnection")
_PACKAGES = ("aiortc", "aioice", "av", "pylibsrtp", "streamlit", "streamlit-webrtc", "torch", "mediapipe")

_lock = threading.Lock()
_handler: logging.Handler | None = None
_env_logged = False


def setup_logging(path: Path | str = LOG_PATH, force: bool = False) -> Path | None:
    """Pasang log file (idempoten). Return path log, atau None bila folder tidak bisa ditulis."""
    global _handler
    with _lock:
        if _handler is not None and not force:
            return Path(_handler.baseFilename)
        if _handler is not None:
            _detach_locked()
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            h = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        except OSError as e:
            logging.getLogger(LOGGER_NAME).warning("Log file tidak bisa dibuat (%s: %s); hanya log konsol.",
                                                   type(e).__name__, e)
            return None
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.setLevel(logging.INFO)
        logger.addHandler(h)
        for name in _ICE_LOGGERS:
            lg = logging.getLogger(name)
            lg.setLevel(logging.DEBUG)
            lg.addHandler(h)
        _handler = h
        return Path(h.baseFilename)


def _detach_locked() -> None:
    global _handler
    if _handler is None:
        return
    for name in (LOGGER_NAME, *_ICE_LOGGERS):
        logging.getLogger(name).removeHandler(_handler)
    _handler.close()
    _handler = None


def teardown_logging() -> None:
    with _lock:
        _detach_locked()


def package_versions() -> dict[str, str]:
    out = {}
    for pkg in _PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "tidak terpasang"
    for cv in ("opencv-python", "opencv-python-headless"):
        try:
            out[cv] = metadata.version(cv)
        except metadata.PackageNotFoundError:
            pass
    return out


def log_environment() -> None:
    """Catat versi paket dan sistem sekali per proses."""
    global _env_logged
    if _env_logged:
        return
    _env_logged = True
    vers = ", ".join(f"{k}={v}" for k, v in package_versions().items())
    logger.info("Python %s | %s | %s", sys.version.split()[0], platform.platform(), vers)


def environment_markdown() -> str:
    rows = "\n".join(f"{k:<24}{v}" for k, v in package_versions().items())
    return (f"**Rincian teknis** (sertakan bila minta bantuan):\n```\nPython {sys.version.split()[0]} | "
            f"{platform.system()} {platform.release()}\n{rows}\n```\n"
            f"Log lengkap (termasuk status ICE): `{LOG_REL}`")


# --------------------------------------------------------------------------- #
# Jalur koneksi (ICE)
# --------------------------------------------------------------------------- #
ICE_LOCAL = "Lokal (tanpa STUN) - bawaan"
ICE_STUN = "STUN Google (butuh internet)"
ICE_PRESETS: dict[str, list[dict]] = {
    ICE_LOCAL: [],
    ICE_STUN: [{"urls": ["stun:stun.l.google.com:19302"]}],
}
# Bisa dinyalakan tanpa mengubah kode:  set BISINDO_STUN=1  (Windows cmd) sebelum `streamlit run`.
DEFAULT_ICE = ICE_STUN if os.environ.get("BISINDO_STUN", "").strip().lower() in ("1", "true", "yes") else ICE_LOCAL


def rtc_configuration(choice: str) -> dict:
    servers = ICE_PRESETS.get(choice, [])
    return {"iceServers": [dict(s) for s in servers]}


# --------------------------------------------------------------------------- #
# Deteksi "START diklik tapi tidak ada video"
# --------------------------------------------------------------------------- #
SLOW_AFTER = 8.0  # detik tanpa frame video sebelum dianggap macet


class ConnectionWatch:
    """Pantau satu sesi live. `started` = pengguna menekan START; `frames` = jumlah frame video diterima server
    (hanya naik bila koneksi WebRTC benar-benar terbentuk)."""

    def __init__(self, slow_after: float = SLOW_AFTER):
        self.slow_after = slow_after
        self.t0: float | None = None
        self.f0 = 0
        self.connected = False
        self.fails = 0

    def update(self, started: bool, frames: int, now: float) -> None:
        if not started:
            # berhenti sebelum ada video setelah cukup lama = percobaan gagal (berhenti cepat = batal biasa)
            if self.t0 is not None and not self.connected and now - self.t0 >= self.slow_after:
                self.fails += 1
                logger.warning("Koneksi live gagal: berhenti setelah %.0f detik tanpa video (gagal ke-%d).",
                               now - self.t0, self.fails)
            self.t0, self.connected = None, False
            return
        if self.t0 is None:
            self.t0, self.f0, self.connected = now, frames, False
        if frames > self.f0 and not self.connected:
            self.connected, self.fails = True, 0

    def hint(self, now: float) -> tuple[str, str] | None:
        """Return (level, pesan) atau None. level: 'warning' | 'error'."""
        if self.t0 is not None and not self.connected and now - self.t0 >= self.slow_after:
            return ("warning",
                    f"Koneksi kamera live belum terbentuk setelah {now - self.t0:.0f} detik. Biasanya penyebabnya "
                    "firewall, VPN, atau jaringan, bukan modelnya. Buka panel **Bantuan koneksi live** di bawah, "
                    "atau pakai tab **Kamera**.")
        if self.t0 is None and self.fails:
            return ("error",
                    "Koneksi kamera live gagal: berhenti sebelum ada video. Buka panel **Bantuan koneksi live** "
                    "di bawah. Tab **Kamera** dan `python src/demo.py --run <model> --webcam` tetap bisa dipakai.")
        return None


GUIDE_MD = """\
Gejala: tulisan *"Connection is taking longer than expected"*, spinner terus berputar, lalu berhenti sendiri.
Artinya jalur WebRTC antara browser dan aplikasi gagal terbentuk. Coba berurutan:

1. **Windows Firewall.** *Windows Security → Firewall & network protection → Allow an app through firewall*,
   centang **python.exe** untuk **Private dan Public**. Tes cepat: matikan firewall sebentar, mulai ulang, klik START.
2. **VPN dan adapter virtual.** Matikan VPN; nonaktifkan sementara adapter WSL/Hyper-V/VirtualBox bila ada.
3. **Chrome menyembunyikan IP lokal.** Buka `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns`, ubah ke
   *Disabled*, mulai ulang Chrome. Atau coba Edge/Firefox.
4. **Ganti "Jalur koneksi"** di atas ke *STUN Google* (butuh internet), lalu klik START. Ubah sebelum START.
   Tanpa mengubah kode: jalankan dengan `set BISINDO_STUN=1` lebih dulu.
5. **Kamera.** Tutup aplikasi lain yang memakai kamera (Zoom/Meet/Teams) dan izinkan kamera di ikon address bar.
6. **Cadangan tanpa WebRTC:** tab **Kamera** (ambil foto), atau di terminal
   `python src/demo.py --run <nama_model> --webcam`.

Pesan *"Task was destroyed but it is pending"* di terminal hanya sisa penutupan koneksi dan tidak berbahaya.
"""
