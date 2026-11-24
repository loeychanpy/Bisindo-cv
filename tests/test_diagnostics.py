"""Tes diagnostics.py (tanpa streamlit/torch): deteksi koneksi live macet, preset ICE, log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import diagnostics as diag  # noqa: E402


def test_slow_connection_warns_then_fails():
    w = diag.ConnectionWatch(slow_after=8)
    w.update(True, 0, 100.0)
    assert w.hint(103.0) is None                      # masih wajar menunggu
    lvl, msg = w.hint(109.0)
    assert lvl == "warning" and "Bantuan koneksi live" in msg
    w.update(False, 0, 120.0)                         # berhenti sendiri tanpa video
    assert w.fails == 1
    lvl, msg = w.hint(121.0)
    assert lvl == "error" and "**Kamera**" in msg


def test_quick_stop_is_not_a_failure():
    w = diag.ConnectionWatch(slow_after=8)
    w.update(True, 0, 10.0)
    w.update(False, 0, 12.0)                          # pengguna menekan STOP cepat
    assert w.fails == 0 and w.hint(13.0) is None


def test_frames_clear_the_warning_and_previous_failures():
    w = diag.ConnectionWatch(slow_after=8)
    w.update(True, 5, 0.0)
    w.update(False, 5, 30.0)
    assert w.fails == 1
    w.update(True, 5, 40.0)                           # percobaan baru
    assert w.hint(60.0)[0] == "warning"
    w.update(True, 9, 61.0)                           # frame baru masuk = tersambung
    assert w.connected and w.fails == 0 and w.hint(62.0) is None
    w.update(False, 200, 90.0)                        # berhenti normal setelah tersambung
    assert w.fails == 0 and w.hint(91.0) is None


def test_old_frames_from_previous_session_do_not_count():
    w = diag.ConnectionWatch(slow_after=8)
    w.update(True, 1000, 0.0)                         # penghitung global sudah besar dari sesi lama
    assert not w.connected
    assert w.hint(20.0)[0] == "warning"


def test_ice_presets_are_independent_copies():
    cfg = diag.rtc_configuration(diag.ICE_STUN)
    assert cfg["iceServers"][0]["urls"][0].startswith("stun:")
    cfg["iceServers"][0]["urls"] = ["x"]
    assert diag.rtc_configuration(diag.ICE_STUN)["iceServers"][0]["urls"][0].startswith("stun:")
    assert diag.rtc_configuration(diag.ICE_LOCAL) == {"iceServers": []}
    assert diag.rtc_configuration("tidak ada") == {"iceServers": []}


def test_package_versions_lists_webrtc_stack():
    v = diag.package_versions()
    for k in ("aiortc", "aioice", "av", "streamlit-webrtc"):
        assert k in v


def test_logging_writes_file_and_is_idempotent(tmp_path):
    p = tmp_path / "sub" / "x.log"
    try:
        got = diag.setup_logging(p, force=True)
        assert got == p and diag.setup_logging(p) == p
        diag.logger.info("halo")
        logging.getLogger("aioice").debug("ice-debug")
        for h in logging.getLogger("aioice").handlers:
            h.flush()
        text = p.read_text(encoding="utf-8")
        assert "halo" in text and "ice-debug" in text and text.count("halo") == 1
    finally:
        diag.teardown_logging()


def test_logging_survives_unwritable_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    try:
        assert diag.setup_logging(blocker / "x.log", force=True) is None   # folder induk adalah file
    finally:
        diag.teardown_logging()
