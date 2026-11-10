"""Aplikasi Streamlit lokal untuk demo BISINDO A-Z.  Jalankan: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="BISINDO A–Z", layout="wide")
st.title("Pengenalan Alfabet BISINDO (A–Z)")
st.caption("Kelompok 4 · Visi Komputer IF0PP006")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with st.spinner("Memuat PyTorch dan model (pertama kali bisa ~20 detik) ..."):
    import pandas as pd  # noqa: E402
    import torch  # noqa: E402
    from app import inference as inf  # noqa: E402
    from app import ui_parts as ui  # noqa: E402
    from dataset import CLASSES  # noqa: E402
    from train_transfer import load_rgb  # noqa: E402

ROOT = inf.ROOT
ui.inject_css()

st.session_state.setdefault("word", "")
st.session_state.setdefault("sample_seed", 0)
st.session_state.setdefault("cur_letter", "?")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
runs = inf.list_runs()
with st.sidebar:
    st.header("Pengaturan")
    if st.button("Muat ulang daftar model"):
        inf._scan.clear()
        st.rerun()
    if not runs:
        st.error("Belum ada model. Latih dulu dengan `python src/train_transfer.py` "
                 "(butuh `best.pth` dan `config_used.json` di `results/transfer/<run>/`; "
                 "`best.pth` tidak di-commit).")
        st.stop()
    names = [r["name"] for r in runs]
    run_name = st.selectbox("Model", names, format_func=lambda n: inf.run_label(runs[names.index(n)]))
    device_name = st.selectbox("Perangkat", inf.device_options())

    def _suggested_threshold(run: str) -> float:
        p = inf.run_dir(run) / "error_analysis.md"
        if p.exists():
            m = re.search(r"threshold[^0-9\n]{0,40}(0?\.\d+|1\.0)", p.read_text(encoding="utf-8"), re.I)
            if m:
                return min(1.0, float(m.group(1)))
        return 0.5

    threshold = st.slider("Threshold keyakinan", 0.0, 1.0, _suggested_threshold(run_name), 0.05,
                          key=f"thr_{run_name}")
    topk_n = st.slider("Top-k", 1, 5, 3)
    crop_ok, crop_why = inf.crop_available()
    use_crop = st.checkbox("Potong ke tangan (MediaPipe)", value=False, disabled=not crop_ok,
                           help=crop_why or "Deteksi tangan lalu potong sebelum klasifikasi.")
    if not crop_ok:
        st.caption(f"Fitur potong tangan nonaktif: {crop_why}")

try:
    model, tf, cfg = inf.get_model(run_name, device_name)
except Exception as e:
    st.error(f"Model `{run_name}` gagal dimuat ({type(e).__name__}: {e}).")
    st.stop()
device = torch.device(device_name)
split_path = cfg.get("split_path", "")
with st.sidebar:
    st.info(f"**Mode:** {cfg.get('mode')}  \n**Unfreeze blocks:** {cfg.get('unfreeze_blocks')}  \n"
            f"**img_size:** {cfg.get('img_size')}  \n**Epoch terbaik:** {cfg.get('_best_epoch')}  \n"
            f"**Split:** {Path(split_path).name}")
if use_crop and cfg.get("crop", "none") == "none":
    st.sidebar.warning("Model ini dilatih tanpa crop; memotong tangan mengubah distribusi masukan.")


def classify(img):
    """Prediksi aman: kembalikan (hasil, pesan_error)."""
    try:
        return inf.predict(model, tf, img, device, k=topk_n, threshold=threshold, crop=use_crop), None
    except (inf.InputImageError, inf.CropUnavailable) as e:
        return None, str(e)
    except Exception as e:
        return None, f"Gagal memprediksi ({type(e).__name__}: {e})"


# --------------------------------------------------------------------------- #
# Cache prediksi data uji
# --------------------------------------------------------------------------- #
def _split_md5(path: str) -> str:
    p = ROOT / path
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else ""


def _test_rows(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["split"] == "test"].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def predict_paths(run: str, dev: str, split_md5: str, data_dir: str, paths: tuple) -> dict:
    """path -> (indeks top-1, prob top-1). Kunci cache: run + perangkat + md5 split + daftar path."""
    mdl, tfm, _ = inf.get_model(run, dev)
    out = {}
    for p in paths:
        try:
            img = load_rgb(ROOT / data_dir / p, 448)
        except (OSError, FileNotFoundError):
            continue
        top, _ = inf.demo.predict_topk(mdl, tfm, img, torch.device(dev), k=1)
        out[p] = top[0]
    return out


def all_test_preds(rows: pd.DataFrame, bar) -> dict:
    """Prediksi seluruh data uji per potongan 64 (progress bar jalan; tiap potongan di-cache)."""
    md5, out, paths = _split_md5(split_path), {}, list(rows["path"])
    for i in range(0, len(paths), 64):
        out.update(predict_paths(run_name, device_name, md5, cfg["data_dir"], tuple(paths[i:i + 64])))
        done = min(i + 64, len(paths))
        bar.progress(done / len(paths), text=f"Memprediksi data uji {done}/{len(paths)}")
    bar.empty()
    return out


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_live, tab_up, tab_cam, tab_test, tab_res, tab_about = st.tabs(
    ["Live", "Unggah Gambar", "Kamera", "Contoh Data Uji", "Hasil Model", "Tentang"])

with tab_live:
    from app import live

    live_ok, live_why = live.live_available()
    if not live_ok:
        st.warning(f"Mode live nonaktif: {live_why}")
    else:
        from streamlit_webrtc import webrtc_streamer

        st.caption("Prediksi langsung dari video kamera. Letakkan tangan di dalam kotak hijau "
                   "(tengah bingkai). Huruf yang stabil otomatis masuk ke kata.")
        live_mirror = st.checkbox("Cerminkan gambar (live)", value=False,
                                  help="Model dilatih tanpa flip; aktifkan hanya bila gambar kamera terbalik.")
        cropper, crop_err = None, None
        if use_crop:
            try:
                cropper = inf.get_cropper()
            except inf.CropUnavailable as e:
                crop_err = str(e)
        if crop_err:
            st.error(crop_err)
        ctx = webrtc_streamer(
            key="live", video_processor_factory=live.make_processor_class(),
            rtc_configuration={"iceServers": []},  # lokal saja, tanpa STUN
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        proc = ctx.video_processor
        if proc is not None:
            proc.configure(model, tf, device, cropper, threshold, live_mirror)

        @st.fragment(run_every=0.5)
        def live_panel():
            if proc is None:
                st.info("Klik **START**, lalu izinkan akses kamera di browser.")
                return
            word, letter, conf, ms = proc.state()
            c1, c2 = st.columns(2)
            c1.metric("Huruf saat ini", letter, f"{conf:.0%} · {ms:.0f} ms", delta_color="off")
            c2.markdown(f'<div class="big-word">{word or "&nbsp;"}</div>', unsafe_allow_html=True)

        live_panel()
        if proc is not None:
            b1, b2, b3 = st.columns(3)
            if b1.button("Spasi", key="live_space", width="stretch"):
                proc.edit_word("space")
            if b2.button("Hapus", key="live_back", width="stretch"):
                proc.edit_word("back")
            if b3.button("Reset", key="live_reset", width="stretch"):
                proc.edit_word("reset")

with tab_up:
    files = st.file_uploader("Unggah foto isyarat (JPG/PNG)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if not files:
        st.info("Unggah satu atau lebih foto untuk diprediksi.")
    else:
        results = []
        for f in files:
            try:
                img = inf.prepare_image(f, draft=1024 if use_crop else 448)
                res, err = classify(img)
            except inf.InputImageError as e:
                img, res, err = None, None, str(e)
            results.append((f.name, img, res, err))
        lat = [r["latency_ms"] for _, _, r, _ in results if r and r["hand_found"] is not False]
        st.markdown(f"**{len(files)} gambar** · rata-rata latensi **{(sum(lat) / len(lat)) if lat else 0:.1f} ms** "
                    f"· {sum(1 for r in results if r[3])} gagal")
        cols = st.columns(3)
        for i, (name, img, res, err) in enumerate(results):
            with cols[i % 3], st.container(border=True):
                st.markdown(f"**{name}**")
                if err:
                    st.error(err)
                    continue
                st.image(ui.draw_box(img, res["crop_box"]), width="stretch")
                ui.prediction_card(res, threshold)

with tab_cam:
    mirror = st.checkbox("Cerminkan gambar", value=False)
    st.caption("Model dilatih tanpa flip horizontal, jadi kiri/kanan berpengaruh. "
               "Aktifkan hanya bila kamera Anda menghasilkan gambar terbalik.")
    left, right = st.columns(2)
    with left:
        shot = st.camera_input("Ambil foto isyarat")
    st.session_state.cur_letter = "?"
    with right:
        if shot is None:
            st.info("Ambil foto dari kamera. Bila kamera tidak tersedia, pakai tab *Unggah Gambar*.")
        else:
            try:
                img = inf.prepare_image(shot, mirror=mirror, draft=1024 if use_crop else 448)
                res, err = classify(img)
            except inf.InputImageError as e:
                res, err = None, str(e)
            if err:
                st.error(err)
            else:
                st.session_state.cur_letter = res["letter"]
                if res["crop_box"]:
                    st.image(ui.draw_box(img, res["crop_box"]), width="stretch")
                ui.prediction_card(res, threshold)

    st.subheader("Susun kata")

    def _add():
        if st.session_state.cur_letter not in ("?", "—"):
            st.session_state.word += st.session_state.cur_letter

    b1, b2, b3, b4 = st.columns(4)
    b1.button("Tambah huruf", on_click=_add, width="stretch")
    b2.button("Spasi", on_click=lambda: st.session_state.update(word=st.session_state.word + " "),
              width="stretch")
    b3.button("Hapus", on_click=lambda: st.session_state.update(word=st.session_state.word[:-1]),
              width="stretch")
    b4.button("Reset", on_click=lambda: st.session_state.update(word=""), width="stretch")
    st.markdown(f'<div class="big-word">{st.session_state.word or "&nbsp;"}</div>', unsafe_allow_html=True)
    st.caption("Video real-time tetap lewat terminal: `python src/demo.py --run <run> --webcam`.")

with tab_test:
    rows = _test_rows(split_path)
    if rows.empty:
        st.warning(f"File split `{split_path}` tidak ditemukan atau tidak berisi data uji.")
    elif not (ROOT / cfg["data_dir"]).exists():
        st.warning("Folder dataset tidak ditemukan di laptop ini, contoh data uji tidak bisa ditampilkan.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        letter = c1.selectbox("Huruf", ["Semua"] + CLASSES)
        if c2.button("Acak 9 gambar"):
            st.session_state.sample_seed += 1
        only_wrong = c3.checkbox("Hanya tampilkan yang salah")
        pool = rows if letter == "Semua" else rows[rows["label"] == letter]
        preds = None
        if only_wrong:
            preds = all_test_preds(pool, st.progress(0.0, text="Memprediksi data uji ..."))
            truth = dict(zip(pool["path"], pool["label"]))
            wrong = {p for p, (i, _) in preds.items() if CLASSES[i] != truth[p]}
            pool = pool[pool["path"].isin(wrong)]
        if pool.empty:
            st.info("Tidak ada gambar yang cocok dengan filter.")
        else:
            pick = pool.sample(n=min(9, len(pool)), random_state=st.session_state.sample_seed)
            pr = preds or predict_paths(run_name, device_name, _split_md5(split_path), cfg["data_dir"],
                                        tuple(pick["path"]))
            correct, shown, grid = 0, 0, st.columns(3)
            for _, r in pick.iterrows():
                if r["path"] not in pr:
                    continue
                idx, conf = pr[r["path"]]
                ok = CLASSES[idx] == r["label"]
                correct += ok
                with grid[shown % 3], st.container(border=True):
                    st.image(ui.border(load_rgb(ROOT / cfg["data_dir"] / r["path"], 448), ok),
                             width="stretch")
                    st.markdown(f"Asli: **{r['label']}** · Prediksi: **{CLASSES[idx]}** ({conf:.0%})")
                shown += 1
            st.markdown(f"### {correct}/{shown} benar")

with tab_res:
    rd = inf.run_dir(run_name)
    m = inf._read_json(rd / "metrics.json")
    if not m:
        st.warning("Run ini belum punya `metrics.json` (belum dievaluasi).")
    else:
        which = "test" if m.get("test") else "val"
        part = m.get(which) or {}
        st.caption(f"Metrik pada split **{which}**")
        c = st.columns(4)
        c[0].metric("Akurasi", f"{part.get('accuracy', float('nan')):.3f}")
        c[1].metric("Macro-F1", f"{part.get('macro_f1', float('nan')):.3f}")
        c[2].metric("Weighted-F1", f"{part.get('weighted_f1', float('nan')):.3f}")
        lat = (m.get("latency_ms") or {}).get("total_ms_median")
        c[3].metric("Latensi (median)", f"{lat:.1f} ms" if lat else "—")
    imgs = [rd / "curves.png"] + sorted(rd.glob("confusion_matrix_*.png")) + sorted(rd.glob("sample_predictions_*.png"))
    for p in [p for p in imgs if p.exists()]:
        st.image(str(p), caption=p.name, width="stretch")
    for pat, title, sort in (("per_class_*.csv", "Per kelas (F1 terendah dulu)", "f1"),
                             ("top_confusions_*.csv", "Kebingungan teratas", None)):
        for p in sorted(rd.glob(pat)):
            st.subheader(title)
            df = pd.read_csv(p)
            st.dataframe(df.sort_values(sort) if sort and sort in df else df,
                         width="stretch", hide_index=True)
    comp = ROOT / "results" / "transfer" / "comparison.csv"
    if comp.exists():
        st.subheader("Perbandingan semua run")
        cdf = pd.read_csv(comp)
        st.dataframe(cdf.style.apply(
            lambda r: ["background-color: #fff3b0; color: #000" if r.iloc[0] == run_name else "" for _ in r], axis=1),
            width="stretch", hide_index=True)
    for p in (ROOT / "results" / "transfer" / "final_table.md", rd / "error_analysis.md"):
        if p.exists():
            st.markdown(p.read_text(encoding="utf-8"))

with tab_about:
    st.markdown("""
**Masalah.** Mengenali huruf isyarat BISINDO (A–Z) dari satu foto tangan.

**Dataset.** Sanjaya, UMN 2024, Mendeley Data — DOI 10.17632/ywnjpbcz8m.1, lisensi CC BY 4.0.

**Model.** MobileNetV3-Small dengan bobot ImageNet pra-latih (transfer learning).
""")
    st.markdown(f"**Split yang dipakai model terpilih:** `{split_path}`")
    st.subheader("Keterbatasan")
    st.markdown("""
- Model hanya mengenali **huruf statis A–Z**, bukan kata atau gerakan.
- Tanpa potong tangan, model **selalu menebak satu huruf**, bahkan bila tidak ada tangan di gambar.
- Hasil bergantung pada pencahayaan dan latar belakang.
- Ini demo akademik, **bukan alat produksi atau alat bantu aksesibilitas**.
""")
    if Path(split_path).stem == "split_v2":
        st.warning("Model ini memakai `split_v2`: kontributor (perangkat dan latar) tumpang tindih antar split, "
                   "sehingga skor test bisa terlalu optimis dibanding foto dari HP atau latar baru.")
