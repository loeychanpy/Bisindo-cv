"""
error_analysis.py - Analisis kesalahan model dari predictions_<split>.csv sebuah run (CP11).

  python src/error_analysis.py --run v3_partial4_final --split test
  python src/error_analysis.py --run v3_partial4_crop --split test --config configs/config_v3.yaml

Butuh results/transfer/<run>/predictions_<split>.csv (ada setelah run dievaluasi pada split itu;
untuk run yang hanya dievaluasi di val: python src/train_transfer.py --eval-only --run <run>).

Keluaran di results/transfer/<run>/ :
  error_by_background.csv/.png   akurasi dan macro-F1 per bucket latar
  error_by_group.csv             akurasi per grup ponsel (grup di split ini saja); 5 terburuk dicetak
  error_top_pairs.csv            10 pasangan huruf paling tertukar
  error_pairs.png                4 contoh gambar tiap pasangan M<->N, D<->P, J<->I
  error_worst_group.png          contoh salah dari grup ponsel terburuk
  error_reliability.csv          tabel akurasi vs confidence + saran --threshold demo
  error_analysis.md              ringkasan (Indonesia): kategori kegagalan, hipotesis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("M", "N"), ("D", "P"), ("J", "I")]
BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _load(path: Path, size: int = 448) -> Image.Image:
    img = Image.open(path)
    img.draft("RGB", (size, size))
    img = img.convert("RGB")
    img.thumbnail((224, 224))
    return img


def _grid(items: list[tuple[Path, str, bool]], title: str, out: Path, cols: int = 4) -> None:
    """items: (path gambar, caption, benar?)."""
    if not items:
        return
    rows = int(np.ceil(len(items) / cols))
    fig = Figure(figsize=(cols * 2.4, rows * 2.7 + 0.5))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (p, cap, ok) in zip(axes, items):
        try:
            ax.imshow(_load(p))
        except OSError:
            ax.text(0.5, 0.5, "gagal dibuka", ha="center")
        ax.set_title(cap, fontsize=7, color="green" if ok else "red")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def _acc_f1(sub: pd.DataFrame) -> tuple[float, float]:
    acc = float((sub["true"] == sub["pred"]).mean())
    f1 = float(f1_score(sub["true"], sub["pred"], labels=sorted(set(sub["true"])), average="macro", zero_division=0))
    return acc, f1


def main():
    ap = argparse.ArgumentParser(description="Analisis kesalahan satu run")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--config", default="configs/config_v3.yaml")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()

    cfg = yaml.safe_load(_abs(a.config).read_text(encoding="utf-8"))
    root = _abs(cfg["data_dir"])
    out = _abs(a.results_dir) / "transfer" / a.run
    pred_csv = out / f"predictions_{a.split}.csv"
    if not pred_csv.exists():
        sys.exit(f"{pred_csv} tidak ada. Evaluasi dulu: python src/train_transfer.py --eval-only --run {a.run}")

    pred = pd.read_csv(pred_csv)
    split = pd.read_csv(_abs(cfg["split_path"]))
    if not {"group", "background"} <= set(split.columns):
        sys.exit("Split harus memuat kolom group dan background (gunakan split_v3).")
    df = pred.merge(split[["path", "group", "background"]], on="path", how="left")
    df["ok"] = df["true"] == df["pred"]
    n, acc_all = len(df), float(df["ok"].mean())
    f1_all = float(f1_score(df["true"], df["pred"], average="macro", zero_division=0))

    # 1. per latar
    rows = []
    for b, sub in df.groupby("background"):
        acc, f1 = _acc_f1(sub)
        rows.append(dict(background=b, n=len(sub), accuracy=acc, macro_f1=f1))
    bg = pd.DataFrame(rows).sort_values("accuracy")
    bg.to_csv(out / "error_by_background.csv", index=False)
    fig = Figure(figsize=(7, 3.8))
    ax = fig.subplots()
    x = np.arange(len(bg))
    ax.bar(x - 0.2, bg["accuracy"], 0.4, label="akurasi")
    ax.bar(x + 0.2, bg["macro_f1"], 0.4, label="macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={k})" for b, k in zip(bg["background"], bg["n"])], fontsize=8)
    ax.set(ylim=(0, 1.05), title=f"Per bucket latar - {a.run} ({a.split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_by_background.png", dpi=130)

    # 2. per grup ponsel
    gr = (df.groupby("group")["ok"].agg(n="size", accuracy="mean").reset_index().sort_values("accuracy"))
    gr.to_csv(out / "error_by_group.csv", index=False)
    worst = gr[gr["n"] >= 10].head(5)

    # 3. pasangan tertukar
    wrong = df[~df["ok"]]
    pairs = (wrong.groupby(["true", "pred"]).size().reset_index(name="jumlah").sort_values("jumlah", ascending=False))
    pairs.head(10).to_csv(out / "error_top_pairs.csv", index=False)
    items = []
    for u, v in PAIRS:
        sel = wrong[((wrong["true"] == u) & (wrong["pred"] == v)) | ((wrong["true"] == v) & (wrong["pred"] == u))]
        sel = sel.sort_values("conf", ascending=False).head(4)
        for _, r in sel.iterrows():
            items.append((root / r["path"], f"T:{r['true']} P:{r['pred']} ({r['conf']:.2f})\n{r['group']}", False))
    _grid(items, "Contoh salah paling yakin: M<->N, D<->P, J<->I (T = sebenarnya, P = prediksi)",
          out / "error_pairs.png")
    if len(worst):
        wg = worst.iloc[0]["group"]
        sel = wrong[wrong["group"] == wg].sort_values("conf", ascending=False).head(8)
        _grid([(root / r["path"], f"T:{r['true']} P:{r['pred']} ({r['conf']:.2f})", False) for _, r in sel.iterrows()],
              f"Grup terburuk: {wg}", out / "error_worst_group.png")

    # 4. reliability + saran threshold
    df["bin"] = pd.cut(df["conf"], BINS, right=False)
    rel = df.groupby("bin", observed=True).agg(n=("ok", "size"), accuracy=("ok", "mean"), mean_conf=("conf", "mean"))
    rel.reset_index().to_csv(out / "error_reliability.csv", index=False)
    thr = []
    for t in np.arange(0.5, 0.96, 0.05):
        kept = df[df["conf"] >= t]
        if len(kept):
            thr.append((round(float(t), 2), float(kept["ok"].mean()), len(kept) / n))
    sug = next((t for t, acc, cov in thr if acc >= 0.9 and cov >= 0.5), None)

    # 5. ringkasan
    top = pairs.head(10)
    hi_err = wrong[wrong["conf"] >= 0.8]
    md = [f"# Analisis kesalahan: {a.run} ({a.split})", "",
          f"- Gambar: {n} | akurasi {acc_all:.3f} | macro-F1 {f1_all:.3f} | salah: {len(wrong)}",
          f"- Grup ponsel di split ini: {df['group'].nunique()}", "",
          "## Kategori kegagalan", "",
          "### 1. Huruf yang bentuk tangannya mirip",
          "Pasangan tertukar terbanyak: " + ", ".join(f"{r.true}->{r.pred} ({r.jumlah})" for r in top.head(5).itertuples()) + ".",
          "Contoh gambar (salah paling yakin) untuk M/N, D/P, J/I: `error_pairs.png`. Hipotesis: perbedaan jari "
          "(mis. posisi ibu jari pada M vs N) kecil pada foto 3000 px yang diperkecil ke 224 px, sehingga detail hilang. "
          "Crop ke tangan (Phase 3) menargetkan kategori ini.", "",
          "### 2. Bergantung pada latar",
          "Akurasi per latar (rendah ke tinggi): " + ", ".join(f"{r.background} {r.accuracy:.2f} (n={r.n})" for r in bg.itertuples()) + ".",
          "Grafik: `error_by_background.png`. Latar ramai atau mirip warna kulit/baju menyulitkan model yang melihat "
          "seluruh gambar; hipotesis ini juga ditargetkan crop.", "",
          "### 3. Bergantung pada kontributor atau perangkat",
          "Lima grup terburuk (n>=10): " + (", ".join(f"{r.group} {r.accuracy:.2f} (n={r.n})" for r in worst.itertuples()) or "-") + ".",
          "Contoh: `error_worst_group.png`. Hipotesis: gaya tangan, jarak, dan kamera tiap kontributor berbeda dari data latih. "
          "Ini tidak ditangani crop; butuh lebih banyak ragam kontributor atau augmentasi yang lebih kuat.", "",
          "### 4. Salah dengan keyakinan tinggi",
          f"{len(hi_err)} dari {len(wrong)} kesalahan punya confidence >= 0.8. Model tidak bisa mengenali sendiri kapan ia salah, "
          "sehingga `--threshold` demo hanya menyaring sebagian.", "",
          "## Confidence vs akurasi", "", "| bin confidence | n | akurasi | rata-rata conf |", "|---|---|---|---|"]
    md += [f"| {i} | {r.n} | {r.accuracy:.3f} | {r.mean_conf:.3f} |" for i, r in rel.iterrows()]
    md += ["", "| threshold | akurasi di atas threshold | cakupan |", "|---|---|---|"]
    md += [f"| {t} | {acc:.3f} | {cov:.2f} |" for t, acc, cov in thr]
    md += ["", f"**Saran `--threshold` demo:** {sug if sug is not None else 'tidak ada threshold dengan akurasi >= 0.9 dan cakupan >= 50%; pakai 0.8 dan terima banyak ?'}"
           " (akurasi >= 0.9 dengan cakupan >= 50%).", ""]
    (out / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print(f"akurasi {acc_all:.4f} | macro-F1 {f1_all:.4f} | salah {len(wrong)}")
    print("latar terburuk:", bg.iloc[0]["background"], f"{bg.iloc[0]['accuracy']:.3f}")
    print("5 grup terburuk:", [(r.group, round(r.accuracy, 3), r.n) for r in worst.itertuples()])
    print("saran threshold:", sug)
    print(f"[OK] {out / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
