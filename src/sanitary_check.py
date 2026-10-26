"""
sanitary_check.py - Audit kebersihan dataset BISINDO sebelum training.

Yang diperiksa:
  - struktur: ekstensi file, file non-gambar, file tanpa label A-Z, kelas yang hilang
  - integritas: file rusak/tidak bisa dibuka, ukuran terlalu kecil, gambar kosong (hampir seragam)
  - distribusi: jumlah per kelas & per varian, rasio ketidakseimbangan, statistik resolusi & mode warna
  - duplikat: exact (MD5) dan near-duplicate (dHash), termasuk duplikat lintas kelas (indikasi label noise)
  - subject/session: apakah terdeteksi dari path (penting untuk split anti-leakage)

Output (folder --out, default reports/):
  sanitary_report.json, class_counts.csv, issues.csv, duplicates_exact.csv,
  near_duplicates.csv, excluded_files.txt  (dibaca otomatis oleh dataset.py)

Contoh:
  python sanitary_check.py --root data/raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dataset import CLASSES, IMG_EXTS, scan_images

try:
    from tqdm import tqdm
except ImportError:                                   # tqdm opsional
    def tqdm(x, **kw):
        return x

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Pemeriksaan per-file
# --------------------------------------------------------------------------- #
def dhash64(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()                       # 64 bit
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def check_file(args):
    root, rel, min_side, blank_std = args
    p = Path(root) / rel
    out = dict(path=rel, ok=True, issue=None, width=None, height=None, mode=None,
               fmt=None, bytes=p.stat().st_size, md5=None, dhash=None, std=None)
    try:
        out["md5"] = hashlib.md5(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            im.verify()                                           # cek struktur file
        with Image.open(p) as im:                                 # verify() menutup file -> buka lagi
            im.load()                                             # paksa decode penuh
            out.update(width=im.width, height=im.height, mode=im.mode, fmt=im.format)
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            out["std"] = float(gray.std())
            out["dhash"] = dhash64(im)
        if min(out["width"], out["height"]) < min_side:
            out.update(ok=False, issue=f"terlalu_kecil(<{min_side}px)")
        elif out["std"] < blank_std:
            out.update(ok=False, issue="gambar_kosong/seragam")
    except Exception as e:                                        # noqa: BLE001
        out.update(ok=False, issue=f"rusak:{type(e).__name__}")
    return out


# --------------------------------------------------------------------------- #
# Near-duplicate (dHash, Hamming distance) - dibandingkan per varian
# --------------------------------------------------------------------------- #
def near_duplicate_pairs(hashes: np.ndarray, thr: int, chunk: int = 256):
    pairs = []
    n = len(hashes)
    for i in range(0, n, chunk):
        x = hashes[i:i + chunk, None] ^ hashes[None, :]            # (c, n) uint64
        dist = POPCOUNT[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)
        rows, cols = np.nonzero(dist <= thr)
        for r, c in zip(rows, cols):
            if i + r < c:                                          # setiap pasangan sekali
                pairs.append((i + r, c, int(dist[r, c])))
    return pairs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Sanity check dataset BISINDO")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--min-side", type=int, default=32, help="sisi terpendek minimum (px)")
    ap.add_argument("--blank-std", type=float, default=1.0, help="std piksel di bawah ini = gambar kosong")
    ap.add_argument("--near", type=int, default=3, help="ambang Hamming dHash near-duplicate; -1 = lewati")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # ---------------- 1. struktur ---------------- #
    all_files = [p for p in root.rglob("*") if p.is_file()]
    ext_counts = Counter(p.suffix.lower() or "(tanpa ekstensi)" for p in all_files)
    non_images = [p.relative_to(root).as_posix() for p in all_files if p.suffix.lower() not in IMG_EXTS]
    report["total_files"] = len(all_files)
    report["extensions"] = dict(ext_counts)
    report["non_image_files"] = len(non_images)
    print(f"[1/5] {len(all_files)} file ditemukan | ekstensi: {dict(ext_counts)}")

    df = scan_images(root, a.group_regex)
    issues = [dict(path=p, issue="bukan_gambar") for p in non_images]
    unlabeled = df[df["label_idx"] < 0]
    issues += [dict(path=p, issue="label_A-Z_tidak_terdeteksi") for p in unlabeled["path"]]
    report["unlabeled_images"] = len(unlabeled)

    # ---------------- 2. integritas ---------------- #
    print(f"[2/5] Memeriksa integritas {len(df)} gambar ...")
    jobs = [(str(root), rel, a.min_side, a.blank_std) for rel in df["path"]]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(tqdm(ex.map(check_file, jobs), total=len(jobs)))
    info = pd.DataFrame(res)
    df = df.merge(info, on="path")
    bad = df[~df["ok"]]
    issues += [dict(path=r.path, issue=r.issue) for r in bad.itertuples()]
    report["corrupt_or_invalid"] = int(len(bad))
    report["issue_breakdown"] = dict(Counter(i["issue"].split("(")[0].split(":")[0] for i in issues))
    print(f"      bermasalah: {len(bad)}")

    good = df[df["ok"] & (df["label_idx"] >= 0)].copy()

    # ---------------- 3. distribusi ---------------- #
    print("[3/5] Distribusi kelas & varian ...")
    counts = pd.crosstab(good["label"], good["variant"]).reindex(CLASSES, fill_value=0)
    counts["TOTAL"] = counts.sum(axis=1)
    counts.to_csv(out / "class_counts.csv")
    tot = counts["TOTAL"]
    report["missing_classes"] = [c for c in CLASSES if tot[c] == 0]
    report["class_min"] = {"label": tot.idxmin(), "n": int(tot.min())}
    report["class_max"] = {"label": tot.idxmax(), "n": int(tot.max())}
    report["imbalance_ratio_max_min"] = round(float(tot.max() / max(tot.min(), 1)), 2)
    report["per_variant"] = good["variant"].value_counts().to_dict()
    report["image_modes"] = good["mode"].value_counts().to_dict()
    report["formats"] = good["fmt"].value_counts().to_dict()
    report["resolution"] = {
        "width": good["width"].describe().round(1).to_dict(),
        "height": good["height"].describe().round(1).to_dict(),
        "unique_sizes_top5": Counter(zip(good["width"], good["height"])).most_common(5),
    }
    print(counts.to_string())

    # ---------------- 4. duplikat ---------------- #
    print("[4/5] Mencari duplikat ...")
    exclude: set[str] = set(bad["path"]) | set(unlabeled["path"])

    exact_rows = []
    for md5, g in good.groupby("md5"):
        if len(g) < 2:
            continue
        g = g.sort_values("path")
        labels = set(g["label"])
        kind = "lintas_kelas" if len(labels) > 1 else "satu_kelas"
        for r in g.itertuples():
            exact_rows.append(dict(md5=md5, kind=kind, path=r.path, label=r.label, variant=r.variant))
        if kind == "lintas_kelas":
            exclude |= set(g["path"])                              # label ambigu -> buang semua
        else:
            exclude |= set(g["path"].iloc[1:])                     # simpan 1, buang sisanya
    pd.DataFrame(exact_rows).to_csv(out / "duplicates_exact.csv", index=False)
    n_cross = len({r["md5"] for r in exact_rows if r["kind"] == "lintas_kelas"})
    report["exact_duplicate_groups"] = len({r["md5"] for r in exact_rows})
    report["exact_duplicate_groups_cross_class"] = n_cross

    near_rows = []
    if a.near >= 0:
        for variant, g in good.groupby("variant"):
            g = g.reset_index(drop=True)
            if len(g) < 2:
                continue
            h = np.array(g["dhash"].astype(object).tolist(), dtype=np.uint64)
            for i, j, d in near_duplicate_pairs(h, a.near):
                ri, rj = g.iloc[i], g.iloc[j]
                if ri["md5"] == rj["md5"]:
                    continue                                       # sudah tercatat sebagai exact
                near_rows.append(dict(variant=variant, hamming=d,
                                      path_a=ri["path"], label_a=ri["label"], group_a=ri["subject"],
                                      path_b=rj["path"], label_b=rj["label"], group_b=rj["subject"]))
    nd = pd.DataFrame(near_rows)
    nd.to_csv(out / "near_duplicates.csv", index=False)
    report["near_duplicate_pairs"] = int(len(nd))
    if len(nd):
        report["near_duplicate_pairs_cross_class"] = int((nd["label_a"] != nd["label_b"]).sum())
    print(f"      exact: {report['exact_duplicate_groups']} grup "
          f"({n_cross} lintas kelas) | near-dup: {len(nd)} pasangan")

    # ---------------- 5. subject/session ---------------- #
    print("[5/5] Info subject/session ...")
    has_subj = good["subject"].notna()
    report["subject_detected_ratio"] = round(float(has_subj.mean()), 3)
    report["n_subjects_detected"] = int(good["subject"].nunique())
    report["n_source_ids"] = int(good["source_id"].nunique())
    if has_subj.mean() < 0.5:
        report["subject_note"] = (
            "Subject/session tidak terdeteksi dari path. Cek struktur folder/nama file, lalu "
            "jalankan ulang dengan --group-regex. Tanpa itu split hanya mencegah kebocoran antar "
            "varian foto yang sama, bukan antar orang."
        )
        print("      ! subject/session tidak terdeteksi")
    # contoh path per varian, untuk membantu verifikasi inferensi varian/label
    report["sample_paths"] = {v: g["path"].head(3).tolist() for v, g in good.groupby("variant")}

    # ---------------- simpan ---------------- #
    pd.DataFrame(issues).to_csv(out / "issues.csv", index=False)
    (out / "excluded_files.txt").write_text("\n".join(sorted(exclude)), encoding="utf-8")
    report["excluded_total"] = len(exclude)
    report["usable_images"] = int(len(good) - len(exclude & set(good["path"])))
    (out / "sanitary_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )

    print("\n================ RINGKASAN ================")
    for k in ("total_files", "non_image_files", "unlabeled_images", "corrupt_or_invalid",
              "missing_classes", "imbalance_ratio_max_min", "exact_duplicate_groups",
              "near_duplicate_pairs", "subject_detected_ratio", "excluded_total", "usable_images"):
        print(f"{k:28s}: {report.get(k)}")
    print(f"\nLaporan lengkap di: {out}/")


if __name__ == "__main__":
    main()
