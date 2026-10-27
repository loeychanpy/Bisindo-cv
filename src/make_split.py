"""
make_split.py  (CP07 - Orang 2)
Bikin split train/val/test SEKALI, simpan ke splits/split_<versi>.csv, lalu dibekukan.

Cara pakai (jalankan dari root repo bisindo-cv/):
    python src/make_split.py --inspect              # intip struktur dataset dulu
    python src/make_split.py                        # bikin splits/split_v1.csv
    python src/make_split.py --check-only           # cek ulang leakage split yang sudah ada
    python src/make_split.py --version v2           # kalau memang perlu split baru (catat alasannya di README)

Kunci yang dibaca dari configs/config.yaml (samain dgn punya Orang 1):
    seed: 42
    data_dir: "data/bisindo/01. Original Images"   # folder GAMBAR ASLI saja (jangan folder induk!)
    split_path: splits/split_v1.csv
    label_map_path: splits/label_map.json
    val_frac: 0.15
    test_frac: 0.15
    group_regex: null        # regex dgn named group (?P<group>...) buat ambil signer/latar dari path
Kunci opsional (ada default): image_exts, label_regex, group_mode (auto|global|per_class), hash_files
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------- seed
def set_seed(seed):
    """Satu seed buat semua: random, numpy, dan framework (kalau ada)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------- scan dataset
def infer_label(parts, label_regex):
    """Label = nama folder terdalam yang cuma 1 huruf (A-Z). Fallback: label_regex."""
    for d in reversed(parts[:-1]):
        if re.fullmatch(r"[A-Za-z]", d):
            return d.upper()
    if label_regex:
        m = re.search(label_regex, "/".join(parts))
        if m:
            return m.group("label").upper()
    return None


def scan_images(data_dir, exts, label_regex, group_regex):
    recs, skipped = [], []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()  # urutan tetap -> hasil deterministik
        for f in sorted(files):
            if Path(f).suffix.lower() not in exts:
                continue
            rel = (Path(root) / f).relative_to(data_dir).as_posix()
            label = infer_label(rel.split("/"), label_regex)
            if label is None:
                skipped.append(rel)
                continue
            group = None
            if group_regex:
                m = re.search(group_regex, rel)
                if m:
                    group = m.group("group")
            recs.append({"path": rel, "label": label, "group": group, "sha256": ""})
    return recs, skipped


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_hashes(recs, data_dir):
    for i, r in enumerate(recs, 1):
        try:
            r["sha256"] = sha256_file(Path(data_dir) / r["path"])
        except OSError as e:
            print(f"[WARN] gagal baca {r['path']}: {e}")
        if i % 2000 == 0:
            print(f"  hashing {i}/{len(recs)} ...")


# ----------------------------------------------------------------- grup (anti-leakage)
class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def assign_group_ids(recs):
    """
    Tiap gambar dapat group id. Kalau ada signer/latar -> pakai itu.
    Kalau nggak -> tiap gambar grup sendiri. File dengan hash SAMA digabung ke
    satu grup, jadi gambar duplikat nggak mungkin kepisah ke split berbeda.
    """
    uf = UnionFind()
    for r in recs:
        r["_gkey"] = f"g:{r['group']}" if r["group"] else f"p:{r['path']}"
        uf.find(r["_gkey"])
    by_hash = defaultdict(list)
    for r in recs:
        if r["sha256"]:
            by_hash[r["sha256"]].append(r["_gkey"])
    for keys in by_hash.values():
        for k in keys[1:]:
            uf.union(keys[0], k)
    for r in recs:
        r["_gid"] = uf.find(r["_gkey"])


def assign_to_splits(groups, ratios, rng):
    """groups: gid -> list index gambar. Greedy: grup terbesar dulu, masuk split yang paling 'kurang'."""
    total = sum(len(v) for v in groups.values())
    target = {s: total * ratios[s] for s in ratios}
    got = {s: 0 for s in ratios}
    gids = sorted(groups)
    rng.shuffle(gids)
    gids.sort(key=lambda g: -len(groups[g]))  # stable -> yang ukuran sama tetap acak
    out = {}
    for g in gids:
        s = max(ratios, key=lambda k: (target[k] - got[k]) / target[k])
        out[g] = s
        got[s] += len(groups[g])
    return out


def do_split(recs, ratios, seed, group_mode):
    rng = random.Random(seed)
    n_labels = len({r["label"] for r in recs})

    # tentukan mode
    has_group = any(r["group"] for r in recs)
    if group_mode == "auto":
        if has_group:
            g_labels = defaultdict(set)
            for r in recs:
                if r["group"]:
                    g_labels[r["_gid"]].add(r["label"])
            avg_cov = sum(len(v) for v in g_labels.values()) / len(g_labels) / n_labels
            group_mode = "global" if avg_cov >= 0.5 else "per_class"
            print(f"[INFO] group_mode=auto -> {group_mode} (rata-rata 1 grup nutup {avg_cov:.0%} kelas)")
        else:
            group_mode = "per_class"
    print(f"[INFO] mode split: {group_mode}")

    if group_mode == "global":
        # signer/latar dipisah total antar split (paling bersih dari leakage)
        groups = defaultdict(list)
        for i, r in enumerate(recs):
            groups[r["_gid"]].append(i)
        gsplit = assign_to_splits(groups, ratios, rng)
        for r in recs:
            r["split"] = gsplit[r["_gid"]]
    else:
        # stratifikasi per kelas: tiap kelas dibagi sendiri-sendiri
        by_label = defaultdict(list)
        for i, r in enumerate(recs):
            by_label[r["label"]].append(i)
        for label in sorted(by_label):
            groups = defaultdict(list)
            for i in by_label[label]:
                groups[recs[i]["_gid"]].append(i)
            gsplit = assign_to_splits(groups, ratios, rng)
            for i in by_label[label]:
                recs[i]["split"] = gsplit[recs[i]["_gid"]]
    return group_mode


# ----------------------------------------------------------------- cek leakage
def leakage_checks(recs):
    """Return (lines_laporan, jumlah_masalah)."""
    lines, problems = [], 0
    by_split = {s: [r for r in recs if r["split"] == s] for s in SPLITS}

    lines.append("== Jumlah gambar per kelas per split ==")
    labels = sorted({r["label"] for r in recs})
    lines.append("kelas  " + "  ".join(f"{s:>6}" for s in SPLITS))
    for lb in labels:
        c = [sum(1 for r in by_split[s] if r["label"] == lb) for s in SPLITS]
        flag = "   <-- KOSONG di salah satu split!" if 0 in c else ""
        if flag:
            problems += 1
        lines.append(f"{lb:<6} " + "  ".join(f"{x:>6}" for x in c) + flag)
    lines.append("total  " + "  ".join(f"{len(by_split[s]):>6}" for s in SPLITS))

    lines.append("\n== Irisan path antar split (harus 0) ==")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len({r["path"] for r in by_split[a]} & {r["path"] for r in by_split[b]})
        problems += n > 0
        lines.append(f"{a} & {b}: {n} {'OK' if n == 0 else 'BOCOR!'}")

    if any(r["group"] for r in recs):
        lines.append("\n== Irisan signer/latar antar split (0 = bersih) ==")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ga = {r["group"] for r in by_split[a] if r["group"]}
            gb = {r["group"] for r in by_split[b] if r["group"]}
            n = len(ga & gb)
            lines.append(f"{a} & {b}: {n} grup sama {'OK' if n == 0 else '(cek: wajar kalau mode per_class)'}")
    else:
        lines.append("\n[CATATAN] Metadata signer/latar nggak ada -> split cuma stratifikasi per kelas. "
                     "Tulis ini sebagai keterbatasan di README.")

    if any(r["sha256"] for r in recs):
        lines.append("\n== Duplikat file (hash sama) lintas split (harus 0) ==")
        h2s = defaultdict(set)
        for r in recs:
            if r["sha256"]:
                h2s[r["sha256"]].add(r["split"])
        n_dup = sum(1 for v in h2s.values() if len(v) > 1)
        n_inner = sum(1 for r in recs if r["sha256"]) - len(h2s)
        problems += n_dup > 0
        lines.append(f"hash yang muncul di >1 split: {n_dup} {'OK' if n_dup == 0 else 'BOCOR!'}")
        lines.append(f"(info) gambar duplikat di dalam split yang sama: {n_inner}")
    else:
        lines.append("\n[CATATAN] Hash file nggak dihitung -> cek duplikat dilewati.")
    return lines, problems


# ----------------------------------------------------------------- inspect
def inspect(data_dir, exts, label_regex, group_regex):
    data_dir = Path(data_dir)
    print(f"Folder: {data_dir.resolve()}")
    if not data_dir.exists():
        sys.exit("Folder nggak ketemu. Cek data_dir di config.yaml / hasil ekstrak zip.")
    print("\n-- isi 2 level pertama --")
    for p in sorted(data_dir.iterdir())[:30]:
        if p.is_dir():
            subs = sorted(x.name for x in p.iterdir())[:8]
            print(f"[D] {p.name}/  ({len(list(p.iterdir()))} item) contoh: {subs}")
        else:
            print(f"[F] {p.name}")
    recs, skipped = scan_images(data_dir, exts, label_regex, group_regex)
    ext_count = Counter()
    for root, _, files in os.walk(data_dir):
        for f in files:
            ext_count[Path(f).suffix.lower()] += 1
    print("\n-- ekstensi file --", dict(ext_count))
    print(f"gambar berlabel: {len(recs)} | kena skip (label nggak kebaca): {len(skipped)}")
    print("kelas ketemu:", sorted({r['label'] for r in recs}))
    print("per kelas:", dict(sorted(Counter(r['label'] for r in recs).items())))
    print("contoh path:", [r["path"] for r in recs[:5]])
    if skipped:
        print("contoh yang di-skip:", skipped[:5], "-> atur label_regex di config")
    if group_regex:
        gs = {r["group"] for r in recs if r["group"]}
        print(f"group_regex cocok ke {sum(1 for r in recs if r['group'])} gambar, {len(gs)} grup:", sorted(gs)[:10])
    else:
        print("group_regex belum diisi -> kalau ada signer/latar di nama folder/file, isi di config.")


# ----------------------------------------------------------------- csv io
COLS = ["path", "label", "group", "sha256", "split"]


def write_csv(recs, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in sorted(recs, key=lambda r: r["path"]):
            w.writerow([r["path"], r["label"], r["group"] or "", r["sha256"], r["split"]])


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_label_map(recs, path):
    """Label A-Z -> indeks 0-25, ditulis di SATU tempat (dipakai dataset.py Orang 3)."""
    labels = sorted({r["label"] for r in recs})
    mapping = {lb: i for i, lb in enumerate(labels)}
    if len(labels) != 26:
        print(f"[WARN] kelas yang ketemu {len(labels)}, bukan 26: {labels}")
    if path.exists():
        old = json.load(open(path, encoding="utf-8"))
        if old != mapping:
            print(f"[WARN] {path} sudah ada & BEDA dengan hasil scan, nggak ditimpa.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"[OK] label map: {path}")

def resolve_data_dir(data_dir):
    """Kalau path di config salah level, cari otomatis folder dengan nama yang sama di bawah data/."""
    p = Path(data_dir)
    if p.exists():
        return p
    hits = [x for x in Path("data").rglob("*") if x.is_dir() and x.name == p.name]
    if len(hits) == 1:
        print(f"[INFO] data_dir di config nggak ada, pakai: {hits[0]}")
        return hits[0]
    sys.exit(f"Folder '{data_dir}' nggak ketemu"
             + (f" dan ada {len(hits)} kandidat: {hits}" if hits else ".")
             + " Cek data_dir di config.yaml.")
    
# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = cfg.get("seed", 42)
    data_dir = resolve_data_dir(cfg["data_dir"])
    if "val_frac" in cfg or "test_frac" in cfg:
        val_f, test_f = cfg.get("val_frac", 0.15), cfg.get("test_frac", 0.15)
        ratios = {"train": round(1 - val_f - test_f, 6), "val": val_f, "test": test_f}
    else:
        ratios = cfg.get("split_ratio", {"train": 0.7, "val": 0.15, "test": 0.15})
    exts = {e.lower() for e in cfg.get("image_exts", DEFAULT_EXTS)}
    label_regex = cfg.get("label_regex")
    group_regex = cfg.get("group_regex")
    out = Path(cfg.get("split_path", f"splits/split_{args.version}.csv"))
    if args.version != "v1":  # split baru -> nama file ikut versi
        out = out.with_name(f"split_{args.version}.csv")
    splits_dir = out.parent
    report = splits_dir / f"leakage_report_{args.version}.txt"
    label_map_path = Path(cfg.get("label_map_path", splits_dir / "label_map.json"))

    if abs(sum(ratios.values()) - 1) > 1e-6:
        sys.exit(f"split_ratio harus total 1.0, sekarang {sum(ratios.values())}")

    if args.inspect:
        return inspect(data_dir, exts, label_regex, group_regex)

    if args.check_only:
        if not out.exists():
            sys.exit(f"{out} belum ada.")
        recs = read_csv(out)
        lines, problems = leakage_checks(recs)
        print("\n".join(lines))
        print(f"\nHASIL: {'BERSIH' if problems == 0 else str(problems) + ' MASALAH'}")
        return sys.exit(1 if problems else 0)

    # split dibekukan: nggak boleh ketimpa
    if out.exists():
        sys.exit(f"{out} sudah ada & dibekukan. Kalau perlu split baru: --version v2 (catat alasannya).")

    set_seed(seed)
    recs, skipped = scan_images(data_dir, exts, label_regex, group_regex)
    if not recs:
        sys.exit("Nggak ada gambar berlabel. Jalankan --inspect dulu.")
    print(f"[INFO] {len(recs)} gambar, {len({r['label'] for r in recs})} kelas, {len(skipped)} di-skip")
    if cfg.get("hash_files", True):
        print("[INFO] hitung sha256 (agak lama buat dataset gede) ...")
        add_hashes(recs, data_dir)

    assign_group_ids(recs)
    mode = do_split(recs, ratios, seed, cfg.get("group_mode", "auto"))
    write_csv(recs, out)
    write_label_map(recs, label_map_path)

    lines, problems = leakage_checks(recs)
    header = [f"split_{args.version} | seed={seed} | ratio={ratios} | mode={mode} | gambar={len(recs)}", ""]
    text = "\n".join(header + lines + [f"\nHASIL: {'BERSIH' if problems == 0 else str(problems) + ' MASALAH'}"])
    print("\n" + text)
    report.write_text(text, encoding="utf-8")
    print(f"\n[OK] tersimpan: {out}\n[OK] laporan  : {report}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
