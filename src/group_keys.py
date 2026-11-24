"""
group_keys.py - Turunkan ID kontributor (ponsel) dan bucket latar dari nama file untuk split_v3.

Nama file mengikuti pola <Huruf>_<Latar>_<Merek>_<Model>.jpg (mis. G_Hitam Polos_Vivo_V21.jpg).
Tiap ponsel memuat sekitar 104 gambar (26 huruf x 4 latar), jadi ponsel ~ satu kontributor.

Keluaran (default splits/groups_v3.csv): path,label,group,background,filename_letter,label_fixed
  - path    : relatif ke folder '01. Original Images' (sama dengan splits/split_v2.csv)
  - label   : huruf dari nama folder; DIGANTI huruf nama file bila keduanya berbeda (label_fixed=1)
  - group   : id ponsel ternormalisasi, atau ts:<tanggal> / img:IMG untuk file tanpa pola
  - background : polos_terang | polos_gelap | polos | bebas | baju | other

Contoh (dari root repo):
  python src/group_keys.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images"
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BRANDS = ("apple", "iphone", "iph", "infinix", "oppo", "poco", "realme", "redmi", "rog",
          "samsung", "sony", "vivo", "xiaomi", "note9")
MIN_GROUP = 20


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", s.lower())


def normalize_phone(fields: list[str]) -> str:
    """Gabungkan field ponsel lalu satukan alias merek / penamaan ganda."""
    s = "_".join(fields).lower()
    s = re.sub(r"\.(jpe?g|png)\b", "", s)             # 'oppo_a series.jpg' (ekstensi ganda)
    s = re.sub(r"\s*\(\d+\)", "", s)                  # penghitung duplikat '(2)', '(50)'
    s = re.sub(r"[^a-z0-9]", "", s)                    # buang spasi, '_', '+', dst.
    s = re.sub(r"jpg$", "", s)
    s = s.replace("peo", "pro") if s.startswith("redminote12pe") else s   # typo 'Note 12 Peo'
    if s.startswith("note9"):                          # 'Note 9' tanpa merek -> Redmi Note 9
        s = "redmi" + s
    s = re.sub(r"^apple", "", s)                       # 'appleiphone11' -> 'iphone11'
    s = re.sub(r"^iphiph", "iph", s)                   # 'iphiph15' -> 'iph15'
    s = re.sub(r"^iph(?!one)", "iphone", s)            # 'iph15' -> 'iphone15'
    s = re.sub(r"^samsunggalaxy", "samsung", s)
    s = re.sub(r"^xiaomi(redmi|poco)", r"\1", s)
    s = re.sub(r"^(samsunga\d\d)5g$", r"\1", s)        # 'A53 5G' = 'A53'; 'Galaxy A32 5G' = 'A32'
    return s


def normalize_background(fields: list[str]) -> str:
    s = _alnum("".join(fields))
    s = s.replace("backgorund", "").replace("background", "").replace("latar", "")
    if "baju" in s or s.startswith("body") or s in ("batik", "stipe", "stripe"):
        return "baju"
    if "bebas" in s:
        return "bebas"
    if "terang" in s or "putih" in s or s in ("wallwhite", "white"):
        return "polos_terang"
    if "gelap" in s or "hitam" in s:
        return "polos_gelap"
    if "polos" in s or s.startswith("wall"):
        return "polos"
    return "other"


def parse_name(stem: str):
    """Return (filename_letter, group, background, raw_phone, pattern_ok)."""
    stem = re.sub(r"^salinan\s+", "", stem, flags=re.I)
    m = re.fullmatch(r"(\d{8})_\d{6}", stem)
    if m:
        return None, f"ts:{m.group(1)}", "other", stem, False
    m = re.fullmatch(r"IMG_(\d{8})_\d{6}", stem)
    if m:
        return None, f"ts:{m.group(1)}", "other", stem, False
    if re.fullmatch(r"IMG_\d+", stem):
        return None, "img:IMG", "other", stem, False
    parts = stem.split("_")
    if len(parts) < 3 or len(parts[0].strip()) != 1:
        return None, f"misc:{_alnum(stem)}", "other", stem, False
    letter = parts[0].strip().upper()
    letter = letter if letter.isalpha() else None      # '0_BEBAS_Apple_Iphone 12': huruf salah ketik, pakai folder
    if letter is None:
        parts[0] = "x"
    idx = next((i for i in range(1, len(parts)) if _alnum(parts[i]).startswith(BRANDS)), None)
    if idx is None:
        return letter, f"unk:{_alnum('_'.join(parts[2:]))}", normalize_background(parts[1:2]), "_".join(parts[1:]), False
    raw_phone = "_".join(parts[idx:])
    return letter, normalize_phone(parts[idx:]), normalize_background(parts[1:idx]), raw_phone, True


def load_exclusions(exclude_file: str | None, root: Path) -> set[str]:
    """excluded_files.txt memuat path relatif ke folder induk varian ('01. Original Images/A/x.jpg')."""
    if not exclude_file or not Path(exclude_file).exists():
        return set()
    prefix = root.name + "/"
    lines = Path(exclude_file).read_text(encoding="utf-8").splitlines()
    return {l.strip()[len(prefix):] for l in lines if l.strip().startswith(prefix)}


def build(root: Path, exclude: set[str]):
    rows, mapping = [], defaultdict(Counter)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        folder_label = d.name.strip().upper()
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMG_EXTS or f.name.startswith("._"):
                continue
            rel = f"{d.name}/{f.name}"
            if rel in exclude:
                continue
            letter, group, bg, raw, ok = parse_name(f.stem)
            fixed = int(letter is not None and letter != folder_label)
            rows.append(dict(path=rel, label=letter if fixed else folder_label, group=group, background=bg,
                             filename_letter=letter or "", label_fixed=fixed))
            mapping[group][raw.strip()] += 1
    return rows, mapping


def main():
    ap = argparse.ArgumentParser(description="Turunkan group (ponsel) dan background dari nama file")
    ap.add_argument("--root", required=True, help="folder '01. Original Images'")
    ap.add_argument("--exclude", default="results/sanitary/excluded_files.txt")
    ap.add_argument("--out", default="splits/groups_v3.csv")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir():
        sys.exit(f"Folder tidak ditemukan: {root}")
    rows, mapping = build(root, load_exclusions(a.exclude, root))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "group", "background", "filename_letter", "label_fixed"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["path"]))

    sizes = Counter(r["group"] for r in rows)
    v = sorted(sizes.values())
    print(f"gambar: {len(rows)} | grup: {len(sizes)} | gambar per grup min/median/max: "
          f"{v[0]}/{statistics.median(v):g}/{v[-1]}")
    print("\n== Pemetaan nama mentah -> grup (cek penggabungan) ==")
    for g in sorted(mapping):
        raws = ", ".join(f"{k!r}x{n}" for k, n in sorted(mapping[g].items()))
        print(f"{g:<28} {sizes[g]:>4}  <- {raws}")
    small = [(g, n) for g, n in sorted(sizes.items()) if n < MIN_GROUP]
    print(f"\n== Grup < {MIN_GROUP} gambar ({len(small)}) ==")
    for g, n in small:
        print(f"  {g}: {n}")
    print("\n== Distribusi background ==", dict(Counter(r["background"] for r in rows)))
    fixed = [r for r in rows if r["label_fixed"]]
    print(f"\n== Label diperbaiki ({len(fixed)}) ==")
    for r in fixed:
        print(f"  {r['path']}: folder {r['path'][0]} -> {r['label']}")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
