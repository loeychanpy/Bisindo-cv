"""
dataset.py - Pipeline data BISINDO A-Z untuk CNN Transfer Learning (MobileNetV3-Small).

Isi:
  1. scan_images()      : scan folder dataset -> tabel (path, label, variant, group, source_id)
  2. build_manifest()   : scan + buang file bermasalah (hasil sanitary_check.py)
  3. make_splits()      : split train/val/test BERDASARKAN GRUP (subject/session) -> cegah data leakage
  4. get_transforms()   : preprocessing + augmentasi (ImageNet normalization, 224x224)
  5. BisindoDataset     : torch Dataset
  6. get_dataloaders()  : DataLoader + class weights

Contoh pemakaian CLI:
  python dataset.py --root data/raw --out data/splits.csv
  python dataset.py --root data/raw --variants background --group-regex "(subject_\\d+)"
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]          # A..Z
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG_SIZE = 224                                                      # input MobileNetV3-Small
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# kata kunci untuk mengenali varian citra dari path/nama file (sesuaikan bila struktur berbeda)
BINARY_KEYS = ("binary", "biner")
REMOVED_KEYS = ("removed", "remove", "nobg", "no_bg", "no-bg", "rembg", "tanpa")
BACKGROUND_KEYS = ("background", "bg")
VARIANT_TOKENS = ("background", "removed", "binary", "rembg", "nobg", "bg", "mask")

# pola default untuk mengenali subject/session/device dari path (opsional, bisa di-override --group-regex)
DEFAULT_GROUP_REGEX = (
    r"(?i)(?<![a-z])((?:subject|subjek|person|orang|user|participant|sesi|session|device|hp)"
    r"[_\- ]?\d+)"
)


# --------------------------------------------------------------------------- #
# Inferensi metadata dari path
# --------------------------------------------------------------------------- #
def infer_label(path: Path, root: Path) -> str | None:
    """Ambil huruf A-Z dari nama folder (prioritas) atau awalan nama file."""
    rel = path.relative_to(root)
    for part in reversed(rel.parts[:-1]):
        p = part.strip().upper()
        if p in CLASS_TO_IDX:
            return p
        m = re.fullmatch(r"(?:HURUF|CLASS|KELAS|LABEL)[_\- ]?([A-Z])", p)
        if m:
            return m.group(1)
    m = re.match(r"^([A-Za-z])(?=[^A-Za-z]|$)", path.stem)
    return m.group(1).upper() if m else None


def infer_variant(path: Path, root: Path) -> str:
    text = "/".join(path.relative_to(root).parts).lower()
    if "binary" in text or "biner" in text:
        return "binary"
    if "resized" in text:
        return "resized"
    if "original" in text:
        return "original"
    if "chain" in text:
        return "chaincode"
    return "unknown"


def normalize_stem(stem: str) -> str:
    """ID 'foto sumber' = nama file tanpa kata varian -> versi original/removed/binary dari foto
    yang sama diharapkan punya ID sama sehingga tidak terpisah antar split."""
    s = stem.lower()
    for tok in VARIANT_TOKENS:
        s = s.replace(tok, "")
    return re.sub(r"[^a-z0-9]", "", s)


def infer_group(path: Path, root: Path, regex: str | None) -> str | None:
    rel = "/".join(path.relative_to(root).parts)
    m = re.search(regex or DEFAULT_GROUP_REGEX, rel)
    if not m:
        return None
    return (m.group(1) if m.groups() else m.group(0)).lower()


# --------------------------------------------------------------------------- #
# Scan & manifest
# --------------------------------------------------------------------------- #
def scan_images(root: str | Path, group_regex: str | None = None) -> pd.DataFrame:
    """Scan rekursif semua file gambar. Tidak membuka file (cepat)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")

    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        label = infer_label(p, root)
        subject = infer_group(p, root, group_regex)
        source_id = f"{label}_{normalize_stem(p.stem)}"
        rows.append(
            dict(
                path=p.relative_to(root).as_posix(),
                label=label,
                label_idx=CLASS_TO_IDX.get(label, -1),
                variant=infer_variant(p, root),
                subject=subject,
                source_id=source_id,
            )
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Tidak ada gambar ({', '.join(sorted(IMG_EXTS))}) di {root}")
    return df


def load_exclusions(exclude_file: str | Path | None) -> set[str]:
    if exclude_file and Path(exclude_file).exists():
        return {l.strip() for l in Path(exclude_file).read_text(encoding="utf-8").splitlines() if l.strip()}
    return set()


def build_manifest(
    root: str | Path,
    variants: list[str] | None = None,
    group_regex: str | None = None,
    exclude_file: str | Path | None = "reports/excluded_files.txt",
) -> pd.DataFrame:
    """Manifest siap split. File di exclude_file (hasil sanitary_check) dibuang."""
    df = scan_images(root, group_regex)

    n0 = len(df)
    df = df[df["label_idx"] >= 0]
    if len(df) < n0:
        warnings.warn(f"{n0 - len(df)} file tanpa label A-Z valid dibuang.")

    excl = load_exclusions(exclude_file)
    if excl:
        n1 = len(df)
        df = df[~df["path"].isin(excl)]
        print(f"[manifest] {n1 - len(df)} file dibuang berdasarkan {exclude_file}")

    if variants:
        df = df[df["variant"].isin(variants)]
        if df.empty:
            raise RuntimeError(f"Tidak ada gambar dengan variant={variants}. Cek hasil sanitary_check.")

    # grup untuk split: subject/session bila terdeteksi, kalau tidak -> source_id
    df = df.copy()
    has_subject = df["subject"].notna()
    df["group"] = np.where(has_subject, df["subject"], df["source_id"])
    df["group_type"] = np.where(has_subject, "subject", "source_id")
    if has_subject.mean() < 0.5:
        warnings.warn(
            "Informasi subject/session TIDAK terdeteksi pada sebagian besar file. "
            "Split memakai source_id (mencegah kebocoran antar varian foto yang sama), "
            "tetapi orang yang sama masih bisa muncul di train & test. "
            "Gunakan --group-regex jika nama folder/file memuat info subjek."
        )
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Split berbasis grup
# --------------------------------------------------------------------------- #
def make_splits(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Tambahkan kolom 'split' in {train,val,test}. Satu grup tidak pernah terpecah antar split."""
    df = df.copy()
    groups = df["group"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, df["label_idx"], groups))
    trainval = df.iloc[trainval_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    tr_rel, va_rel = next(gss2.split(trainval, trainval["label_idx"], trainval["group"]))

    df["split"] = ""
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    df.loc[trainval.index[va_rel], "split"] = "val"
    df.loc[trainval.index[tr_rel], "split"] = "train"

    # ---- validasi ----
    sets = {s: set(df.loc[df["split"] == s, "group"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (
        sets["val"] & sets["test"]
    ), "Ada grup yang muncul di lebih dari satu split (leakage)!"
    for s in ("train", "val", "test"):
        missing = set(range(len(CLASSES))) - set(df.loc[df["split"] == s, "label_idx"])
        if missing:
            warnings.warn(f"Split '{s}' tidak memuat kelas: {[IDX_TO_CLASS[i] for i in sorted(missing)]}")
    return df


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def get_transforms(train: bool, img_size: int = IMG_SIZE) -> T.Compose:
    """Catatan: TIDAK memakai horizontal flip karena bisa mengubah makna isyarat (kiri/kanan)."""
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomRotation(10),                                     # rotasi ringan
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),   # blur ringan
                T.ToTensor(),
                norm,
            ]
        )
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])


# --------------------------------------------------------------------------- #
# Dataset & DataLoader
# --------------------------------------------------------------------------- #
class BisindoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")      # binary/grayscale -> 3 channel
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_idx"])


def compute_class_weights(df_train: pd.DataFrame) -> torch.Tensor:
    counts = np.bincount(df_train["label_idx"], minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1
    w = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32)


def get_dataloaders(
    root: str | Path,
    batch_size: int = 32,
    variants: list[str] | None = None,
    group_regex: str | None = None,
    splits_csv: str | Path | None = None,
    num_workers: int = 2,
    seed: int = 42,
    exclude_file: str | Path | None = "reports/excluded_files.txt",
):
    """Return (loaders: dict, class_weights: Tensor, df: DataFrame)."""
    if splits_csv and Path(splits_csv).exists():
        df = pd.read_csv(splits_csv)                                  # pakai split yang sudah tersimpan
    else:
        df = make_splits(build_manifest(root, variants, group_regex, exclude_file), seed=seed)
        if splits_csv:
            Path(splits_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(splits_csv, index=False)

    pin = torch.cuda.is_available()
    loaders = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        ds = BisindoDataset(sub, root, get_transforms(train=(s == "train")))
        loaders[s] = DataLoader(
            ds, batch_size=batch_size, shuffle=(s == "train"),
            num_workers=num_workers, pin_memory=pin, drop_last=False,
        )
    return loaders, compute_class_weights(df[df["split"] == "train"]), df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Bangun manifest & split dataset BISINDO")
    ap.add_argument("--root", required=True, help="folder dataset hasil ekstrak")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="pilih varian: background background_removed binary (default: semua)")
    ap.add_argument("--group-regex", default=None, help="regex (1 capture group) untuk subject/session")
    ap.add_argument("--exclude-file", default="reports/excluded_files.txt")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = build_manifest(a.root, a.variants, a.group_regex, a.exclude_file)
    df = make_splits(df, a.val_size, a.test_size, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\nTotal gambar : {len(df)}   |   jumlah grup: {df['group'].nunique()} ({df['group_type'].iloc[0]})")
    print("\nJumlah per split:\n", df["split"].value_counts().to_string())
    print("\nJumlah kelas per split:\n", pd.crosstab(df["label"], df["split"]).to_string())
    print(f"\nManifest disimpan ke {a.out}")


if __name__ == "__main__":
    main()
