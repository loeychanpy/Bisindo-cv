"""
train_transfer.py - Transfer learning MobileNetV3-Small untuk BISINDO A-Z.

Yang dibaca:
  configs/config.yaml    : data_dir, split_path (split beku), label_map_path, results_dir, seed, img_size
  configs/transfer.yaml  : hyperparameter eksperimen (bisa ditimpa lewat argumen CLI)
Skrip ini TIDAK membuat split. Semua eksperimen memakai split beku yang sama.

Mode:
  head     backbone beku, hanya Linear terakhir (kepala baru) yang dilatih   -> baseline
  partial  N blok terakhir `features` + seluruh `classifier` dilatih         -> metode utama
  full     semua layer dilatih

Contoh (folder kerja bebas, path dibaca relatif ke root repo):
  python src/train_transfer.py
  python src/train_transfer.py --mode partial --unfreeze-blocks 3 --run-name mnv3s_partial3
  python src/train_transfer.py --run-name tuning1 --no-test      # tuning: test tidak disentuh
  python src/train_transfer.py --compare                         # tabel semua run

Keluaran di results/transfer/<run_name>/ :
  config_used.json  log.csv  curves.png  best.pth (di-.gitignore)  metrics.json
  per_class_<split>.csv  confusion_matrix_<split>.csv/.png  top_confusions_<split>.csv
  predictions_<split>.csv  sample_predictions_<split>.png
  (<split> = test, atau val jika dijalankan dengan --no-test)

Aturan yang dijaga skrip:
  - augmentasi hanya di train (tanpa horizontal flip, lihat dataset.get_transforms)
  - BatchNorm pada layer beku tetap mode eval, jadi backbone beku benar-benar tidak berubah
  - model terbaik dipilih dari val macro-F1; test dievaluasi sekali di akhir
  - run yang sudah ada tidak ditimpa kecuali overwrite=True / --overwrite
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))  # supaya `import dataset` jalan dari notebook juga
from dataset import CLASSES, BisindoDataset, compute_class_weights, get_transforms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODES = ("head", "partial", "full")
SPLITS = ("train", "val", "test")

DEFAULTS = dict(
    run_name="mnv3s_head", mode="head", unfreeze_blocks=2, pretrained=True,
    batch_size=64, epochs=12, lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4,
    class_weights=False, augment=True, patience=4, num_workers=0, draft_size=256,
    cache_ram=True, device="auto", overwrite=False,
)

# Cache gambar di RAM lintas-epoch dan lintas-run supaya tidak baca ulang ribuan foto besar dari disk/OneDrive
_IMAGE_CACHE: dict[tuple[str, int], Image.Image] = {}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _load_yaml(p) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def build_cfg(overrides: dict | None = None, config: str = "configs/config.yaml",
              transfer_config: str = "configs/transfer.yaml") -> dict:
    """Gabungkan DEFAULTS < transfer.yaml < overrides. Path dari config.yaml dijadikan absolut."""
    base = _load_yaml(_abs(config))
    cfg = dict(DEFAULTS)
    tp = _abs(transfer_config)
    if tp.exists():
        cfg.update(_load_yaml(tp))
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})

    cfg["seed"] = int(cfg.get("seed", base.get("seed", 42)))
    cfg["img_size"] = int(base.get("img_size", 224))
    cfg["data_dir"] = str(_abs(base["data_dir"]))
    cfg["split_path"] = str(_abs(base["split_path"]))
    cfg["label_map_path"] = str(_abs(base.get("label_map_path", "splits/label_map.json")))
    cfg["results_dir"] = str(_abs(base.get("results_dir", "results")))
    if cfg["mode"] not in MODES:
        raise ValueError(f"mode harus salah satu dari {MODES}, bukan {cfg['mode']!r}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cpu_threads = os.cpu_count() or 8
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_rgb(path, draft_size: int = 0) -> Image.Image:
    """Buka gambar sebagai RGB. draft_size>0: JPEG didekode pada skala >= draft_size (cepat untuk foto besar).
    EXIF orientation sengaja tidak diterapkan, sama dengan dataset.BisindoDataset."""
    img = Image.open(path)
    if draft_size:
        img.draft("RGB", (draft_size, draft_size))  # no-op untuk format non-JPEG
    return img.convert("RGB")


def _load_and_resize_cached(full_path: Path, draft_size: int) -> Image.Image:
    key = (str(full_path), draft_size)
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached
    img = load_rgb(full_path, draft_size)
    if draft_size and (img.width > draft_size or img.height > draft_size):
        img = img.resize((draft_size, draft_size), Image.BILINEAR)
    _IMAGE_CACHE[key] = img
    return img


def preload_images(paths: list[Path], draft_size: int) -> None:
    """Preload & perkecil gambar ke RAM secara paralel (multi-thread) agar epoch berikutnya 0 detik disk I/O."""
    todo = [p for p in paths if (str(p), draft_size) not in _IMAGE_CACHE]
    if not todo:
        return
    t0 = time.perf_counter()
    workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda p: _load_and_resize_cached(p, draft_size), todo))
    dt = time.perf_counter() - t0
    print(f"Preload {len(todo)} gambar ke RAM ({draft_size}px, {workers} threads) selesai dalam {dt:.1f}s")


class FastBisindoDataset(BisindoDataset):
    """BisindoDataset dengan decode JPEG cepat (draft) + opsi cache RAM. Transformasi tetap dari dataset.get_transforms."""

    def __init__(self, df, root, transform=None, draft_size: int = 256, cache_ram: bool = True):
        super().__init__(df, root, transform)
        self.paths = self.df["path"].tolist()
        self.labels = self.df["label_idx"].astype(int).tolist()
        self.draft_size = draft_size
        self.cache_ram = cache_ram

    def __getitem__(self, i):
        full_path = self.root / self.paths[i]
        if self.cache_ram:
            img = _load_and_resize_cached(full_path, self.draft_size)
        else:
            img = load_rgb(full_path, self.draft_size)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def _check_files(df: pd.DataFrame, root: Path, per_split: int = 15) -> None:
    missing = [p for _, sub in df.groupby("split") for p in sub["path"].head(per_split) if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} file contoh tidak ada di {root}, misalnya {missing[:3]}. "
            "Cek data_dir di configs/config.yaml (harus folder '01. Original Images' hasil ekstrak Mendeley)."
        )


def make_loaders(cfg: dict):
    df = pd.read_csv(cfg["split_path"])
    need = {"path", "label", "split"}
    if need - set(df.columns):
        raise ValueError(f"Kolom split kurang: {sorted(need - set(df.columns))}")
    if set(df["split"]) != set(SPLITS):
        raise ValueError(f"Nilai split harus {SPLITS}, ditemukan {sorted(set(df['split']))}")
    label_map = json.loads(Path(cfg["label_map_path"]).read_text(encoding="utf-8"))
    if any(label_map.get(c) != i for i, c in enumerate(CLASSES)):
        raise ValueError("label_map.json tidak sama dengan urutan A-Z di dataset.CLASSES")
    df["label_idx"] = df["label"].map(label_map)
    if df["label_idx"].isna().any():
        raise ValueError("Ada label di split yang tidak ada di label_map.json")

    root = Path(cfg["data_dir"])
    _check_files(df, root)

    cache_ram = bool(cfg.get("cache_ram", True))
    if cache_ram:
        preload_images([root / p for p in df["path"].tolist()], int(cfg["draft_size"]))

    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    # Jika sudah di-cache di RAM pada proses utama, num_workers=0 menghindari overhead IPC multiprocessing di Windows
    nw = 0 if cache_ram else int(cfg["num_workers"])
    loaders = {}
    for s in SPLITS:
        train = s == "train"
        tf = get_transforms(train=(train and cfg["augment"]), img_size=cfg["img_size"])
        ds = FastBisindoDataset(df[df["split"] == s], root, tf, cfg["draft_size"], cache_ram=cache_ram)
        loaders[s] = DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=train, num_workers=nw,
            generator=gen if train else None, pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
        )
    return loaders, df


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(mode: str = "head", unfreeze_blocks: int = 2, pretrained: bool = True,
                num_classes: int = len(CLASSES)) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)  # kepala baru

    for p in m.parameters():
        p.requires_grad = False
    if mode == "full":
        for p in m.parameters():
            p.requires_grad = True
    elif mode == "partial":
        n = len(m.features)
        k = max(1, min(int(unfreeze_blocks), n))
        for blk in m.features[n - k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in m.classifier.parameters():
            p.requires_grad = True
    for p in m.classifier[-1].parameters():  # kepala baru selalu dilatih
        p.requires_grad = True
    return m


def set_train(model: nn.Module) -> None:
    """model.train(), tapi BatchNorm yang seluruh parameternya beku tetap eval (statistik tidak berubah)."""
    model.train()
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in mod.parameters(recurse=False)):
                mod.eval()


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    set_train(model)
    loss_sum, correct, total = 0.0, 0, 0
    use_amp = device.type == "cuda" and scaler is not None
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device) -> dict:
    model.eval()
    logits, labels = [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(x)
        else:
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    loss = nn.functional.cross_entropy(logits, labels).item()
    conf, pred = logits.softmax(1).max(1)
    pred, labels = pred.numpy(), labels.numpy()
    return dict(
        loss=loss,
        acc=float(accuracy_score(labels, pred)),
        f1=float(f1_score(labels, pred, average="macro", zero_division=0)),
        pred=pred, labels=labels, conf=conf.numpy(),
    )


@torch.no_grad()
def measure_latency(model, df_split, root, tf, device, draft_size, seed, n=50, warmup=5) -> dict:
    """Median ms per gambar: total (decode + preprocess + forward) dan forward saja, batch 1."""
    model.eval()
    paths = df_split["path"].tolist()
    order = np.random.RandomState(seed).permutation(len(paths))[: n + warmup]
    total, fwd = [], []
    for k, idx in enumerate(order):
        t0 = time.perf_counter()
        x = tf(load_rgb(root / paths[idx], draft_size)).unsqueeze(0).to(device)
        t1 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if k >= warmup:
            total.append((t2 - t0) * 1000)
            fwd.append((t2 - t1) * 1000)
    return dict(total_ms_median=float(np.median(total)), forward_ms_median=float(np.median(fwd)), n=len(total))


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _save_curves(log: pd.DataFrame, path: Path) -> None:
    fig = Figure(figsize=(11, 4))
    a1, a2 = fig.subplots(1, 2)
    a1.plot(log["epoch"], log["train_loss"], label="train")
    a1.plot(log["epoch"], log["val_loss"], label="val")
    a1.set(title="Loss", xlabel="epoch")
    a1.legend()
    a2.plot(log["epoch"], log["train_acc"], label="train acc")
    a2.plot(log["epoch"], log["val_acc"], label="val acc")
    a2.plot(log["epoch"], log["val_f1"], label="val macro-F1")
    a2.set(title="Akurasi / F1", xlabel="epoch", ylim=(0, 1.02))
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def _save_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig = Figure(figsize=(10, 9))
    ax = fig.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xlabel="Prediksi", ylabel="Sebenarnya", title=title)
    ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    thr = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > thr else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def _save_samples(df_split, res, root, draft_size, seed, path: Path, n_each: int = 6) -> None:
    wrong = np.where(res["pred"] != res["labels"])[0]
    right = np.where(res["pred"] == res["labels"])[0]
    wrong = wrong[np.argsort(-res["conf"][wrong])][:n_each]  # salah dengan keyakinan tertinggi dulu
    right = np.random.RandomState(seed).permutation(right)[:n_each]
    idxs = list(wrong) + list(right)
    if not idxs:
        return
    cols = 6
    rows = int(np.ceil(len(idxs) / cols))
    fig = Figure(figsize=(cols * 2.2, rows * 2.5))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes:
        ax.axis("off")
    paths = df_split["path"].tolist()
    for ax, i in zip(axes, idxs):
        img = load_rgb(root / paths[i], draft_size)
        img.thumbnail((224, 224))
        ok = res["pred"][i] == res["labels"][i]
        ax.imshow(img)
        ax.set_title(f"T:{CLASSES[res['labels'][i]]}  P:{CLASSES[res['pred'][i]]} ({res['conf'][i]:.2f})",
                     fontsize=8, color="green" if ok else "red")
    fig.suptitle("Contoh prediksi (T = sebenarnya, P = prediksi): merah = salah dengan keyakinan tertinggi, "
                 "hijau = benar (acak)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _git_info() -> dict:
    def sh(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return dict(commit=sh("rev-parse", "--short", "HEAD"), branch=sh("rev-parse", "--abbrev-ref", "HEAD"),
                    dirty=bool(sh("status", "--porcelain")))
    except Exception:
        return dict(commit=None, branch=None, dirty=None)


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _report(split: str, res: dict, df_split: pd.DataFrame, out: Path) -> dict:
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(res["labels"], res["pred"], labels=labels, zero_division=0)
    pd.DataFrame(dict(kelas=CLASSES, precision=p, recall=r, f1=f, support=s)).to_csv(
        out / f"per_class_{split}.csv", index=False)
    cm = confusion_matrix(res["labels"], res["pred"], labels=labels)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_matrix_{split}.csv")
    _save_confusion(cm, f"Confusion matrix ({split})", out / f"confusion_matrix_{split}.png")
    pairs = [(CLASSES[i], CLASSES[j], int(cm[i, j])) for i in labels for j in labels if i != j and cm[i, j]]
    pairs.sort(key=lambda t: -t[2])
    pd.DataFrame(pairs[:15], columns=["sebenarnya", "diprediksi", "jumlah"]).to_csv(
        out / f"top_confusions_{split}.csv", index=False)
    pd.DataFrame(dict(
        path=df_split["path"].values, true=[CLASSES[i] for i in res["labels"]],
        pred=[CLASSES[i] for i in res["pred"]], conf=res["conf"].round(4),
    )).to_csv(out / f"predictions_{split}.csv", index=False)
    return dict(accuracy=res["acc"], macro_f1=res["f1"],
                weighted_f1=float(f1_score(res["labels"], res["pred"], average="weighted", zero_division=0)))


def run(cfg: dict, evaluate_test: bool = True) -> dict:
    """Latih satu run. Return dict ringkas (termasuk 'out_dir' dan 'metrics')."""
    set_seed(cfg["seed"])
    device = _pick_device(cfg["device"])
    out = Path(cfg["results_dir"]) / "transfer" / cfg["run_name"]
    if out.exists() and any(out.iterdir()) and not cfg.get("overwrite"):
        raise FileExistsError(f"{out} sudah berisi hasil. Ganti run_name atau pakai overwrite=True / --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    loaders, df = make_loaders(cfg)
    model = build_model(cfg["mode"], cfg["unfreeze_blocks"], cfg["pretrained"]).to(device)
    total_p, train_p = count_params(model)
    n = {s: int((df["split"] == s).sum()) for s in SPLITS}
    split_md5 = hashlib.md5(Path(cfg["split_path"]).read_bytes()).hexdigest()
    print(f"Run: {cfg['run_name']} | mode={cfg['mode']} | device={device} | torch {torch.__version__}")
    print(f"Split: {_rel(cfg['split_path'])} (md5 {split_md5[:8]}) | {n}")
    print(f"Parameter: total {total_p:,} | dilatih {train_p:,} ({100 * train_p / total_p:.2f}%)")

    head_params = list(model.classifier[-1].parameters())
    head_ids = {id(p) for p in head_params}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [dict(params=head_params, lr=cfg["lr_head"])]
    if rest:
        groups.append(dict(params=rest, lr=cfg["lr_backbone"]))
    optimizer = optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    w = compute_class_weights(df[df["split"] == "train"]).to(device) if cfg["class_weights"] else None
    crit_train = nn.CrossEntropyLoss(weight=w)

    used = {k: v for k, v in cfg.items() if k not in ("data_dir", "split_path", "label_map_path", "results_dir")}
    used.update(data_dir=_rel(cfg["data_dir"]), split_path=_rel(cfg["split_path"]),
                label_map_path=_rel(cfg["label_map_path"]), split_md5=split_md5, n=n,
                params_total=total_p, params_trainable=train_p, torch=torch.__version__,
                python=platform.python_version(), git=_git_info())
    (out / "config_used.json").write_text(json.dumps(used, indent=2, ensure_ascii=False), encoding="utf-8")

    rows, best_f1, best_epoch, best_val, bad = [], -1.0, 0, None, 0
    t_start = time.perf_counter()
    try:
        for ep in range(1, cfg["epochs"] + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = train_one_epoch(model, loaders["train"], crit_train, optimizer, device, scaler=scaler)
            va = predict(model, loaders["val"], device)
            rows.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=va["loss"],
                             val_acc=va["acc"], val_f1=va["f1"], seconds=time.perf_counter() - t0))
            pd.DataFrame(rows).to_csv(out / "log.csv", index=False)
            print(f"Epoch {ep:02d}/{cfg['epochs']} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['f1']:.3f} | {rows[-1]['seconds']:.1f}s")
            if va["f1"] > best_f1 + 1e-4:
                best_f1, best_epoch, best_val, bad = va["f1"], ep, va, 0
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                epoch=ep, val_f1=best_f1, mode=cfg["mode"], classes=CLASSES), out / "best.pth")
            else:
                bad += 1
                if cfg["patience"] and bad >= cfg["patience"]:
                    print(f"Early stopping: val macro-F1 tidak naik {bad} epoch berturut-turut.")
                    break
    except KeyboardInterrupt:
        print("\nDihentikan manual. Memakai model terbaik sejauh ini.")
    train_seconds = time.perf_counter() - t_start
    if best_val is None:
        raise RuntimeError("Tidak ada epoch yang selesai.")

    ckpt = torch.load(out / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    log = pd.DataFrame(rows)
    _save_curves(log, out / "curves.png")

    root = Path(cfg["data_dir"])
    split_name = "test" if evaluate_test else "val"
    res = predict(model, loaders[split_name], device) if evaluate_test else best_val
    df_s = df[df["split"] == split_name].reset_index(drop=True)
    summary = _report(split_name, res, df_s, out)
    _save_samples(df_s, res, root, cfg["draft_size"], cfg["seed"], out / f"sample_predictions_{split_name}.png")
    lat = measure_latency(model, df_s, root, get_transforms(False, cfg["img_size"]), device,
                          cfg["draft_size"], cfg["seed"])

    metrics = dict(
        run_name=cfg["run_name"], mode=cfg["mode"], best_epoch=best_epoch, epochs_run=len(rows),
        best_val_macro_f1=best_f1, best_val_accuracy=best_val["acc"], report_split=split_name,
        test=summary if evaluate_test else None, val=None if evaluate_test else summary,
        latency_ms=lat, device=str(device), train_seconds=round(train_seconds, 1),
        params_total=total_p, params_trainable=train_p, split_md5=split_md5, n=n,
        hparams={k: cfg[k] for k in ("unfreeze_blocks", "lr_head", "lr_backbone", "weight_decay", "batch_size",
                                     "augment", "class_weights", "pretrained", "seed", "epochs", "patience")},
    )
    if "group" not in df.columns or df["group"].isna().all():
        metrics["catatan"] = ("Split tanpa grup perangkat/latar: gambar dengan kondisi sama bisa ada di train dan "
                              "test, jadi skor bisa lebih optimis daripada kondisi nyata.")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nModel terbaik: epoch {best_epoch} (val macro-F1 {best_f1:.3f})")
    print(f"[{split_name}] accuracy {summary['accuracy']:.4f} | macro-F1 {summary['macro_f1']:.4f}")
    print(f"Latency median: {lat['total_ms_median']:.1f} ms total, {lat['forward_ms_median']:.1f} ms forward saja")
    print(f"Hasil: {_rel(out)}")
    return dict(out_dir=out, metrics=metrics, log=log)


def compare_runs(results_dir: str | None = None, save: bool = True) -> pd.DataFrame:
    """Tabel perbandingan semua run di results/transfer/*/metrics.json (untuk eksperimen terkontrol)."""
    base = Path(results_dir) if results_dir else _abs(_load_yaml(_abs("configs/config.yaml")).get("results_dir", "results"))
    rows = []
    for mp in sorted((base / "transfer").glob("*/metrics.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        h, t = m.get("hparams", {}), m.get("test") or {}
        rows.append(dict(
            run=m["run_name"], mode=m["mode"], unfreeze=h.get("unfreeze_blocks") if m["mode"] == "partial" else None,
            lr_head=h.get("lr_head"), lr_backbone=h.get("lr_backbone"), augment=h.get("augment"),
            class_w=h.get("class_weights"), seed=h.get("seed"), epochs=m["epochs_run"], best_ep=m["best_epoch"],
            val_f1=round(m["best_val_macro_f1"], 4),
            test_acc=round(t["accuracy"], 4) if t else None, test_f1=round(t["macro_f1"], 4) if t else None,
            ms=round(m["latency_ms"]["total_ms_median"], 1), trainable=m["params_trainable"],
        ))
    df = pd.DataFrame(rows)
    if save and len(df):
        df.to_csv(base / "transfer" / "comparison.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Transfer learning MobileNetV3-Small untuk BISINDO")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--transfer-config", default="configs/transfer.yaml")
    ap.add_argument("--run-name")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--unfreeze-blocks", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr-head", type=float)
    ap.add_argument("--lr-backbone", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--draft-size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--no-cache-ram", action="store_true", help="matikan cache gambar di RAM")
    ap.add_argument("--no-augment", action="store_true", help="train tanpa augmentasi")
    ap.add_argument("--class-weights", action="store_true", help="CrossEntropy berbobot")
    ap.add_argument("--no-pretrained", action="store_true", help="tanpa bobot ImageNet (hanya untuk uji kode)")
    ap.add_argument("--no-test", action="store_true", help="jangan evaluasi test (untuk tuning)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare", action="store_true", help="tampilkan tabel semua run lalu keluar")
    a = ap.parse_args()

    if a.compare:
        df = compare_runs()
        print(df.to_string(index=False) if len(df) else "Belum ada run di results/transfer/")
        return

    ov = dict(run_name=a.run_name, mode=a.mode, unfreeze_blocks=a.unfreeze_blocks, epochs=a.epochs,
              batch_size=a.batch_size, lr_head=a.lr_head, lr_backbone=a.lr_backbone,
              weight_decay=a.weight_decay, patience=a.patience, num_workers=a.num_workers,
              draft_size=a.draft_size, seed=a.seed, device=a.device)
    if a.no_cache_ram:
        ov["cache_ram"] = False
    if a.no_augment:
        ov["augment"] = False
    if a.class_weights:
        ov["class_weights"] = True
    if a.no_pretrained:
        ov["pretrained"] = False
    if a.overwrite:
        ov["overwrite"] = True
    try:
        run(build_cfg(ov, a.config, a.transfer_config), evaluate_test=not a.no_test)
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
