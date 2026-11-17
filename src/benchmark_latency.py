"""
benchmark_latency.py - ukur latensi inferensi per tahap untuk satu run (batch 1, seperti demo/app).

  python src/benchmark_latency.py --config configs/config_v3.yaml --run v3_full_cos30_ls_strong
  python src/benchmark_latency.py --config configs/config_v3.yaml --run v3_full_cos30_ls_strong --device cpu

Tahap yang diukur (median ms per gambar, gambar val acak):
  decode      JPEG -> RGB (draft ke >= img_size, sama dengan demo/app)
  preprocess  resize 224 + normalisasi (transform evaluasi)
  forward     model saja: eager (inference_mode) dan teroptimasi (TorchScript di GPU; di CPU sama dengan eager)
  total       decode + preprocess + forward teroptimasi
Hasil ditulis ke results/transfer/<run>/latency_<device>.json.
Tutup aplikasi berat lain (dan jangan jalankan training bersamaan) agar angka stabil.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import get_transforms  # noqa: E402
from demo import load_model, optimize_for_device  # noqa: E402
from train_transfer import _load_yaml, _abs, load_rgb  # noqa: E402


def _median_ms(fn, inputs, warmup: int, sync) -> float:
    for x in inputs[:warmup]:
        fn(x)
        sync()
    ts = []
    for x in inputs[warmup:]:
        t = time.perf_counter()
        fn(x)
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Latensi inferensi per tahap (batch 1)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--n", type=int, default=100, help="jumlah gambar diukur (setelah pemanasan)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    base = _load_yaml(_abs(a.config))
    root, split = _abs(base["data_dir"]), pd.read_csv(_abs(base["split_path"]))
    run_dir = _abs(base.get("results_dir", "results")) / "transfer" / a.run
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if a.device == "auto" else a.device))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    eager, img_size, _ = load_model(run_dir, device, optimize=False)
    fast = optimize_for_device(eager, img_size, device)
    tf = get_transforms(False, img_size)

    paths = split[split["split"] == "val"]["path"].tolist()
    order = np.random.RandomState(a.seed).permutation(len(paths))[: a.n + a.warmup]
    files = [root / paths[i] for i in order]
    imgs = [load_rgb(p, img_size) for p in files]
    xs = [tf(im).unsqueeze(0).to(device) for im in imgs]

    def fwd(m):
        def f(x):
            with torch.inference_mode():
                m(x)
        return f

    def total(p):
        with torch.inference_mode():
            fast(tf(load_rgb(p, img_size)).unsqueeze(0).to(device))

    res = dict(
        run=a.run, device=str(device), n=len(files) - a.warmup, torch=torch.__version__,
        optimized=type(fast).__name__ != type(eager).__name__,
        decode_ms=_median_ms(lambda p: load_rgb(p, img_size), files, a.warmup, lambda: None),
        preprocess_ms=_median_ms(tf, imgs, a.warmup, lambda: None),
        forward_eager_ms=_median_ms(fwd(eager), xs, a.warmup, sync),
        forward_optimized_ms=_median_ms(fwd(fast), xs, a.warmup, sync),
        total_ms=_median_ms(total, files, a.warmup, sync),
    )
    if device.type == "cuda":
        res["gpu"] = torch.cuda.get_device_name(device)
    out = run_dir / f"latency_{device.type}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"Run {a.run} | {device} | n={res['n']} | TorchScript: {'ya' if res['optimized'] else 'tidak'}")
    for k in ("decode_ms", "preprocess_ms", "forward_eager_ms", "forward_optimized_ms", "total_ms"):
        print(f"  {k:22s} {res[k]:7.2f} ms")
    print(f"  ~{1000 / res['total_ms']:.0f} gambar/detik (foto dari disk) | "
          f"~{1000 / (res['preprocess_ms'] + res['forward_optimized_ms']):.0f} fps tanpa decode (webcam)")
    print(f"Hasil: {out}")


if __name__ == "__main__":
    main()
