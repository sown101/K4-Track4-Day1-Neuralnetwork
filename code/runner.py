"""runner.py — chạy một thí nghiệm hoặc nạp lại kết quả đã có (Colab có thể ngắt kết nối giữa chừng).

Mỗi lần chạy: run_experiment -> results/<exp_id>.json + figures/<exp_id>.png.
Nếu JSON đã tồn tại với CÙNG cấu hình thì nạp lại thay vì huấn luyện lại (đặt force=True để chạy lại).
Trọng số (chỉ cho baseline / cấu hình cuối, cần để dự đoán eval) lưu ở <repo>/checkpoints/, KHÔNG nộp.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from plots import plot_run
from results_table import load_result, save_result
from train import DEFAULT_CFG, run_experiment


def _norm(cfg: dict) -> str:
    return json.dumps({**DEFAULT_CFG, **cfg}, sort_keys=True, default=list)


def run_or_load(cfg: dict, data: dict, out_dir: str, ckpt_dir: str | None = None,
                keep_state: bool = False, force: bool = False, show: bool = False) -> dict:
    cfg = {**DEFAULT_CFG, **cfg}
    exp_id = cfg["exp_id"]
    results_dir, fig_path = f"{out_dir}/results", f"{out_dir}/figures/{exp_id}.png"
    ckpt = Path(ckpt_dir) / f"{exp_id}.pt" if ckpt_dir else None

    cached = None if force else load_result(exp_id, results_dir)
    if cached is not None and json.loads(_norm(cached["cfg"])) == json.loads(_norm(cfg)) \
            and (not keep_state or (ckpt is not None and ckpt.is_file())):
        if keep_state:
            cached["best_state"] = torch.load(ckpt, map_location="cpu")
        s = cached["summary"]
        h = cached["history"]
        print(f"[{exp_id}] nạp lại từ {results_dir}/{exp_id}.json (không huấn luyện lại)")
        for i, ep in enumerate(h["epoch"]):
            print(f"[{exp_id}] ep {ep:2d}  train {h['train_loss'][i]:.4f}  val {h['val_loss'][i]:.4f}  "
                  f"acc {h['val_acc'][i]:.4f}  F1 {h['val_macro_f1'][i]:.4f}  |g| {h['grad_norm'][i]:.3f}")
        print(f"[{exp_id}] best epoch {s['best_epoch']}: val_F1 {s['val_macro_f1']}  diverged={s['diverged']}")
        if not Path(fig_path).is_file():
            plot_run(cached, fig_path)
        result = cached
    else:
        result = run_experiment(cfg, data)
        save_result(result, results_dir)
        plot_run(result, fig_path)
        if keep_state and ckpt is not None and result["best_state"] is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result["best_state"], ckpt)
    if show:
        from IPython.display import Image, display
        display(Image(fig_path, width=900))
    return result
