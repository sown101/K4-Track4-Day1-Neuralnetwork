"""plots.py — ảnh từng thí nghiệm (figures/<exp_id>.png) và ảnh chồng theo nhóm (figures/compare_<nhóm>.png)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def cfg_label(cfg: dict) -> str:
    """Mô tả cấu hình chính trên một dòng, dùng cho tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg["hidden"])
    clip = cfg["clip_norm"] if cfg["clip_norm"] is not None else "none"
    s = (f"{cfg['loss'].upper()} | {cfg['optimizer']} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} | "
         f"batch={cfg['batch']} ep={cfg['epochs']} | {hidden} q={cfg['dropout']} | "
         f"clip={clip} {cfg['precision']} init={cfg['init']} | seed={cfg['seed']}")
    if cfg.get("scheduler"):
        s += f" | sched={cfg['scheduler']}"
    return s


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> PNG 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm (trước clip)."""
    cfg, s = result["cfg"], result["summary"]
    # None (NaN đã lưu trong JSON) -> nan để matplotlib/numpy xử lý được
    h = {k: (np.array(v, dtype=float) if k != "epoch" else v) for k, v in result["history"].items()}
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    title = f"{cfg['exp_id']}  —  {cfg_label(cfg)}"
    if s.get("diverged"):
        title += "   [DIVERGED]"
    fig.suptitle(title, fontsize=10)

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode, 50k mẫu cố định)")
    ax.plot(ep, h["val_loss"], "o-", ms=3, label="val loss")
    if s.get("step0_loss") is not None and np.isfinite(s["step0_loss"]):
        ax.plot([0], [s["step0_loss"]], "k*", ms=9, label=f"loss bước 0 = {s['step0_loss']:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel(f"loss ({cfg['loss'].upper()})"); ax.set_title("(1) train / val loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, label="val macro-F1")
    ax.axhline(0.4876, color="gray", ls=":", lw=1, label="đoán lớp đa số (acc 0,4876)")
    ax.set_xlabel("epoch"); ax.set_ylabel("điểm"); ax.set_title("(2) val accuracy / macro-F1")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="grad_norm TB của epoch (trước clip)")
    if "grad_norm_max" in h:
        ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.7, label="grad_norm lớn nhất của epoch")
    if cfg["clip_norm"] is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls="--", lw=1, label=f"ngưỡng clip c = {cfg['clip_norm']:g}")
    ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖₂ toàn cục"); ax.set_title("(3) grad_norm")
    if len(ep) and np.nanmax(h.get("grad_norm_max", h["grad_norm"])) / max(np.nanmin(h["grad_norm"]), 1e-12) > 50:
        ax.set_yscale("log")

    best = s.get("best_epoch")
    for ax in axes:
        if best:
            ax.axvline(best, color="green", ls="--", lw=0.8, label=f"best epoch = {best}" if ax is axes[0] else None)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một hoặc nhiều chỉ số của nhiều thí nghiệm; mỗi thí nghiệm một đường (chú thích bằng exp_id).

    metric: tên một chỉ số (vd "val_loss") hoặc danh sách tên -> mỗi chỉ số một ô.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            h = r["history"]
            label = r["cfg"]["exp_id"] + (" [div]" if r["summary"].get("diverged") else "")
            ax.plot(h["epoch"], np.array(h[m], dtype=float), "o-", ms=2.5, label=label)
        ax.set_xlabel("epoch"); ax.set_ylabel(m); ax.set_title(m)
        if m.startswith("grad_norm"):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
