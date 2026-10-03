"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000   # train_loss đo trên một tập con CỐ ĐỊNH của train (ở eval mode)
SUBSET_SEED = 0              # tập con không phụ thuộc seed của thí nghiệm -> so sánh được giữa các run

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn k×k, hàng = nhãn thật, cột = dự đoán (giống scripts/evaluate.py)."""
    idx = y_true.long() * k + y_pred.long()
    return torch.bincount(idx, minlength=k * k).reshape(k, k).cpu().numpy()


def per_class_scores(cm: np.ndarray):
    """precision, recall, F1 từng lớp; bằng 0 khi mẫu số bằng 0 (cùng quy ước với evaluate.py)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, ở eval mode (dropout tắt), FP32."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str):
    """"ce"  : F.cross_entropy trên logit thô và nhãn int64 (softmax nằm trong hàm loss).
       "mse" : nn.MSELoss mặc định giữa logit và one-hot(y): trung bình trên MỌI phần tử (B·7), không có hệ số 1/2.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits.float(), y)
    if loss_name == "mse":
        target = F.one_hot(y, N_CLASSES).float()
        return F.mse_loss(logits.float(), target)
    raise ValueError(f"loss không hỗ trợ: {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192, return_cm: bool = False) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad, FP32."""
    model.eval()
    total_loss = 0.0
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        # reduction="sum" cho CE; với MSE, mean trên 7 phần tử của mỗi mẫu rồi cộng theo mẫu
        total_loss += compute_loss(logits, y[i:i + batch_size], loss_name).item() * len(logits)
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    cm = confusion_matrix(y, pred)
    out = dict(loss=total_loss / len(X), acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))
    if return_cm:
        out["cm"] = cm
    return out


def _autocast(precision: str, device_type: str):
    if precision == "fp32":
        return torch.autocast(device_type=device_type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


def build_model(cfg: dict, device) -> MLP:
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"])
    n = count_params(model)
    assert n == EXPECTED_PARAMS[hidden], f"số tham số {n} != {EXPECTED_PARAMS[hidden]} cho {hidden}"
    return model.to(device)


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    Chỉ dùng X_tr để cập nhật và X_val để chọn best epoch. X_eval KHÔNG được dùng ở đây.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    assert cfg["lr"] is not None, "cần đặt lr"
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    dev_type = device.type
    is_cuda = dev_type == "cuda"

    # ---- 0. khởi tạo
    set_seed(cfg["seed"])
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps=steps_per_epoch * cfg["epochs"])
    use_scaler = cfg["precision"] == "fp16"
    scaler = torch.amp.GradScaler(dev_type, enabled=use_scaler)
    gen = torch.Generator().manual_seed(cfg["seed"])  # thứ tự lô phụ thuộc seed, tái lập được

    sub_gen = torch.Generator().manual_seed(SUBSET_SEED)
    sub_idx = torch.randperm(len(X_tr), generator=sub_gen)[:TRAIN_EVAL_SUBSET].to(device)
    X_sub, y_sub = X_tr[sub_idx], y_tr[sub_idx]

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 1. loss bước 0 (trước cập nhật đầu tiên) + độ lệch chuẩn kích hoạt ở bước 0
    step0 = evaluate(model, X_val, y_val, cfg["loss"])
    act_std0 = activation_stats(model, X_val[:4096])

    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                               "grad_norm", "grad_norm_max", "clip_frac", "skipped_steps", "epoch_time_s", "lr")}
    best = dict(val_loss=math.inf, epoch=None, state=None, metrics=None)
    diverged = False

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gnorms, n_clipped, n_skipped = [], 0, 0
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            with _autocast(cfg["precision"], dev_type):  # chỉ bọc forward + loss
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # no-op khi scaler tắt; với fp16 phải unscale TRƯỚC khi đo/clip
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])  # chuẩn TRƯỚC khi cắt
            scaler.step(optimizer)       # fp16: tự bỏ qua bước nếu gradient inf/NaN
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            if math.isfinite(gn):
                gnorms.append(gn)
                if cfg["clip_norm"] is not None and gn > cfg["clip_norm"]:
                    n_clipped += 1
            elif use_scaler:
                n_skipped += 1           # tràn số FP16: GradScaler bỏ bước này và giảm hệ số s
            if not math.isfinite(loss.item()) or (not use_scaler and not math.isfinite(gn)):
                diverged = True
                break
        if is_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        if diverged:
            if verbose:
                print(f"[{cfg['exp_id']}] epoch {epoch}: loss/grad NaN/inf -> diverged, dừng sớm")
            break

        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        if not math.isfinite(va["loss"]):
            diverged = True
            if verbose:
                print(f"[{cfg['exp_id']}] epoch {epoch}: val loss không hữu hạn -> diverged, dừng sớm")
            break

        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_macro_f1"].append(va["macro_f1"])
        history["grad_norm"].append(float(np.mean(gnorms)) if gnorms else float("nan"))
        history["grad_norm_max"].append(float(np.max(gnorms)) if gnorms else float("nan"))
        history["clip_frac"].append(n_clipped / steps_per_epoch)
        history["skipped_steps"].append(n_skipped)
        history["epoch_time_s"].append(epoch_time)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch=epoch, metrics=va,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if verbose:
            print(f"[{cfg['exp_id']}] ep {epoch:2d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
                  f"acc {va['acc']:.4f}  F1 {va['macro_f1']:.4f}  |g| {history['grad_norm'][-1]:.3f}  "
                  f"{epoch_time:.1f}s")

    # ---- 3. tóm tắt tại best epoch
    nan = float("nan")
    summary = dict(
        step0_loss=step0["loss"],
        best_val_loss=best["val_loss"] if best["epoch"] else nan,
        best_epoch=best["epoch"],
        final_train_loss=history["train_loss"][-1] if history["epoch"] else nan,
        final_val_loss=history["val_loss"][-1] if history["epoch"] else nan,
        val_acc=best["metrics"]["acc"] if best["metrics"] else nan,
        val_macro_f1=best["metrics"]["macro_f1"] if best["metrics"] else nan,
        time_per_epoch_s=float(np.mean(history["epoch_time_s"])) if history["epoch"] else nan,
        peak_mem_MB=torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else nan,
        diverged=diverged,
        act_std_step0=act_std0,
        n_params=count_params(model),
        steps_per_epoch=steps_per_epoch,
    )
    if verbose:
        print(f"[{cfg['exp_id']}] best epoch {summary['best_epoch']}: val_loss {summary['best_val_loss']:.4f}  "
              f"val_acc {summary['val_acc']:.4f}  val_F1 {summary['val_macro_f1']:.4f}  "
              f"({summary['time_per_epoch_s']:.2f}s/epoch)")
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best["state"]}


def write_predictions(row_id, preds, path: str) -> None:
    """CSV `row_id,pred` cho scripts/evaluate.py; đủ mọi dòng eval, mỗi row_id đúng một lần."""
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán TOÀN BỘ eval, ghi predictions."""
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    model = build_model(cfg, device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {len(preds):,} dự đoán -> {pred_path}")

