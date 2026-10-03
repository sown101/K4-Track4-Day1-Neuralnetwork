"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def _check(X, y, name):
    assert X.ndim == 2 and X.shape[1] == N_FEATURES, f"{name}: X phải có shape (N, 54), hiện là {X.shape}"
    assert X.dtype == np.float32, f"{name}: X phải là float32, hiện là {X.dtype}"
    assert y.shape == (X.shape[0],) and y.dtype == np.int64, f"{name}: y phải là int64 (N,)"
    assert y.min() >= 0 and y.max() <= N_CLASSES - 1, f"{name}: nhãn phải nằm trong 0..6"


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz. Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id."""
    with np.load(f"{processed_dir}/train.npz") as d:
        X_train, y_train = d["X"], d["y"]
    with np.load(f"{processed_dir}/eval.npz") as d:
        X_eval, y_eval, eval_row_id = d["X"], d["y"], d["row_id"]
    _check(X_train, y_train, "train")
    _check(X_eval, y_eval, "eval")
    assert len(eval_row_id) == len(X_eval)
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn. Trả về: X_tr, y_tr, X_val, y_val."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean và std của 10 cột số, tính CHỈ trên phần train còn lại (sau khi tách val).

    Không tính trên val/eval vì như vậy thống kê của dữ liệu dùng để đánh giá rò rỉ vào bước tiền xử lý
    (data leakage) và điểm val/eval không còn là ước lượng trung thực cho dữ liệu chưa thấy.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std[std == 0] = 1.0  # tránh chia cho 0 nếu có cột hằng
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X: 10 cột đầu thành (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader)."""
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)  # chỉ dùng X_tr
    X_tr, X_val, X_eval = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_eval))

    def t(a, dtype):
        return torch.tensor(a, dtype=dtype, device=device)

    data = dict(
        X_tr=t(X_tr, torch.float32), y_tr=t(y_tr, torch.int64),
        X_val=t(X_val, torch.float32), y_val=t(y_val, torch.int64),
        X_eval=t(X_eval, torch.float32), y_eval=t(y_eval, torch.int64),
        eval_row_id=eval_row_id, mean=mean, std=std,
    )

    if verbose:
        print(f"train (sau tách val): {len(X_tr):,}   val: {len(X_val):,}   eval: {len(X_eval):,}")
        majority = np.bincount(y_tr, minlength=N_CLASSES).argmax()
        print(f"lớp đa số trên train = {majority}; accuracy 'luôn đoán lớp đa số' trên val = "
              f"{(y_val == majority).mean():.4f}")
        print("tỉ lệ lớp (%)  train / val / eval:")
        for name, yy in (("train", y_tr), ("val", y_val), ("eval", y_eval)):
            print(f"  {name:5s}", np.round(100 * np.bincount(yy, minlength=N_CLASSES) / len(yy), 2))
        num = X_tr[:, :N_NUMERIC]
        print("X_tr 10 cột số: mean ∈ [%.2e, %.2e], std ∈ [%.4f, %.4f]"
              % (num.mean(0).min(), num.mean(0).max(), num.std(0).min(), num.std(0).max()))
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Sinh từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối có thể nhỏ hơn batch_size; ta GIỮ nó (không bỏ) để mỗi epoch đi qua đủ mọi mẫu.
    """
    n = len(X)
    if shuffle:
        # randperm trên CPU với generator CPU rồi chuyển sang device: tái lập được trên mọi thiết bị
        perm = torch.randperm(n, generator=generator).to(X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
