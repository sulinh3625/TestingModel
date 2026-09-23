from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def get_device(config_device: str = "auto") -> torch.device:
    """Auto-detect thiết bị tốt nhất: cuda > cpu."""
    config_device = config_device.lower().strip()
    if config_device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return dev
        print("  ⓘ GPU không khả dụng, dùng CPU")
        return torch.device("cpu")
    return torch.device(config_device)


def _gpu_mem_str() -> str:
    """Trả về chuỗi mô tả GPU memory nếu đang dùng CUDA."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return f" | GPU {alloc:.0f}/{reserved:.0f} MB"
    return ""


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Optimizer không hỗ trợ: {name}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_model(
    model,
    train_dataset,
    val_records,
    eval_model_fn,
    optimizer,
    device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    monitor: str = "NDCG@10",
    seed: int = 42,
    model_name: str = "model",
):
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  🚀 Training: {model_name}")
    print(f"  Parameters: {n_params:,} | Device: {device}")
    print(f"  Max epochs: {max_epochs} | Patience: {patience} | Batch: {batch_size}")
    print(f"  Monitor: {monitor}")
    print(f"{'='*60}")

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_metric = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0
    history = []
    rng = np.random.default_rng(seed)
    total_train_time = 0.0

    epoch_bar = tqdm(
        range(1, int(max_epochs) + 1),
        desc=f"[{model_name}] Epochs",
        unit="epoch",
        ncols=110,
        leave=True,
    )

    for epoch in epoch_bar:
        t0 = time.perf_counter()
        train_dataset.resample()
        n = len(train_dataset.users)
        perm = rng.permutation(n)
        total_batches = (n + batch_size - 1) // batch_size

        model.train()
        total_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(
            range(0, n, batch_size),
            desc=f"  Epoch {epoch:>3d}",
            unit="batch",
            total=total_batches,
            ncols=100,
            leave=False,
        )

        for start in batch_bar:
            idx = perm[start:start + batch_size]
            u = torch.from_numpy(train_dataset.users[idx]).long().to(device)
            i = torch.from_numpy(train_dataset.items[idx]).long().to(device)
            y = torch.from_numpy(train_dataset.labels[idx]).float().to(device)
            w = torch.from_numpy(train_dataset.sample_weights[idx]).float().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(u, i)
            loss = (criterion(logits, y) * w).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

            # Cập nhật batch progress bar
            avg_loss = total_loss / n_batches
            batch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        batch_bar.close()

        epoch_time = time.perf_counter() - t0
        total_train_time += epoch_time
        val_metrics = eval_model_fn(model, val_records)
        if monitor not in val_metrics:
            raise KeyError(f"Metric early stopping '{monitor}' không có trong evaluator: {sorted(val_metrics)}")
        metric = float(val_metrics[monitor])
        avg_loss = total_loss / max(n_batches, 1)
        row = {
            "epoch": epoch,
            "loss": avg_loss,
            monitor: metric,
            "epoch_time_s": epoch_time,
        }
        history.append(row)

        # Cập nhật epoch progress bar
        star = ""
        if metric > best_metric + 1e-12:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            star = " ★"
        else:
            no_improve += 1

        epoch_bar.set_postfix_str(
            f"loss={avg_loss:.4f} | {monitor}={metric:.4f} | "
            f"best={best_metric:.4f}@E{best_epoch} | "
            f"wait={no_improve}/{patience} | {epoch_time:.1f}s{_gpu_mem_str()}{star}"
        )

        if no_improve >= patience:
            tqdm.write(
                f"  ⏹ Early stopping tại epoch {epoch} — "
                f"không cải thiện sau {patience} epochs. "
                f"Best: {monitor}={best_metric:.4f} @ epoch {best_epoch}"
            )
            break
    else:
        tqdm.write(
            f"  ✓ Hoàn thành {max_epochs} epochs. "
            f"Best: {monitor}={best_metric:.4f} @ epoch {best_epoch}"
        )

    epoch_bar.close()

    print(f"  ⏱ Tổng thời gian train {model_name}: {total_train_time:.1f}s")
    print(f"  🏆 Best {monitor}: {best_metric:.4f} @ epoch {best_epoch}")
    print()

    model.load_state_dict(best_state)
    return model, history, {
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "train_time_s": float(total_train_time),
        "n_parameters": int(n_params),
        "model_name": model_name,
    }