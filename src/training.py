"""Training core matching the frozen Stage16/Stage18B FiLM-TCN training loop.

The dataset and sampler must be constructed beforehand using the corresponding
predefined-three-source or regional-LOBO protocol. This module deliberately does
not choose a sampler or fit a scaler.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 5e-4
    epochs: int = 80
    batch_size: int = 256  # Loader configuration, for documentation only.
    early_stopping_patience: int = 15
    grad_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_learning_rate: float = 1e-5


def set_seed(seed: int) -> None:
    """Preserve formal GPU settings. Bitwise GPU determinism is not claimed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Mask value 0 marks an observed target; all-invalid batches have zero loss.

    Formal training still calls backward() and optimizer.step() when this loss
    is finite. Never silently skip such a batch, as Adam momentum may persist.
    """
    valid = mask == 0
    if valid.sum() == 0:
        return (pred * 0.0).sum()
    return torch.mean((pred[valid] - target[valid]) ** 2)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_one_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    seed: int,
    run_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    config: TrainConfig = TrainConfig(),
    verbose: bool = False,
) -> dict:
    """Train exactly as the Stage16/Stage18B core on prepared source loaders.

    Required 7-field batch order:
      (dynamic_x, static_features, target_y, missing_mask,
       basin_id, previous_discharge, timestamp)

    Each epoch averages losses equally over finite batches; it does not weight
    batch losses by the number of valid labels. Scheduler uses that same
    validation mean. The best state is selected only on source validation loss.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    set_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.min_learning_rate,
    )
    best_val_loss = float('inf')
    best_epoch = None
    best_state = None
    patience_count = 0
    logs = []
    n_params = count_trainable_params(model)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for bX, bS, bY, bM, bID, bQPrev, bDate in train_loader:
            bX = bX.to(device, non_blocking=True)
            bS = bS.to(device, non_blocking=True)
            bY = bY.to(device, non_blocking=True).view(-1)
            bM = bM.to(device, non_blocking=True).view(-1)
            optimizer.zero_grad()
            if model_name == 'MS-TCN(RS)':
                out = model(bX)
            else:
                gate = torch.ones(bX.shape[0], dtype=torch.float32, device=device)
                out = model(bX, bS, gate)
            loss = masked_mse_loss(out, bY, bM)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                train_sum += loss.item()
                train_batches += 1
        train_loss = train_sum / max(train_batches, 1)

        model.eval()
        val_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for bX, bS, bY, bM, bID, bQPrev, bDate in val_loader:
                bX = bX.to(device, non_blocking=True)
                bS = bS.to(device, non_blocking=True)
                bY = bY.to(device, non_blocking=True).view(-1)
                bM = bM.to(device, non_blocking=True).view(-1)
                if model_name == 'MS-TCN(RS)':
                    out = model(bX)
                else:
                    gate = torch.ones(bX.shape[0], dtype=torch.float32, device=device)
                    out = model(bX, bS, gate)
                loss = masked_mse_loss(out, bY, bM)
                if torch.isfinite(loss):
                    val_sum += loss.item()
                    val_batches += 1
        val_loss = val_sum / max(val_batches, 1)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1
        logs.append({
            'model': model_name, 'seed': seed, 'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'best_val_loss': best_val_loss, 'lr': lr_now,
            'is_best': is_best, 'patience_count': patience_count,
        })
        if verbose and (is_best or epoch % 5 == 0 or patience_count >= config.early_stopping_patience):
            print(f'Epoch {epoch:03d}/{config.epochs}: train={train_loss:.6f}, '
                  f'val={val_loss:.6f}, best={is_best}, patience={patience_count}')
        if patience_count >= config.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError(f'{model_name} has no best_state')
    model.load_state_dict(best_state)
    pd.DataFrame(logs).to_csv(
        run_dir / 'train_log.csv', index=False, encoding='utf-8-sig',
    )
    torch.save({
        'model_name': model_name, 'seed': seed, 'best_epoch': best_epoch,
        'best_val_loss': best_val_loss, 'n_params': n_params,
        'state_dict': model.state_dict(),
    }, run_dir / 'best_model.pth')
    return {
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'n_params': n_params,
    }
