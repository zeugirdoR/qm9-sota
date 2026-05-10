from __future__ import annotations

import torch


def normalize_y(y: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return (y - target_mean) / target_std


def denormalize_y(y_norm: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return y_norm * target_std + target_mean


@torch.no_grad()
def evaluate_both(model, loader, device, target_mean, target_std):
    model.eval()
    raw_abs_err_sum = torch.zeros(19, device=device)
    norm_abs_err_sum = torch.zeros(19, device=device)
    n = 0

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        y_norm = normalize_y(batch.y, target_mean, target_std)
        pred_raw = denormalize_y(pred_norm, target_mean, target_std)

        raw_abs_err_sum += (pred_raw - batch.y).abs().sum(dim=0)
        norm_abs_err_sum += (pred_norm - y_norm).abs().sum(dim=0)
        n += batch.y.size(0)

    raw_mae = raw_abs_err_sum / max(n, 1)
    norm_mae = norm_abs_err_sum / max(n, 1)
    return raw_mae.detach().cpu(), norm_mae.detach().cpu()
