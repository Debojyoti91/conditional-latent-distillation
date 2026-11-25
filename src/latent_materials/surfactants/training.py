"""Heteroscedastic graph-surrogate fitting and evaluation."""

import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from .models import gaussian_nll


@torch.no_grad()
def _to_2col(arr):
    arr = np.array(arr)
    if arr.ndim == 1:
        n = arr.shape[0]
        if n % 2 == 0:
            arr = arr.reshape(-1, 2)
        else:
            arr = arr.reshape(2, -1).T
    elif arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    elif arr.shape[1] != 2 and arr.shape[0] % 2 == 0 and (arr.shape[1] == 1):
        arr = arr.reshape(-1, 2)
    return arr


@torch.no_grad()
def eval_loader(model, ld, device):
    model.eval()
    preds, trues = ([], [])
    for d in ld:
        d = d.to(device)
        mu, _ = model(d)
        p = mu.detach().cpu().numpy()
        t = d.y.detach().cpu().numpy()
        p = _to_2col(p)
        t = _to_2col(t)
        preds.append(p)
        trues.append(t)
    preds, trues = (np.vstack(preds), np.vstack(trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2, axis=0))
    mae = np.mean(np.abs(preds - trues), axis=0)
    r2 = [r2_score(trues[:, i], preds[:, i]) for i in range(trues.shape[1])]
    return (rmse, mae, r2)


def train_one_config(cfg, tr_ds, va_ds, te_ds, device, *, model_class, output_dir):
    tr_ld = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=max(128, cfg["batch_size"]), shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=max(128, cfg["batch_size"]), shuffle=False)
    sample = next(iter(tr_ld))
    atom_dim, aux_dim = (sample.x.size(1), sample.aux.size(1))
    model = model_class(
        atom_dim, aux_dim, layer_sizes=cfg["layer_sizes"], dropout=cfg["dropout"]
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    best_val, wait = (float("inf"), 0)
    best_state = None
    hist = {"epoch": [], "train_loss": [], "val_rmse": [], "val_r2": []}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, steps = (0.0, 0)
        for d in tr_ld:
            d = d.to(device)
            mu, logvar = model(d)
            y = d.y
            if y.dim() == 1:
                y = y.view(-1, 2)
            loss = gaussian_nll(mu, logvar, y)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.item()
            steps += 1
        tr_loss = total / max(1, steps)
        va_rmse, _, va_r2 = eval_loader(model, va_ld, device)
        val_mean = float(np.mean(va_rmse))
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss)
        hist["val_rmse"].append(val_mean)
        hist["val_r2"].append(float(np.mean(va_r2)))
        if val_mean < best_val:
            best_val = val_mean
            wait = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    tr_rmse, tr_mae, tr_r2 = eval_loader(model, tr_ld, device)
    va_rmse, va_mae, va_r2 = eval_loader(model, va_ld, device)
    te_rmse, te_mae, te_r2 = eval_loader(model, te_ld, device)
    run_tag = f"sizes{cfg['layer_sizes']}_do{cfg['dropout']}_lr{cfg['lr']}_wd{cfg['weight_decay']}".replace(
        " ", ""
    )
    curve_path = os.path.join(output_dir, f"curve_{run_tag}.png")
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(
        hist["epoch"], hist["train_loss"], color="tab:blue", lw=1.8, label="Train Loss"
    )
    ax1.plot(
        hist["epoch"], hist["val_rmse"], color="tab:orange", lw=1.8, label="Val RMSE"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / RMSE")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(
        hist["epoch"],
        hist["val_r2"],
        color="tab:green",
        lw=1.4,
        alpha=0.6,
        label="Val R²",
    )
    ax2.set_ylabel("Validation R²")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Training Curve – {run_tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    res = dict(cfg)
    res.update(
        {
            "val_rmse_aw": float(va_rmse[0]),
            "val_rmse_pcmc": float(va_rmse[1]),
            "val_rmse_mean": float(np.mean(va_rmse)),
            "train_rmse_aw": float(tr_rmse[0]),
            "train_rmse_pcmc": float(tr_rmse[1]),
            "test_rmse_aw": float(te_rmse[0]),
            "test_rmse_pcmc": float(te_rmse[1]),
            "train_mae_aw": float(tr_mae[0]),
            "train_mae_pcmc": float(tr_mae[1]),
            "test_mae_aw": float(te_mae[0]),
            "test_mae_pcmc": float(te_mae[1]),
            "test_r2_aw": float(te_r2[0]),
            "test_r2_pcmc": float(te_r2[1]),
            "epochs_ran": len(hist["epoch"]),
            "curve_path": curve_path,
        }
    )
    print(
        f"Train RMSE {np.round(tr_rmse, 3)} | Val RMSE {np.round(va_rmse, 3)} | Test RMSE {np.round(te_rmse, 3)}"
    )
    print(f"Test MAE {np.round(te_mae, 3)} | Test R² {np.round(te_r2, 3)}")
    return (model, res)
