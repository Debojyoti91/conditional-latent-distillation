"""Graph-autoencoder pretraining, joint conductivity fitting and evaluation."""

import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import bond_feature_dim
from .models import JointGraphAER


def make_loader(graphs, batch_size, shuffle=False):
    if len(graphs) == 0:
        raise ValueError("Empty graph list.")
    batch_size = max(1, min(batch_size, len(graphs)))
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def print_metrics(metrics):
    print(f"{metrics['label']}:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")


def weighted_supervised_loss(y_hat, y_true, sample_weight, alpha=0.65):
    y_hat = y_hat.view(-1, 1)
    y_true = y_true.view(-1, 1)
    sample_weight = sample_weight.view(-1, 1).to(y_hat.device).float()
    sl1_each = F.smooth_l1_loss(y_hat, y_true, reduction="none")
    mse_each = F.mse_loss(y_hat, y_true, reduction="none")
    per_sample = alpha * sl1_each + (1.0 - alpha) * mse_each
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-08)


def augment_train_batch(data, cfg):
    """Light regularizing augmentation for ONE training batch: random edge dropout
    plus small Gaussian noise on the already-scaled AUX features. Only ever called
    from train_single_joint's training loop -- never from pretrain_encoder or
    eval_model/eval_ensemble, so evaluation and AE reconstruction targets stay clean.
    """
    if not cfg.get("use_graph_augmentation", False):
        return data
    edge_drop_p = float(cfg.get("edge_drop_p", 0.0))
    if edge_drop_p > 0.0 and data.edge_index.numel() > 0:
        num_edges = data.edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=data.edge_index.device) >= edge_drop_p
        if keep_mask.sum().item() > 0:
            data.edge_index = data.edge_index[:, keep_mask]
            data.edge_attr = data.edge_attr[keep_mask]
    aux_noise_std = float(cfg.get("aux_noise_std", 0.0))
    if (
        aux_noise_std > 0.0
        and hasattr(data, "num_feats")
        and (data.num_feats.numel() > 0)
    ):
        data.num_feats = (
            data.num_feats + torch.randn_like(data.num_feats) * aux_noise_std
        )
    return data


def inverse_target(y_transformed, *, target_transformer):
    """
    Safely inverse-transform model outputs back to original conductivity scale.
    Protects against NaN/Inf and extreme transformed predictions.
    """
    y_transformed = np.asarray(y_transformed, dtype=np.float64).reshape(-1, 1)
    y_transformed = np.nan_to_num(y_transformed, nan=0.0, posinf=10.0, neginf=-10.0)
    y_transformed = np.clip(y_transformed, -10, 10)
    y_original = target_transformer.inverse_transform(y_transformed).reshape(-1)
    finite_mask = np.isfinite(y_original)
    if np.any(finite_mask):
        finite_vals = y_original[finite_mask]
        fill_nan = float(np.nanmedian(finite_vals))
        fill_pos = float(np.nanmax(finite_vals))
        fill_neg = float(np.nanmin(finite_vals))
    else:
        fill_nan = 0.0
        fill_pos = 1.0
        fill_neg = 0.0
    y_original = np.nan_to_num(
        y_original, nan=fill_nan, posinf=fill_pos, neginf=fill_neg
    )
    return y_original


def clean_metric_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return (None, None)
    return (y_true[mask], y_pred[mask])


@torch.no_grad()
def eval_model(model, graphs, cfg, label="eval", *, device, target_transformer):
    model.eval()
    loader = make_loader(graphs, cfg["batch_size"], shuffle=False)
    pred_t, true_t = ([], [])
    for data in loader:
        data = data.to(device)
        y_hat, _, _, _ = model(data)
        pred_t.append(y_hat.detach().cpu().numpy().reshape(-1))
        true_t.append(data.y.detach().cpu().numpy().reshape(-1))
    pred_t = np.concatenate(pred_t)
    true_t = np.concatenate(true_t)
    pred_t = np.nan_to_num(pred_t, nan=0.0, posinf=10.0, neginf=-10.0)
    true_t = np.nan_to_num(true_t, nan=0.0, posinf=10.0, neginf=-10.0)
    pred = inverse_target(pred_t, target_transformer=target_transformer)
    true = inverse_target(true_t, target_transformer=target_transformer)
    true_clean, pred_clean = clean_metric_arrays(true, pred)
    if true_clean is None:
        print(
            f"WARNING: {label} evaluation failed because predictions contain NaN/Inf."
        )
        return {
            "label": label,
            "mae": 999.0,
            "mse": 999.0,
            "rmse": 999.0,
            "r2": -999.0,
            "y_true": true,
            "y_pred": pred,
        }
    mae = float(mean_absolute_error(true_clean, pred_clean))
    mse = float(mean_squared_error(true_clean, pred_clean))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(true_clean, pred_clean))
    return {
        "label": label,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_true": true,
        "y_pred": pred,
    }


@torch.no_grad()
def eval_ensemble(models, graphs, cfg, label="eval", *, device, target_transformer):
    for m in models:
        m.eval()
    loader = make_loader(graphs, cfg["batch_size"], shuffle=False)
    pred_t_all, pred_std_t_all, true_t_all = ([], [], [])
    for data in loader:
        data = data.to(device)
        member_preds = []
        for model in models:
            y_hat, _, _, _ = model(data)
            y_hat_np = y_hat.detach().cpu().numpy().reshape(-1)
            y_hat_np = np.nan_to_num(y_hat_np, nan=0.0, posinf=10.0, neginf=-10.0)
            member_preds.append(y_hat_np)
        member_preds = np.vstack(member_preds)
        pred_t_all.append(np.mean(member_preds, axis=0))
        pred_std_t_all.append(np.std(member_preds, axis=0))
        true_t_all.append(data.y.detach().cpu().numpy().reshape(-1))
    pred_t_all = np.concatenate(pred_t_all)
    pred_std_t_all = np.concatenate(pred_std_t_all)
    true_t_all = np.concatenate(true_t_all)
    pred_t_all = np.nan_to_num(pred_t_all, nan=0.0, posinf=10.0, neginf=-10.0)
    true_t_all = np.nan_to_num(true_t_all, nan=0.0, posinf=10.0, neginf=-10.0)
    pred = inverse_target(pred_t_all, target_transformer=target_transformer)
    true = inverse_target(true_t_all, target_transformer=target_transformer)
    true_clean, pred_clean = clean_metric_arrays(true, pred)
    if true_clean is None:
        print(
            f"WARNING: {label} ensemble evaluation failed because predictions contain NaN/Inf."
        )
        return {
            "label": label,
            "mae": 999.0,
            "mse": 999.0,
            "rmse": 999.0,
            "r2": -999.0,
            "y_true": true,
            "y_pred": pred,
            "pred_std_transformed_space": pred_std_t_all,
        }
    mae = float(mean_absolute_error(true_clean, pred_clean))
    mse = float(mean_squared_error(true_clean, pred_clean))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(true_clean, pred_clean))
    return {
        "label": label,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_true": true,
        "y_pred": pred,
        "pred_std_transformed_space": pred_std_t_all,
    }


def pretrain_encoder(train_graphs, cfg, bin_dim, cont_dim, seed=2023, *, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sample = train_graphs[0]
    node_in = sample.x.shape[1]
    edge_in = (
        sample.edge_attr.shape[1]
        if sample.edge_attr is not None and sample.edge_attr.numel() > 0
        else bond_feature_dim(cfg)
    )
    model = JointGraphAER(node_in, edge_in, cfg, num_extra_feats=0).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=cfg["ae_lr"],
        weight_decay=cfg["ae_weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=40, T_mult=1
    )
    loader = make_loader(train_graphs, cfg["ae_batch_size"], shuffle=True)
    bce = nn.BCEWithLogitsLoss()
    mse_recon = nn.MSELoss()
    for epoch in range(cfg["ae_epochs"]):
        model.train()
        total_loss, steps = (0.0, 0)
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            _, x_logits, _, _ = model(data)
            x_bin = data.x[:, :bin_dim]
            x_cont = data.x[:, bin_dim : bin_dim + cont_dim]
            xl_bin = x_logits[:, :bin_dim]
            xl_cont = x_logits[:, bin_dim : bin_dim + cont_dim]
            loss = bce(xl_bin, x_bin) + mse_recon(xl_cont, x_cont)
            if not torch.isfinite(loss):
                print("WARNING: non-finite AE loss detected; skipping batch.")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        scheduler.step(epoch)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"AE epoch {epoch + 1:04d}/{cfg['ae_epochs']} | loss={total_loss / max(1, steps):.6f}"
            )
    return {
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
        "node_in": node_in,
        "edge_in": edge_in,
    }


def train_single_joint(
    train_graphs,
    val_graphs,
    ae_init,
    cfg,
    bin_dim,
    cont_dim,
    seed_offset=0,
    *,
    device,
    target_transformer,
    aux_columns,
    base_seed,
):
    seed = base_seed + seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader = make_loader(train_graphs, cfg["batch_size"], shuffle=True)
    model = JointGraphAER(
        ae_init["node_in"], ae_init["edge_in"], cfg, num_extra_feats=len(aux_columns)
    ).to(device)
    model.encoder.load_state_dict(ae_init["encoder"])
    model.decoder.load_state_dict(ae_init["decoder"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["T0"], T_mult=cfg["Tmult"]
    )
    swa_start_epoch = int((1.0 - cfg["swa_ratio"]) * cfg["epochs"])
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg["lr"] * 0.1)
    bce = nn.BCEWithLogitsLoss()
    mse_recon = nn.MSELoss()
    hist = {
        "epoch": [],
        "train_loss": [],
        "train_sup_loss": [],
        "train_recon_loss": [],
        "val_rmse": [],
        "val_r2": [],
    }
    best_rmse = float("inf")
    best_state = None
    print(
        f"[DEBUG] Encoder latent={cfg['enc_hidden']} | Numeric AUX={len(aux_columns)} | Mean+max pooling={cfg.get('use_mean_max_pool', True)} | Regressor input={model.regressor.blocks[0][0].in_features}"
    )
    print(
        f"[DEBUG] Weighted loss={cfg.get('use_sample_weight', True)} | high_cond_threshold={cfg.get('high_cond_threshold', 0.09)} | high_cond_weight={cfg.get('high_cond_weight', 3.0)}"
    )
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, total_sup, total_recon, steps = (0.0, 0.0, 0.0, 0)
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            y_hat, x_logits, _, _ = model(data)
            if not torch.isfinite(y_hat).all():
                print("WARNING: non-finite y_hat detected; skipping batch.")
                continue
            y_true = data.y.view(-1, 1)
            sample_weight = getattr(data, "sample_weight", None)
            sample_weight = (
                torch.ones_like(y_true, device=device)
                if sample_weight is None
                else sample_weight.view(-1, 1).to(device)
            )
            sup = weighted_supervised_loss(
                y_hat, y_true, sample_weight, alpha=cfg["alpha"]
            )
            x_bin = data.x[:, :bin_dim]
            x_cont = data.x[:, bin_dim : bin_dim + cont_dim]
            xl_bin = x_logits[:, :bin_dim]
            xl_cont = x_logits[:, bin_dim : bin_dim + cont_dim]
            recon = bce(xl_bin, x_bin) + mse_recon(xl_cont, x_cont)
            loss = cfg["sup_weight"] * sup + cfg["recon_weight"] * recon
            if not torch.isfinite(loss):
                print("WARNING: non-finite loss detected; skipping batch.")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_sup += sup.item()
            total_recon += recon.item()
            steps += 1
        if epoch < swa_start_epoch:
            scheduler.step(epoch)
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == cfg["epochs"] - 1:
            val_metrics = eval_model(
                model,
                val_graphs,
                cfg,
                label="val",
                device=device,
                target_transformer=target_transformer,
            )
            val_rmse, val_r2 = (val_metrics["rmse"], val_metrics["r2"])
            hist["epoch"].append(epoch + 1)
            hist["train_loss"].append(total_loss / max(1, steps))
            hist["train_sup_loss"].append(total_sup / max(1, steps))
            hist["train_recon_loss"].append(total_recon / max(1, steps))
            hist["val_rmse"].append(val_rmse)
            hist["val_r2"].append(val_r2)
            if np.isfinite(val_rmse) and val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                f"Joint epoch {epoch + 1:04d}/{cfg['epochs']} | loss={total_loss / max(1, steps):.6f} | sup={total_sup / max(1, steps):.6f} | recon={total_recon / max(1, steps):.6f} | val RMSE={val_rmse:.6f} | val R2={val_r2:.4f}"
            )
    best_model = JointGraphAER(
        ae_init["node_in"], ae_init["edge_in"], cfg, num_extra_feats=len(aux_columns)
    ).to(device)
    if best_state is not None:
        best_model.load_state_dict(best_state)
    else:
        print(
            "WARNING: no finite best validation state found; using final model state."
        )
        best_model.load_state_dict(model.state_dict())
    try:
        update_bn(train_loader, swa_model, device=device)
        swa_metrics = eval_model(
            swa_model,
            val_graphs,
            cfg,
            label="swa_val",
            device=device,
            target_transformer=target_transformer,
        )
    except Exception as e:
        print("WARNING: SWA evaluation failed; using best epoch model.", repr(e))
        swa_metrics = {
            "label": "swa_val",
            "rmse": float("inf"),
            "r2": -999.0,
            "mae": 999.0,
            "mse": 999.0,
            "y_true": np.array([]),
            "y_pred": np.array([]),
        }
    best_metrics = eval_model(
        best_model,
        val_graphs,
        cfg,
        label="best_epoch_val",
        device=device,
        target_transformer=target_transformer,
    )
    if best_metrics["rmse"] <= swa_metrics["rmse"]:
        final_model, chosen, chosen_metrics = (best_model, "best_epoch", best_metrics)
    else:
        final_model, chosen, chosen_metrics = (swa_model, "swa", swa_metrics)
    print(f"Chosen model: {chosen}")
    print_metrics(chosen_metrics)
    return (final_model, hist, chosen_metrics)
