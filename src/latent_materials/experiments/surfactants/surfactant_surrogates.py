"""Final manuscript-matched pC20, two-output GNN and CV workflow implementations. Original final checkpoints/prediction arrays are not supplied. New training is a new realization, not an exact replacement of the reported run.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path
from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the surfactant surrogates experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = (
        str(root / "IEMS_GEN_files/AEM/CVAE") if "surfactant" == "aem" else str(root)
    )
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    import shutil

    src = str(root / "Datasets/surfpro_literature(version 1.1).xlsx")
    dst = str(root / "data/surfpro_literature(version 1.1).xlsx")
    shutil.copy(src, dst)
    print("File copied successfully.")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os

    data_dir = str(root / "data")
    os.makedirs(data_dir, exist_ok=True)
    main_path = f"{data_dir}/surfpro_imputed.csv"
    lit_path = f"{data_dir}/surfpro_literature(version 1.1).xlsx"
    df = pd.read_csv(main_path)
    if "types" in df.columns and "type" not in df.columns:
        df = df.rename(columns={"types": "type"})
    df = df.copy()
    df["is_PO"] = 0
    lit_df = pd.read_excel(lit_path)
    po_ids = {1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639}
    po_df = lit_df[lit_df["ID"].isin(po_ids)].copy()
    po_df = po_df.rename(columns={"Surfactant_Type": "type", "pC20": "PC20"})
    po_df = po_df[
        [
            "SMILES",
            "type",
            "pCMC",
            "AW_ST_CMC",
            "Gamma_max",
            "Area_min",
            "Pi_CMC",
            "PC20",
        ]
    ]
    po_df["is_PO"] = 1
    for c in df.columns:
        if c not in po_df.columns:
            po_df[c] = pd.NA
    po_df = po_df[df.columns]
    df_full = pd.concat([df, po_df], ignore_index=True)
    train_df, test_df = train_test_split(
        df_full, test_size=0.2, stratify=df_full["is_PO"], random_state=42
    )
    minor = train_df[train_df["is_PO"] == 1]
    major = train_df[train_df["is_PO"] == 0]
    target_frac = 0.05
    target_n = int(len(train_df) * target_frac)
    needed = max(0, target_n - len(minor))
    if needed > 0:
        dup = minor.sample(needed, replace=True, random_state=42)
        train_bal = pd.concat([major, minor, dup], ignore_index=True)
    else:
        train_bal = train_df.copy()
    print("Original train size:", len(train_df))
    print("Balanced train size:", len(train_bal))
    print("PO in train (original):", int((train_df["is_PO"] == 1).sum()))
    print("PO in train (balanced):", int((train_bal["is_PO"] == 1).sum()))
    print("PO fraction (balanced):", (train_bal["is_PO"] == 1).mean())
    print("Test size (untouched):", len(test_df))
    print("PO in test:", int((test_df["is_PO"] == 1).sum()))
    train_path = f"{data_dir}/surfpro_train80_PO_balanced.csv"
    test_path = f"{data_dir}/surfpro_test20_real.csv"
    train_bal.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Saved:")
    print("  Train:", train_path)
    print("  Test :", test_path)
    train_bal = train_bal.rename(columns={"pC20": "PC20"})
    test_df = test_df.rename(columns={"pC20": "PC20"})
    print(train_bal.columns)
    print(test_df.columns)
    train_path = f"{data_dir}/surfpro_train80_PO_balanced.csv"
    test_path = f"{data_dir}/surfpro_test20_real.csv"
    train_bal.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Saved corrected files:")
    print("  Train:", train_path)
    print("  Test :", test_path)
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error

    df = train_bal.copy()
    df = df[df["PC20"].notna()].reset_index(drop=True)
    exclude = {"SMILES", "PC20", "Unnamed: 0", "is_PO"}
    cand = [c for c in df.columns if c not in exclude and (not c.endswith("_std"))]
    num_df = df[cand].apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in num_df.columns if not num_df[c].isna().all()]
    df[num_cols] = num_df[num_cols]
    print("Auxiliary numeric features:", num_cols)
    from latent_materials.surfactants.data import pc20_graph as mol_to_graph

    graphs = []
    for _, r in df.iterrows():
        aux = r[num_cols].values.astype(np.float32)
        if np.isnan(aux).any():
            continue
        g = mol_to_graph(r["SMILES"], aux, r["PC20"])
        if g is not None:
            graphs.append(g)
    print("Graphs prepared (train-known):", len(graphs))
    from latent_materials.surfactants.models import PC20GraphRegressor as DeeperGNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = {"hidden": [64, 128], "lr": [0.001, 0.0003], "batch": [32, 64]}

    def run_cv(hidden, lr, batch):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses, r2s = ([], [])
        for tr, va in kf.split(graphs):
            tr_set = [graphs[i] for i in tr]
            va_set = [graphs[i] for i in va]
            tr_loader = DataLoader(tr_set, batch_size=batch, shuffle=True)
            va_loader = DataLoader(va_set, batch_size=batch)
            model = DeeperGNN(hidden, len(num_cols)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(80):
                model.train()
                for d in tr_loader:
                    d = d.to(device)
                    opt.zero_grad()
                    loss = F.mse_loss(model(d), d.y.to(device))
                    loss.backward()
                    opt.step()
            model.eval()
            yt, yp = ([], [])
            with torch.no_grad():
                for d in va_loader:
                    d = d.to(device)
                    yp.extend(model(d).cpu().numpy())
                    yt.extend(d.y.cpu().numpy())
            rmse = np.sqrt(mean_squared_error(yt, yp))
            r2 = r2_score(yt, yp)
            rmses.append(rmse)
            r2s.append(r2)
        return (np.mean(rmses), np.mean(r2s))

    results = []
    for h in grid["hidden"]:
        for lr in grid["lr"]:
            for b in grid["batch"]:
                rmse, r2 = run_cv(h, lr, b)
                results.append((h, lr, b, rmse, r2))
                print(h, lr, b, "RMSE:", round(rmse, 3), "R2:", round(r2, 3))
    best = sorted(results, key=lambda x: x[3])[0]
    print("\nBest (hidden, lr, batch, RMSE, R2):", best)
    best_h, best_lr, best_b, _, _ = best
    loader = DataLoader(graphs, batch_size=best_b, shuffle=True)
    model = DeeperGNN(best_h, len(num_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best_lr)
    loss_curve = []
    for epoch in range(1, 201):
        model.train()
        epoch_loss = 0.0
        for d in loader:
            d = d.to(device)
            opt.zero_grad()
            pred = model(d)
            loss = F.mse_loss(pred, d.y.to(device))
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * d.num_graphs
        epoch_loss /= len(graphs)
        loss_curve.append(epoch_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | MSE: {epoch_loss:.4f}")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE (PC20)")
    plt.title("GNN Training Curve for PC20 Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    test_known = test_df[test_df["PC20"].notna()].copy()
    test_graphs = []
    y_true = []
    for _, r in test_known.iterrows():
        aux = r[num_cols].values.astype(np.float32)
        if np.isnan(aux).any():
            continue
        g = mol_to_graph(r["SMILES"], aux, r["PC20"])
        if g is not None:
            test_graphs.append(g)
            y_true.append(r["PC20"])
    test_loader = DataLoader(test_graphs, batch_size=best_b)
    model.eval()
    y_pred = []
    with torch.no_grad():
        for d in test_loader:
            d = d.to(device)
            y_pred.extend(model(d).cpu().numpy())
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true + y_pred), max(y_true + y_pred)]
    plt.plot(lims, lims, "k--")
    plt.xlabel("True PC20")
    plt.ylabel("Predicted PC20")
    plt.title("PC20: True vs Predicted (Test Set)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error

    out_dir = str(root / "data")
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    rmse_test = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_test = r2_score(y_true, y_pred)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.linewidth": 1.25,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    curve_color = "#0072B2"
    scatter_color = "#D55E00"
    edge_color = "#1A1A1A"
    line_color = "#333333"
    grid_color = "#B8B8B8"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    ax = axes[0]
    epochs = np.arange(1, len(loss_curve) + 1)
    ax.plot(epochs, loss_curve, color=curve_color, linewidth=2.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE (pC20)")
    ax.set_title("Training curve", pad=8)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3, color=grid_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.14, 1.05, "A", transform=ax.transAxes, fontsize=14, ha="left", va="bottom"
    )
    ax = axes[1]
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    pad = 0.07 * (max_val - min_val + 1e-09)
    lims = [min_val - pad, max_val + pad]
    ax.scatter(
        y_true,
        y_pred,
        s=46,
        alpha=0.78,
        color=scatter_color,
        edgecolor=edge_color,
        linewidth=0.35,
    )
    ax.plot(lims, lims, linestyle="--", linewidth=1.5, color=line_color)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True pC20")
    ax.set_ylabel("Predicted pC20")
    ax.set_title("Test-set parity plot", pad=8)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3, color=grid_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stats_text = f"RMSE = {rmse_test:.3f}\n$R^2$ = {r2_test:.3f}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        bbox=dict(
            boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.75", alpha=0.95
        ),
    )
    ax.text(
        -0.14, 1.0, "B", transform=ax.transAxes, fontsize=14, ha="left", va="bottom"
    )
    fig.suptitle(
        "Training behavior and test-set predictive performance of the pC20 GNN surrogate",
        fontsize=15.5,
        y=1.0,
    )
    fig.text(
        0.5,
        -0.02,
        "The dashed line in Panel B represents perfect agreement between predicted and true pC20 values.",
        ha="center",
        va="bottom",
        fontsize=10.2,
    )
    fig.tight_layout(rect=[0, 0.04, 0.9, 0.95])
    out_png = os.path.join(out_dir, "pc20_training_parity_combined.png")
    out_pdf = os.path.join(out_dir, "pc20_training_parity_combined.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    from latent_materials.exports import export_ensemble_snapshot

    export_ensemble_snapshot(
        Path(out_dir) / "final_pc20_run",
        models=[model],
        config={
            "architecture": "DeeperGNN",
            "hidden": best_h,
            "learning_rate": best_lr,
            "batch_size": best_b,
        },
        aux_columns=num_cols,
        aux_scaler=None,
        target_transformer=None,
        evaluations={"test": {"y_true": y_true, "y_pred": y_pred}},
        metadata={
            "atom_features": ["atomic_number", "degree", "total_H", "aromatic"],
            "scope": "New realization of the final supplied pC20 training source; not the missing historical checkpoint.",
        },
    )

    def predict_missing_pc20(df_in):
        df_out = df_in.copy()
        missing = df_out[df_out["PC20"].isna()].copy()
        if len(missing) == 0:
            return df_out
        pred_graphs = []
        idx_map = []
        for idx, r in missing.iterrows():
            aux = r[num_cols].values.astype(np.float32)
            if np.isnan(aux).any():
                continue
            g = mol_to_graph(r["SMILES"], aux, 0.0)
            if g is not None:
                pred_graphs.append(g)
                idx_map.append(idx)
        loader = DataLoader(pred_graphs, batch_size=best_b)
        model.eval()
        preds = []
        with torch.no_grad():
            for d in loader:
                d = d.to(device)
                preds.extend(model(d).cpu().numpy())
        for i, p in zip(idx_map, preds):
            df_out.loc[i, "PC20"] = float(p)
        return df_out

    train_filled = predict_missing_pc20(train_bal)
    test_filled = predict_missing_pc20(test_df)
    print("Remaining PC20 NaNs (train):", train_filled["PC20"].isna().sum())
    print("Remaining PC20 NaNs (test) :", test_filled["PC20"].isna().sum())
    train_path = f"{data_dir}/surfpro_train80.csv"
    test_path = f"{data_dir}/surfpro_test20.csv"
    train_filled.to_csv(train_path, index=False)
    test_filled.to_csv(test_path, index=False)
    print("Updated files written:")
    print("  Train:", train_path)
    print("  Test :", test_path)
    import pandas as pd, numpy as np
    import os

    data_dir = str(root / "data")
    os.makedirs(data_dir, exist_ok=True)
    from latent_materials.surfactants.training_features import (
        enrich,
    )

    train_feat = enrich(train_filled)
    test_feat = enrich(test_filled)
    train_path = f"{data_dir}/surfpro_train80.csv"
    test_path = f"{data_dir}/surfpro_test20.csv"
    train_feat.to_csv(train_path, index=False)
    test_feat.to_csv(test_path, index=False)
    print("Feature engineering complete.")
    print("Train:", train_feat.shape)
    print("Test :", test_feat.shape)
    print("Saved:")
    print(train_path)
    print(test_path)
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
    import os

    data_dir = str(root / "data")
    train_path = f"{data_dir}/surfpro_train80.csv"
    test_path = f"{data_dir}/surfpro_test20.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    drop_base = {"Unnamed: 0", "is_PO"}
    drop_std = [c for c in train.columns if c.endswith("_std")]
    drop_cols = [c for c in drop_base | set(drop_std) if c in train.columns]
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)
    print("Dropped columns:", drop_cols)
    TARGET_COLUMNS = ["AW_ST_CMC", "pCMC"]
    X_train = train.drop(columns=TARGET_COLUMNS)
    y_train = train[TARGET_COLUMNS].copy()
    X_test = test.drop(columns=TARGET_COLUMNS)
    y_test = test[TARGET_COLUMNS].copy()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    head_cols = [c for c in X_train.columns if c.startswith("head_")]
    exclude_cols = set(head_cols) | ({"fold"} if "fold" in X_train.columns else set())
    numeric_to_scale = [c for c in numeric_cols if c not in exclude_cols]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    if len(numeric_to_scale) > 0:
        X_train_scaled[numeric_to_scale] = scaler.fit_transform(
            X_train[numeric_to_scale]
        )
        X_test_scaled[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])
    else:
        print("Warning: No numeric columns found for scaling.")
    X_train_scaled.to_csv(f"{data_dir}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{data_dir}/X_test_scaled.csv", index=False)
    y_train.to_csv(f"{data_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{data_dir}/y_test.csv", index=False)
    joblib.dump(scaler, f"{data_dir}/scaler.pkl")
    print("Scaling complete.")
    print(f"Scaled feature shape (train): {X_train_scaled.shape}")
    print(f"Scaled feature shape (test):  {X_test_scaled.shape}")
    print(f"Target shape (train): {y_train.shape}")
    print(f"Target shape (test):  {y_test.shape}")
    print("Scaler stored at:", f"{data_dir}/scaler.pkl")
    print("\nFinal scaled feature columns:")
    print(X_train_scaled.columns.tolist())
    import os, numpy as np, pandas as pd, torch
    from torch_geometric.loader import DataLoader
    from torch.utils.data import random_split
    from rdkit.Chem.rdchem import HybridizationType as Hyb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = str(root / "data")
    TARGET_COLUMNS = ["AW_ST_CMC", "pCMC"]
    EPOCHS, PATIENCE = (300, 25)
    BATCH_TRAIN, BATCH_EVAL = (64, 128)
    LR, WD, HIDDEN = (0.002, 0.0001, 128)
    CKPT = f"{base}/gnn_hetero_best.pt"
    Xtr = pd.read_csv(f"{base}/X_train_scaled.csv")
    Ytr = pd.read_csv(f"{base}/y_train.csv")
    Xte = pd.read_csv(f"{base}/X_test_scaled.csv")
    Yte = pd.read_csv(f"{base}/y_test.csv")
    assert "SMILES" in Xtr.columns, "Column 'SMILES' not found in X_train_scaled.csv"
    assert all(
        (c in Ytr.columns for c in TARGET_COLUMNS)
    ), f"y_train missing {TARGET_COLUMNS}"
    DROP_META = ["SMILES", "type"]
    AUX_COLS = [c for c in Xtr.columns if c not in DROP_META]
    Xtr[AUX_COLS] = Xtr[AUX_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xte[AUX_COLS] = Xte[AUX_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 11, 3]
    HYB_LIST = [Hyb.SP, Hyb.SP2, Hyb.SP3]
    from latent_materials.surfactants.data import SurfactantDataset as SurfDataset

    tr_full = SurfDataset(Xtr, Ytr, aux_columns=AUX_COLS, target_columns=TARGET_COLUMNS)
    te_ds = SurfDataset(Xte, Yte, aux_columns=AUX_COLS, target_columns=TARGET_COLUMNS)
    n_val = max(1, int(len(tr_full) * 0.1))
    n_tr = len(tr_full) - n_val
    tr_ds, va_ds = random_split(
        tr_full, [n_tr, n_val], generator=torch.Generator().manual_seed(42)
    )
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_TRAIN, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_EVAL, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=BATCH_EVAL, shuffle=False)
    print(
        f"Datasets ready | Train: {len(tr_ds)} | Val: {len(va_ds)} | Test: {len(te_ds)}"
    )
    import os, itertools, time, random, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import r2_score

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = str(root / "data")
    Xtr = pd.read_csv(f"{base}/X_train_scaled.csv")
    Ytr = pd.read_csv(f"{base}/y_train.csv")
    Xte = pd.read_csv(f"{base}/X_test_scaled.csv")
    Yte = pd.read_csv(f"{base}/y_test.csv")
    TARGET_COLUMNS = ["AW_ST_CMC", "pCMC"]
    from latent_materials.surfactants.models import SearchGINRegressor as GINHetero
    from latent_materials.surfactants.training import eval_loader, train_one_config
    from latent_materials.surfactants.models import gaussian_nll

    grid = {
        "layer_sizes": [
            [64, 64, 64],
            [128, 128, 128],
            [256, 256, 256],
            [64, 128, 256],
            [128, 256, 128],
            [64, 128, 256, 128, 64],
        ],
        "dropout": [0.0, 0.1],
        "lr": [0.002, 0.001],
        "weight_decay": [0.0001, 0.0003],
        "batch_size": [64],
        "epochs": [200],
        "patience": [25],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total configurations: {len(combos)}")
    results = []
    best_overall = None
    start_all = time.time()
    for idx, values in enumerate(combos, 1):
        cfg = {k: v for k, v in zip(keys, values)}
        print(f"\n=== Config {idx}/{len(combos)} ===")
        print(cfg)
        t0 = time.time()
        model, res = train_one_config(
            cfg, tr_ds, va_ds, te_ds, device, model_class=GINHetero, output_dir=base
        )
        res["seconds"] = round(time.time() - t0, 2)
        results.append(res)
        if best_overall is None or res["val_rmse_mean"] < best_overall["val_rmse_mean"]:
            best_overall = res
            ckpt = os.path.join(base, "gnn_grid_best_varwidth.pt")
            torch.save(model.state_dict(), ckpt)
            best_overall["ckpt_path"] = ckpt
            print(f"New best saved:  {ckpt}")
    df = pd.DataFrame(results)
    csv_path = os.path.join(base, "grid_results_varwidth.csv")
    df.to_csv(csv_path, index=False)
    print("\n=== Grid search complete ===")
    print(f"Total time: {round(time.time() - start_all, 2)} s")
    print(f"Best configuration:\n{best_overall}")
    print(f"Results saved: {csv_path}")
    print("\n Plotting best configuration curve summary...")
    import matplotlib.image as mpimg

    img = mpimg.imread(best_overall["curve_path"])
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Best Model Training Curve Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(base, "best_model_curve_summary.png"), dpi=150)
    plt.show()
    print(
        f"Saved best curve summary: {os.path.join(base, 'best_model_curve_summary.png')}"
    )
    import os, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt

    base = str(root / "data")
    csv_path = os.path.join(base, "grid_results_varwidth.csv")
    best_ckpt = os.path.join(base, "gnn_grid_best_varwidth.pt")
    df = pd.read_csv(csv_path)
    best_row = df.loc[df["val_rmse_mean"].idxmin()]
    print("\n=== Best configuration reloaded from CSV ===")
    print(best_row)
    layer_sizes = eval(best_row["layer_sizes"])
    dropout = float(best_row["dropout"])
    print(f"\nUsing checkpoint: {best_ckpt}")
    print(f"Layer sizes: {layer_sizes} | Dropout: {dropout}")
    from latent_materials.surfactants.models import RefinedGINRegressor as GINHetero

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = next(iter(te_ld))
    atom_dim, aux_dim = (sample.x.size(1), sample.aux.size(1))
    model = GINHetero(atom_dim, aux_dim, layer_sizes=layer_sizes, dropout=dropout).to(
        device
    )
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    print(f"Loaded best model from: {best_ckpt}")
    y_true_all, y_pred_all, y_std_all = ([], [], [])
    for d in te_ld:
        d = d.to(device)
        mu, logvar = model(d)
        sigma = torch.sqrt(torch.exp(torch.clamp(logvar, -10, 10)))
        y_true = d.y.cpu().numpy()
        y_pred = mu.detach().cpu().numpy()
        y_std = sigma.detach().cpu().numpy()
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 2)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 2)
        if y_std.ndim == 1:
            y_std = y_std.reshape(-1, 2)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        y_std_all.append(y_std)
    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)
    y_std_all = np.vstack(y_std_all)
    errors = np.abs(y_true_all - y_pred_all)
    titles = ["AW_ST_CMC (mN/m)", "pCMC"]
    plt.figure(figsize=(12, 5))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.scatter(
            y_true_all[:, i], y_pred_all[:, i], alpha=0.6, s=25, color="royalblue"
        )
        lims = [
            min(y_true_all[:, i].min(), y_pred_all[:, i].min()),
            max(y_true_all[:, i].max(), y_pred_all[:, i].max()),
        ]
        plt.plot(lims, lims, "k--", lw=1)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{titles[i]} — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(base, "best_true_vs_pred.png"), dpi=150)
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.hist(errors[:, 0], bins=30, alpha=0.6, label="AW_ST_CMC", color="steelblue")
    plt.hist(errors[:, 1], bins=30, alpha=0.6, label="pCMC", color="darkorange")
    plt.xlabel("|Error|")
    plt.ylabel("Count")
    plt.title("Absolute Error Distribution — Test Set")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(base, "best_error_hist.png"), dpi=150)
    plt.show()
    plt.figure(figsize=(6, 5))
    plt.scatter(
        y_std_all[:, 0], errors[:, 0], alpha=0.5, label="AW_ST_CMC", color="steelblue"
    )
    plt.scatter(
        y_std_all[:, 1], errors[:, 1], alpha=0.5, label="pCMC", color="darkorange"
    )
    plt.xlabel("Predicted σ")
    plt.ylabel("|Prediction Error|")
    plt.title("Uncertainty Calibration — Test Set")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(base, "best_uncertainty_calib.png"), dpi=150)
    plt.show()
    plt.figure(figsize=(12, 5))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.scatter(
            y_pred_all[:, i],
            y_true_all[:, i] - y_pred_all[:, i],
            alpha=0.6,
            s=25,
            color="slategray",
        )
        plt.axhline(0, color="k", linestyle="--", lw=1)
        plt.xlabel("Predicted Value")
        plt.ylabel("Residual (True − Pred)")
        plt.title(f"Residuals — {titles[i]}")
    plt.tight_layout()
    plt.savefig(os.path.join(base, "best_residuals.png"), dpi=150)
    plt.show()
    mean_err = np.mean(errors, axis=0)
    std_err = np.std(errors, axis=0)
    mean_std = np.mean(y_std_all, axis=0)
    print("\n=== Summary of Predictive Errors and Uncertainties ===")
    for i, name in enumerate(titles):
        print(
            f"{name}: mean|error| = {mean_err[i]:.3f}, std|error| = {std_err[i]:.3f}, meanσ = {mean_std[i]:.3f}"
        )
    print(f"\n Saved plots:")
    print(f" - {os.path.join(base, 'best_true_vs_pred.png')}")
    print(f" - {os.path.join(base, 'best_error_hist.png')}")
    print(f" - {os.path.join(base, 'best_uncertainty_calib.png')}")
    print(f" - {os.path.join(base, 'best_residuals.png')}")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    titles = ["$\\gamma_{\\mathrm{CMC}}$ (mN m$^{-1}$)", "pCMC"]
    colors = ["#0072B2", "#D55E00"]
    edge_color = "#1A1A1A"
    identity_color = "#333333"
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.linewidth": 1.25,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    for i, ax in enumerate(axes):
        true_i = y_true_all[:, i]
        pred_i = y_pred_all[:, i]
        min_val = min(true_i.min(), pred_i.min())
        max_val = max(true_i.max(), pred_i.max())
        pad = 0.07 * (max_val - min_val + 1e-09)
        lims = [min_val - pad, max_val + pad]
        ax.scatter(
            true_i,
            pred_i,
            s=42,
            alpha=0.78,
            color=colors[i],
            edgecolor=edge_color,
            linewidth=0.35,
        )
        ax.plot(
            lims,
            lims,
            linestyle="--",
            linewidth=1.5,
            color=identity_color,
            label="Ideal prediction",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True values")
        ax.set_ylabel(f"Predicted values")
        ax.set_title(f"{titles[i]}", pad=8)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.14,
            1.05,
            chr(65 + i),
            transform=ax.transAxes,
            fontsize=17,
            ha="left",
            va="bottom",
        )
    fig.suptitle(
        "Test-set prediction accuracy of the heteroscedastic GNN surrogate",
        x=0.43,
        y=0.98,
    )
    fig.text(
        0.4,
        -0.025,
        "The dashed line represents perfect agreement between predicted and true values.",
        ha="center",
        va="bottom",
        fontsize=10.2,
    )
    fig.tight_layout(rect=[0, 0.04, 0.8, 0.94])
    out_true_pred_png = os.path.join(base, "best_true_vs_pred_publication.png")
    out_true_pred_pdf = os.path.join(base, "best_true_vs_pred_publication.pdf")
    fig.savefig(out_true_pred_png, bbox_inches="tight")
    fig.savefig(out_true_pred_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG: {out_true_pred_png}")
    print(f"Saved PDF: {out_true_pred_pdf}")
    import os, time, torch, numpy as np, pandas as pd
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = str(root / "data")
    TARGET_COLUMNS = ["AW_ST_CMC", "pCMC"]
    best_df = pd.read_csv(os.path.join(base, "grid_results_varwidth.csv"))
    best_cfg = best_df.loc[best_df["val_rmse_mean"].idxmin()].to_dict()
    print("Best configuration loaded:\n", best_cfg)
    from latent_materials.surfactants.models import GINHetero as GINHetero

    full_dataset = tr_ds + va_ds + te_ds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset))), 1):
        print(f"\n Fold {fold}/5 ")
        tr_subset = [full_dataset[i] for i in train_idx]
        va_subset = [full_dataset[i] for i in val_idx]
        tr_ld = DataLoader(
            tr_subset, batch_size=int(best_cfg["batch_size"]), shuffle=True
        )
        va_ld = DataLoader(
            va_subset, batch_size=int(best_cfg["batch_size"]), shuffle=False
        )
        sample = next(iter(tr_ld))
        atom_dim, aux_dim = (sample.x.size(1), sample.aux.size(1))
        model = GINHetero(
            atom_dim, aux_dim, eval(best_cfg["layer_sizes"]), float(best_cfg["dropout"])
        ).to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(best_cfg["lr"]),
            weight_decay=float(best_cfg["weight_decay"]),
        )
        patience = int(best_cfg["patience"])
        best_val, wait = (float("inf"), 0)
        best_state = None
        for epoch in range(1, int(best_cfg["epochs"]) + 1):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                total += loss.item()
                steps += 1
            tr_loss = total / max(1, steps)
            va_rmse, _, _ = eval_loader(model, va_ld, device)
            val_mean = float(np.mean(va_rmse))
            if val_mean < best_val:
                best_val = val_mean
                wait = 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        va_rmse, va_mae, va_r2 = eval_loader(model, va_ld, device)
        print(
            f"Fold {fold} RMSE {np.round(va_rmse, 3)} | MAE {np.round(va_mae, 3)} | R² {np.round(va_r2, 3)}"
        )
        fold_results.append(
            {
                "fold": fold,
                "val_rmse_aw": va_rmse[0],
                "val_rmse_pcmc": va_rmse[1],
                "val_mae_aw": va_mae[0],
                "val_mae_pcmc": va_mae[1],
                "val_r2_aw": va_r2[0],
                "val_r2_pcmc": va_r2[1],
            }
        )
    fold_df = pd.DataFrame(fold_results)
    summary = fold_df.describe().loc[["mean", "std"]]
    cv_path = os.path.join(base, "gnn_bestmodel_5fold_summary.csv")
    fold_df.to_csv(cv_path, index=False)
    print("\n5-Fold CV Results")
    print(fold_df)
    print("\nSummary (mean ± std):")
    print(summary)
    print(f"\nSaved cross-validation results: {cv_path}")
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        fold_df["fold"],
        fold_df["val_r2_aw"],
        yerr=fold_df["val_r2_aw"].std(),
        fmt="-o",
        capsize=4,
        label="AW_ST_CMC",
    )
    plt.errorbar(
        fold_df["fold"],
        fold_df["val_r2_pcmc"],
        yerr=fold_df["val_r2_pcmc"].std(),
        fmt="-o",
        capsize=4,
        label="pCMC",
    )
    plt.xlabel("Fold")
    plt.ylabel("Validation R²")
    plt.title("5-Fold Cross-Validation R² (Best GNN Model)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base, "gnn_bestmodel_5fold_r2.png"), dpi=150)
    plt.show()
    print(f"Saved plot: {os.path.join(base, 'gnn_bestmodel_5fold_r2.png')}")
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    base = str(root / "data")
    grid_csv = os.path.join(base, "grid_results_varwidth.csv")
    cv_csv = os.path.join(base, "gnn_bestmodel_5fold_summary.csv")
    out_png = os.path.join(base, "gnn_training_cv_combined_2x2_refined.png")
    out_pdf = os.path.join(base, "gnn_training_cv_combined_2x2_refined.pdf")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.linewidth": 1.25,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "legend.fontsize": 8.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    def autocrop_image(img, white_threshold=0.985, pad=6):
        if img.ndim == 2:
            mask = img < white_threshold
        else:
            rgb = img[..., :3]
            mask = np.any(rgb < white_threshold, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        y0 = max(y0 - pad, 0)
        x0 = max(x0 - pad, 0)
        y1 = min(y1 + pad, img.shape[0])
        x1 = min(x1 + pad, img.shape[1])
        return img[y0:y1, x0:x1]

    def inset_axis(fig, parent_ax, shrink=0.88):
        pos = parent_ax.get_position()
        new_w = pos.width * shrink
        new_h = pos.height * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        new_y = pos.y0 + (pos.height - new_h) / 2
        ax = fig.add_axes([new_x, new_y, new_w, new_h])
        parent_ax.set_visible(False)
        return ax

    def plot_metric_histogram(
        ax, values_1, values_2, label_1, label_2, xlabel, color_1, color_2, bins=None
    ):
        values_1 = np.asarray(values_1, dtype=float)
        values_2 = np.asarray(values_2, dtype=float)
        all_values = np.concatenate([values_1, values_2])
        if bins is None:
            vmin, vmax = (np.min(all_values), np.max(all_values))
            span = vmax - vmin
            pad = 0.12 * (span + 1e-09)
            bins = np.linspace(vmin - pad, vmax + pad, 7)
        ax.hist(
            values_1,
            bins=bins,
            alpha=0.58,
            edgecolor="black",
            linewidth=0.8,
            color=color_1,
            label=label_1,
        )
        ax.hist(
            values_2,
            bins=bins,
            alpha=0.58,
            edgecolor="black",
            linewidth=0.8,
            color=color_2,
            label=label_2,
        )
        mean_1 = np.mean(values_1)
        mean_2 = np.mean(values_2)
        std_1 = np.std(values_1, ddof=1)
        std_2 = np.std(values_2, ddof=1)
        ax.axvline(mean_1, color=color_1, linestyle="--", linewidth=1.8)
        ax.axvline(mean_2, color=color_2, linestyle="--", linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fold count")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right")
        stats_text = f"{label_1}: {mean_1:.3f} ± {std_1:.3f}\n{label_2}: {mean_2:.3f} ± {std_2:.3f}"
        ax.text(
            0.03,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.95,
            ),
        )

    best_df = pd.read_csv(grid_csv)
    best_row = best_df.loc[best_df["val_rmse_mean"].idxmin()]
    curve_img_path = best_row["curve_path"]
    fallback_curve_img = os.path.join(base, "best_model_curve_summary.png")
    if isinstance(curve_img_path, str) and os.path.exists(curve_img_path):
        curve_path = curve_img_path
    elif os.path.exists(fallback_curve_img):
        curve_path = fallback_curve_img
    else:
        raise FileNotFoundError("Could not find the saved best-model curve image.")
    img = mpimg.imread(curve_path)
    crop_top_fraction = 0.11
    h = img.shape[0]
    img = img[int(crop_top_fraction * h) :, :, :]
    img_cropped = autocrop_image(img, white_threshold=0.985, pad=6)
    fold_df = pd.read_csv(cv_csv)
    required_cols = [
        "fold",
        "val_rmse_aw",
        "val_rmse_pcmc",
        "val_mae_aw",
        "val_mae_pcmc",
        "val_r2_aw",
        "val_r2_pcmc",
    ]
    for col in required_cols:
        if col not in fold_df.columns:
            raise ValueError(f"Missing required column in {cv_csv}: {col}")
    gamma_rmse = fold_df["val_rmse_aw"].to_numpy()
    gamma_mae = fold_df["val_mae_aw"].to_numpy()
    gamma_r2 = fold_df["val_r2_aw"].to_numpy()
    pcmc_rmse = fold_df["val_rmse_pcmc"].to_numpy()
    pcmc_mae = fold_df["val_mae_pcmc"].to_numpy()
    pcmc_r2 = fold_df["val_r2_pcmc"].to_numpy()
    color_gamma_rmse = "#0072B2"
    color_gamma_mae = "#56B4E9"
    color_pcmc_rmse = "#D55E00"
    color_pcmc_mae = "#E69F00"
    color_r2_gamma = "#0072B2"
    color_r2_pcmc = "#D55E00"
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    axA = axes[0, 0]
    cellB = axes[0, 1]
    cellC = axes[1, 0]
    cellD = axes[1, 1]
    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.92, bottom=0.09, wspace=0.18, hspace=0.22
    )
    axB = inset_axis(fig, cellB, shrink=0.84)
    axC = inset_axis(fig, cellC, shrink=0.84)
    axD = inset_axis(fig, cellD, shrink=0.84)
    axA.imshow(img_cropped)
    axA.axis("off")
    axA.set_title(
        "Best-model training and validation history", pad=10, fontweight="semibold"
    )
    axA.text(
        -0.04,
        1.03,
        "A",
        transform=axA.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    plot_metric_histogram(
        ax=axB,
        values_1=gamma_rmse,
        values_2=gamma_mae,
        label_1="$\\gamma_{\\mathrm{CMC}}$ RMSE",
        label_2="$\\gamma_{\\mathrm{CMC}}$ MAE",
        xlabel="$\\gamma_{\\mathrm{CMC}}$ error (mN/m)",
        color_1=color_gamma_rmse,
        color_2=color_gamma_mae,
    )
    axB.set_title(
        "$\\gamma_{\\mathrm{CMC}}$ cross-validation error",
        pad=10,
        fontweight="semibold",
    )
    axB.text(
        -0.16,
        1.04,
        "B",
        transform=axB.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    plot_metric_histogram(
        ax=axC,
        values_1=pcmc_rmse,
        values_2=pcmc_mae,
        label_1="pCMC RMSE",
        label_2="pCMC MAE",
        xlabel="pCMC error",
        color_1=color_pcmc_rmse,
        color_2=color_pcmc_mae,
    )
    axC.set_title("pCMC cross-validation error", pad=10, fontweight="semibold")
    axC.text(
        -0.16,
        1.04,
        "C",
        transform=axC.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    plot_metric_histogram(
        ax=axD,
        values_1=gamma_r2,
        values_2=pcmc_r2,
        label_1="$\\gamma_{\\mathrm{CMC}}$ $R^2$",
        label_2="pCMC $R^2$",
        xlabel="$R^2$",
        color_1=color_r2_gamma,
        color_2=color_r2_pcmc,
    )
    axD.set_title("Cross-validation $R^2$ distribution", pad=10, fontweight="semibold")
    axD.text(
        -0.16,
        1.04,
        "D",
        transform=axD.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.suptitle(
        "Training behavior and five-fold cross-validation performance of the heteroscedastic GNN surrogate",
        fontsize=16.5,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        "$\\gamma_{\\mathrm{CMC}}$ denotes air–water surface tension at CMC (mN/m). pCMC is the logarithmic CMC descriptor. RMSE and MAE retain the units of the corresponding target, whereas $R^2$ is unitless.",
        ha="center",
        va="bottom",
        fontsize=9.4,
    )
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    return root
