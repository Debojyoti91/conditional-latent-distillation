"""New experiment: validation-only final selection and post-selection model export.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import display, require_dependencies


def run(workspace):
    """Execute the aem surrogate validation selected experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    import os
    import json
    import random
    import warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    import matplotlib.pyplot as plt

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")
    BASE_DIR = str(root / "IEMS_GEN_files/AEM")
    PROJECT_DIR = os.path.join(BASE_DIR, "CVAE")
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    CSV_PATH = os.path.join(DATA_DIR, "modified_data_final.csv")
    ARTIFACT_PATH = os.path.join(DATA_DIR, "aem_graph_ae_regressor_ensemble.pt")
    AUX_SCALER_PATH = os.path.join(DATA_DIR, "aem_aux_scaler.pkl")
    TARGET_TRANSFORMER_PATH = os.path.join(DATA_DIR, "aem_target_transformer.pkl")
    METRICS_PATH = os.path.join(DATA_DIR, "aem_final_metrics.json")
    PREDICTION_PATH = os.path.join(
        DATA_DIR, "aem_augmented_predicted_conductivity_NN.csv"
    )
    SEED = 2023
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Data directory:", DATA_DIR)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Could not find {CSV_PATH}. Put modified_data_final.csv in the data folder."
        )
    HAS_HEADER = False
    USE_POLYMER_TYPE_ONE_HOT = False
    COLUMN_MAP = None
    if HAS_HEADER:
        raw = pd.read_csv(CSV_PATH)
    else:
        raw = pd.read_csv(CSV_PATH, header=None)
    print("Raw file shape:", raw.shape)
    print("First rows of raw file:")
    display(raw.head())

    def load_aem_dataframe(raw, has_header=False, column_map=None):
        """Return clean AEM dataframe with standardized internal names."""
        if has_header and column_map is not None:
            df0 = pd.DataFrame(
                {
                    "SMILES": raw[column_map["smiles"]],
                    "BlockA": raw[column_map["block_a"]],
                    "BlockB": raw[column_map["block_b"]],
                    "BlockA_over_B": raw[column_map["block_ratio"]],
                    "PolymerTypeFeature": raw[column_map["polymer_type"]],
                    "AnionConductivity": raw[column_map["target"]],
                }
            )
        else:
            if raw.shape[1] < 6:
                raise ValueError(
                    "Expected at least 6 columns: SMILES | BlockA | BlockB | BlockA_over_B | polymer-type feature | Conductivity"
                )
            df0 = pd.DataFrame(
                {
                    "SMILES": raw.iloc[:, 0].astype(str),
                    "BlockA": raw.iloc[:, 1],
                    "BlockB": raw.iloc[:, 2],
                    "BlockA_over_B": raw.iloc[:, 3],
                    "PolymerTypeFeature": raw.iloc[:, 4],
                    "AnionConductivity": raw.iloc[:, -1],
                }
            )
        for col in ["BlockA", "BlockB", "BlockA_over_B", "AnionConductivity"]:
            df0[col] = pd.to_numeric(df0[col], errors="coerce")
        poly_numeric = pd.to_numeric(df0["PolymerTypeFeature"], errors="coerce")
        if poly_numeric.notna().mean() >= 0.95:
            df0["PolymerTypeFeatureEncoded"] = poly_numeric
            polymer_type_mapping = None
            polymer_type_is_categorical = False
            print("PolymerTypeFeature interpreted as numeric.")
        else:
            df0["PolymerTypeFeature"] = df0["PolymerTypeFeature"].astype(str)
            codes, uniques = pd.factorize(df0["PolymerTypeFeature"])
            df0["PolymerTypeFeatureEncoded"] = codes.astype(float)
            polymer_type_mapping = {str(name): int(i) for i, name in enumerate(uniques)}
            polymer_type_is_categorical = True
            print("PolymerTypeFeature interpreted as categorical and encoded.")
            print("Polymer type mapping:")
            print(polymer_type_mapping)

        def canonical_smiles_local(smi):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
            return Chem.MolToSmiles(mol, canonical=True)

        df0["Canonical_SMILES"] = df0["SMILES"].apply(canonical_smiles_local)
        before = len(df0)
        df0 = df0.dropna(
            subset=[
                "Canonical_SMILES",
                "BlockA",
                "BlockB",
                "BlockA_over_B",
                "PolymerTypeFeatureEncoded",
                "AnionConductivity",
            ]
        ).reset_index(drop=True)
        df0["SMILES"] = df0["Canonical_SMILES"]
        df0 = df0.drop(columns=["Canonical_SMILES"])
        df0 = df0.drop_duplicates().reset_index(drop=True)
        print(f"Cleaned rows: {len(df0)}/{before}")
        return (df0, polymer_type_mapping, polymer_type_is_categorical)

    df, polymer_type_mapping, polymer_type_is_categorical = load_aem_dataframe(
        raw, has_header=HAS_HEADER, column_map=COLUMN_MAP
    )
    TARGET_COL = "AnionConductivity"
    df["Block_total"] = df["BlockA"] + df["BlockB"]
    df["BlockA_fraction"] = df["BlockA"] / (df["BlockA"] + df["BlockB"] + 1e-09)
    df["BlockB_fraction"] = df["BlockB"] / (df["BlockA"] + df["BlockB"] + 1e-09)
    df["Block_abs_diff"] = (df["BlockA"] - df["BlockB"]).abs()
    df["Block_product"] = df["BlockA"] * df["BlockB"]
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

    def rdkit_global_descriptors(smi):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return pd.Series(
                {
                    "MolWt": np.nan,
                    "TPSA": np.nan,
                    "MolLogP": np.nan,
                    "NumHAcceptors": np.nan,
                    "NumHDonors": np.nan,
                    "NumRotatableBonds": np.nan,
                    "RingCount": np.nan,
                    "AromaticRingCount": np.nan,
                    "HeavyAtomCount": np.nan,
                    "HeteroAtomCount": np.nan,
                    "FractionCSP3": np.nan,
                }
            )
        return pd.Series(
            {
                "MolWt": Descriptors.MolWt(mol),
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "MolLogP": Crippen.MolLogP(mol),
                "NumHAcceptors": Lipinski.NumHAcceptors(mol),
                "NumHDonors": Lipinski.NumHDonors(mol),
                "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
                "RingCount": rdMolDescriptors.CalcNumRings(mol),
                "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
                "HeavyAtomCount": mol.GetNumHeavyAtoms(),
                "HeteroAtomCount": Lipinski.NumHeteroatoms(mol),
                "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            }
        )

    descriptor_df = df["SMILES"].apply(rdkit_global_descriptors)
    df = pd.concat([df, descriptor_df], axis=1)
    RDKit_AUX_COLS = list(descriptor_df.columns)
    for col in RDKit_AUX_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        if not np.isfinite(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
    print("Added RDKit descriptor columns:")
    print(RDKit_AUX_COLS)
    BASE_AUX_COLS = [
        "BlockA",
        "BlockB",
        "BlockA_over_B",
        "Block_total",
        "BlockA_fraction",
        "BlockB_fraction",
        "Block_abs_diff",
        "Block_product",
    ]
    if polymer_type_is_categorical and USE_POLYMER_TYPE_ONE_HOT:
        polymer_ohe = pd.get_dummies(
            df["PolymerTypeFeature"].astype(str), prefix="PolymerType", dtype=float
        )
        df = pd.concat([df, polymer_ohe], axis=1)
        POLYMER_TYPE_COLS = list(polymer_ohe.columns)
        print("Using one-hot polymer type columns:", POLYMER_TYPE_COLS)
    else:
        POLYMER_TYPE_COLS = ["PolymerTypeFeatureEncoded"]
        print("Using polymer type column:", POLYMER_TYPE_COLS)
    AUX_COLS = BASE_AUX_COLS + POLYMER_TYPE_COLS
    NUM_EXTRA_FEATS = len(AUX_COLS)
    clean_path = os.path.join(DATA_DIR, "aem_clean.csv")
    df.to_csv(clean_path, index=False)
    if polymer_type_mapping is not None:
        mapping_path = os.path.join(DATA_DIR, "aem_polymer_type_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(polymer_type_mapping, f, indent=2)
        print("Saved polymer type mapping:", mapping_path)
    print("Final AEM dataframe:", df.shape)
    print("AUX_COLS:", AUX_COLS)
    print("NUM_EXTRA_FEATS:", NUM_EXTRA_FEATS)
    display(df.head())
    print("\nNumeric summary:")
    display(df[AUX_COLS + [TARGET_COL]].describe().T)
    print("\nSaved clean file:", clean_path)
    TEST_SIZE = 0.1
    VAL_SIZE_FROM_TRAIN = 0.15
    N_CONDUCTIVITY_BINS = 5

    def make_conductivity_bins(y, q=5):
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts(dropna=False)
        if bins.nunique() < 2 or counts.min() < 2:
            return None
        return bins

    strat_bins_all = make_conductivity_bins(df[TARGET_COL], q=N_CONDUCTIVITY_BINS)
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=strat_bins_all,
    )
    strat_bins_train_val = make_conductivity_bins(
        train_val_df[TARGET_COL], q=N_CONDUCTIVITY_BINS
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=SEED,
        shuffle=True,
        stratify=strat_bins_train_val,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print("Train:", train_df.shape)
    print("Val:  ", val_df.shape)
    print("Test: ", test_df.shape)
    print("\nTarget ranges:")
    print("Train target range:", train_df[TARGET_COL].min(), train_df[TARGET_COL].max())
    print("Val target range:  ", val_df[TARGET_COL].min(), val_df[TARGET_COL].max())
    print("Test target range: ", test_df[TARGET_COL].min(), test_df[TARGET_COL].max())
    aux_scaler = StandardScaler()
    aux_scaler.fit(train_df[AUX_COLS].values.astype(np.float32))
    target_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    target_transformer.fit(train_df[[TARGET_COL]].values.astype(np.float32))
    joblib.dump(aux_scaler, AUX_SCALER_PATH)
    joblib.dump(target_transformer, TARGET_TRANSFORMER_PATH)
    print("\nSaved aux scaler:", AUX_SCALER_PATH)
    print("Saved target transformer:", TARGET_TRANSFORMER_PATH)
    from latent_materials.aems.config import default_config

    BASE_CFG = default_config()
    from latent_materials.aems.data import (
        build_graph_list,
        get_atom_feature_dims,
        smiles_to_pyg,
    )

    train_graphs = build_graph_list(
        train_df,
        BASE_CFG,
        aux_scaler,
        target_transformer,
        transform_y=True,
        aux_columns=AUX_COLS,
        target_column=TARGET_COL,
    )
    val_graphs = build_graph_list(
        val_df,
        BASE_CFG,
        aux_scaler,
        target_transformer,
        transform_y=True,
        aux_columns=AUX_COLS,
        target_column=TARGET_COL,
    )
    test_graphs = build_graph_list(
        test_df,
        BASE_CFG,
        aux_scaler,
        target_transformer,
        transform_y=True,
        aux_columns=AUX_COLS,
        target_column=TARGET_COL,
    )
    sample = train_graphs[0]
    print("Node feature dim:", sample.x.shape[1])
    print("Edge feature dim:", sample.edge_attr.shape[1])
    print("AUX feature dim:", sample.num_feats.shape[0])
    print("Target shape:", sample.y.shape)
    print(
        "Sample weight examples:",
        [float(g.sample_weight.item()) for g in train_graphs[:10]],
    )
    print("High-conductivity weight threshold:", BASE_CFG["high_cond_threshold"])
    print("High-conductivity weight:", BASE_CFG["high_cond_weight"])
    from latent_materials.aems.models import (
        JointGraphAER,
    )
    from functools import partial
    from latent_materials.aems import training
    from latent_materials.aems.training import (
        make_loader,
        print_metrics,
        weighted_supervised_loss,
        augment_train_batch,
    )

    def plot_training_history(hist, title="AEM training curve"):
        if len(hist["epoch"]) == 0:
            print("No history to plot.")
            return
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(hist["epoch"], hist["train_loss"], label="Train total loss")
        if "train_sup_loss" in hist:
            ax1.plot(
                hist["epoch"],
                hist["train_sup_loss"],
                label="Train supervised loss",
                alpha=0.75,
            )
        ax1.plot(hist["epoch"], hist["val_rmse"], label="Val RMSE")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss / RMSE")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot(hist["epoch"], hist["val_r2"], label="Val R²", alpha=0.6)
        ax2.set_ylabel("R²")
        ax2.legend(loc="upper right")
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    SEED = 2023
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    inverse_target = partial(
        training.inverse_target, target_transformer=target_transformer
    )
    eval_model = partial(
        training.eval_model, device=device, target_transformer=target_transformer
    )
    eval_ensemble = partial(
        training.eval_ensemble, device=device, target_transformer=target_transformer
    )
    pretrain_encoder = partial(training.pretrain_encoder, device=device)
    train_single_joint = partial(
        training.train_single_joint,
        device=device,
        target_transformer=target_transformer,
        aux_columns=AUX_COLS,
        base_seed=SEED,
    )
    print(
        "v9 stability overrides loaded: SEED=2023, NaN-safe evaluation, non-finite batch skipping."
    )
    import copy

    RUN_HPO = True
    HPO_N_TRIALS = 20
    HPO_AE_EPOCHS = 80
    HPO_JOINT_EPOCHS = 220
    HPO_LOG_PATH = os.path.join(DATA_DIR, "aem_hgare_hpo_log.json")

    def _scalar_metrics(m):
        return {
            "mae": float(m["mae"]),
            "mse": float(m["mse"]),
            "rmse": float(m["rmse"]),
            "r2": float(m["r2"]),
        }

    def sample_cfg(base):
        """HGARE.py's search space, adapted to this notebook's cfg keys."""
        c = dict(base)
        c["enc_hidden"] = random.choice([128, 160, 192, 224, 256])
        c["enc_layers"] = random.choice([3, 4, 5])
        c["enc_dropout"] = random.choice([0.05, 0.08, 0.1, 0.12, 0.15])
        c["growth"] = random.choice([192, 224, 256, 320])
        c["n_blocks"] = random.choice([3, 4, 5])
        c["dropout"] = random.choice([0.1, 0.15, 0.2, 0.25])
        c["head_hidden"] = random.choice([256, 320, 384, 512])
        c["lr"] = random.choice([0.0007, 0.0009, 0.001, 0.0012])
        c["weight_decay"] = random.choice([0.0001, 0.0005, 0.0009, 0.0012])
        c["batch_size"] = random.choice([24, 32, 48])
        c["T0"] = random.choice([70, 100, 120, 150])
        c["Tmult"] = random.choice([1, 2])
        c["swa_ratio"] = random.choice([0.25, 0.3, 0.33, 0.4])
        c["alpha"] = random.choice([0.6, 0.65, 0.7, 0.8])
        sup_w = random.choice([0.9, 0.92, 0.95])
        c["sup_weight"] = sup_w
        c["recon_weight"] = 1.0 - sup_w
        return c

    def quick_eval(trial_cfg, seed_offset):
        ae_init_t = pretrain_encoder(
            train_graphs, trial_cfg, bin_dim, cont_dim, seed=SEED + seed_offset
        )
        _, _, val_metrics_t = train_single_joint(
            train_graphs,
            val_graphs,
            ae_init_t,
            trial_cfg,
            bin_dim,
            cont_dim,
            seed_offset=seed_offset,
        )
        return val_metrics_t

    def hpo_random_search(base_cfg, n_trials, hpo_ae_epochs, hpo_joint_epochs):
        logs = []
        best_cfg_local, best_met_local = (None, None)
        for t in range(n_trials):
            trial_cfg = sample_cfg(base_cfg)
            trial_cfg["ae_epochs"] = hpo_ae_epochs
            trial_cfg["epochs"] = hpo_joint_epochs
            metrics_t = quick_eval(trial_cfg, seed_offset=t * 97)
            logs.append(
                {
                    "trial": t,
                    "cfg": make_json_safe(trial_cfg),
                    "metrics": _scalar_metrics(metrics_t),
                }
            )
            print(
                f"[HPO {t:02d}] R2={metrics_t['r2']:.4f}  MAE={metrics_t['mae']:.4f}  RMSE={metrics_t['rmse']:.4f}"
            )
            if (
                best_met_local is None
                or metrics_t["r2"] > best_met_local["r2"] + 1e-06
                or (
                    abs(metrics_t["r2"] - best_met_local["r2"]) < 1e-06
                    and metrics_t["mae"] < best_met_local["mae"]
                )
            ):
                best_cfg_local, best_met_local = (trial_cfg, metrics_t)
        with open(HPO_LOG_PATH, "w") as f:
            json.dump(
                {
                    "logs": logs,
                    "best_cfg": make_json_safe(best_cfg_local),
                    "best_metrics": _scalar_metrics(best_met_local),
                },
                f,
                indent=2,
            )
        print(
            f"\n[HPO] Best R2={best_met_local['r2']:.4f}, MAE={best_met_local['mae']:.4f}, RMSE={best_met_local['rmse']:.4f}"
        )
        return (best_cfg_local, best_met_local)

    bin_dim, cont_dim = get_atom_feature_dims(BASE_CFG)
    if RUN_HPO:
        print(
            f"Running {HPO_N_TRIALS}-trial R2-first random search (ae_epochs={HPO_AE_EPOCHS}, joint_epochs={HPO_JOINT_EPOCHS} per trial)..."
        )
        best_cfg, best_hpo_metrics = hpo_random_search(
            BASE_CFG, HPO_N_TRIALS, HPO_AE_EPOCHS, HPO_JOINT_EPOCHS
        )
    else:
        print("RUN_HPO=False: skipping search, using BASE_CFG unchanged.")
        best_cfg, best_hpo_metrics = ({}, None)
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(best_cfg)
    cfg["sup_weight"] = 1.0 - cfg["recon_weight"]
    cfg["ae_epochs"] = max(120, BASE_CFG["ae_epochs"])
    cfg["epochs"] = BASE_CFG["epochs"]
    cfg["ensemble_size"] = BASE_CFG["ensemble_size"]
    print(
        "Final selected cfg (HGARE-style HPO winner, extended to full training length):"
    )
    for k, v in cfg.items():
        if k not in ["atom_nums", "hybs", "bond_types"]:
            print(f"  {k}: {v}")
    train_graphs = build_graph_list(
        train_df,
        cfg,
        aux_scaler,
        target_transformer,
        transform_y=True,
        aux_columns=AUX_COLS,
        target_column=TARGET_COL,
    )
    val_graphs = build_graph_list(
        val_df,
        cfg,
        aux_scaler,
        target_transformer,
        transform_y=True,
        aux_columns=AUX_COLS,
        target_column=TARGET_COL,
    )
    test_graphs = build_graph_list(
        test_df,
        cfg,
        aux_scaler,
        target_transformer,
        transform_y=True,
        aux_columns=AUX_COLS,
        target_column=TARGET_COL,
    )
    print("\nAUX_COLS:", AUX_COLS)
    print("NUM_EXTRA_FEATS:", len(AUX_COLS))
    bin_dim, cont_dim = get_atom_feature_dims(cfg)
    print("AEM configuration:")
    for k, v in cfg.items():
        if k not in ["atom_nums", "hybs", "bond_types"]:
            print(f"  {k}: {v}")
    print("\nPretraining graph autoencoder...")
    ae_init = pretrain_encoder(train_graphs, cfg, bin_dim, cont_dim, seed=SEED)
    print("\nTraining ensemble...")
    models = []
    histories = []
    member_val_metrics = []
    for i in range(cfg["ensemble_size"]):
        print(
            f"\n================ Ensemble member {i + 1}/{cfg['ensemble_size']} ================"
        )
        model_i, hist_i, val_metrics_i = train_single_joint(
            train_graphs,
            val_graphs,
            ae_init,
            cfg,
            bin_dim,
            cont_dim,
            seed_offset=i * 123,
        )
        models.append(model_i)
        histories.append(hist_i)
        member_val_metrics.append(val_metrics_i)
    print("\nFinished ensemble training.")
    train_metrics = eval_ensemble(models, train_graphs, cfg, label="train")
    val_metrics = eval_ensemble(models, val_graphs, cfg, label="validation")
    test_metrics = eval_ensemble(models, test_graphs, cfg, label="test")
    print_metrics(train_metrics)
    print_metrics(val_metrics)
    print_metrics(test_metrics)

    def make_json_safe(obj):
        """
        Convert objects that json.dump cannot save directly.
        Handles NumPy scalars, arrays, lists, tuples, dictionaries, and PyTorch tensors.
        """
        import numpy as np
        import torch
        import json

        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [make_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    def compact_metrics(metrics):
        """
        Keep only scalar metrics and small metadata.
        Exclude arrays such as y_true, y_pred, and pred_std_transformed_space.
        """
        exclude_keys = ["y_true", "y_pred", "pred_std_transformed_space"]
        return {
            k: make_json_safe(v) for k, v in metrics.items() if k not in exclude_keys
        }

    metrics_to_save = {
        "train": compact_metrics(train_metrics),
        "validation": compact_metrics(val_metrics),
        "test": compact_metrics(test_metrics),
        "config": make_json_safe(cfg),
        "aux_cols": make_json_safe(AUX_COLS),
        "target_col": TARGET_COL,
        "num_extra_feats": len(AUX_COLS),
        "note": "v8 saves scalar metrics to JSON and saves y_true/y_pred arrays to a separate CSV to avoid ndarray JSON serialization errors.",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print("Saved metrics:", METRICS_PATH)

    def save_prediction_table(metrics, split_name):
        pred_df_split = pd.DataFrame(
            {
                "split": split_name,
                "y_true": metrics["y_true"],
                "y_pred": metrics["y_pred"],
                "abs_error": np.abs(metrics["y_true"] - metrics["y_pred"]),
                "squared_error": (metrics["y_true"] - metrics["y_pred"]) ** 2,
            }
        )
        if "pred_std_transformed_space" in metrics:
            pred_df_split["pred_std_transformed_space"] = metrics[
                "pred_std_transformed_space"
            ]
        return pred_df_split

    train_pred_df = save_prediction_table(train_metrics, "train")
    val_pred_df = save_prediction_table(val_metrics, "validation")
    test_pred_df = save_prediction_table(test_metrics, "test")
    all_pred_df = pd.concat(
        [train_pred_df, val_pred_df, test_pred_df], axis=0, ignore_index=True
    )
    PRED_TABLE_PATH = os.path.join(DATA_DIR, "aem_train_val_test_predictions.csv")
    all_pred_df.to_csv(PRED_TABLE_PATH, index=False)
    print("Saved prediction table:", PRED_TABLE_PATH)
    display(all_pred_df.head())
    test_error_df = pd.DataFrame(
        {"y_true": test_metrics["y_true"], "y_pred": test_metrics["y_pred"]}
    )
    test_error_df["abs_error"] = np.abs(
        test_error_df["y_true"] - test_error_df["y_pred"]
    )
    test_error_df["squared_error"] = (
        test_error_df["y_true"] - test_error_df["y_pred"]
    ) ** 2
    test_error_df["conductivity_bin"] = pd.qcut(
        test_error_df["y_true"],
        q=4,
        labels=["low", "mid-low", "mid-high", "high"],
        duplicates="drop",
    )
    bin_summary = test_error_df.groupby("conductivity_bin").agg(
        n=("y_true", "count"),
        y_true_min=("y_true", "min"),
        y_true_max=("y_true", "max"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
    )
    error_csv_path = os.path.join(DATA_DIR, "aem_test_error_by_conductivity_bin.csv")
    bin_summary.to_csv(error_csv_path)
    display(bin_summary)
    print("Saved bin error summary:", error_csv_path)
    plt.figure(figsize=(7, 5.5))
    for b in test_error_df["conductivity_bin"].dropna().unique():
        sub = test_error_df[test_error_df["conductivity_bin"] == b]
        plt.scatter(sub["y_true"], sub["y_pred"], label=str(b), s=55, alpha=0.82)
    mn = min(test_error_df["y_true"].min(), test_error_df["y_pred"].min())
    mx = max(test_error_df["y_true"].max(), test_error_df["y_pred"].max())
    pad = 0.05 * (mx - mn + 1e-09)
    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "--", color="black")
    plt.xlabel("Experimental anion conductivity")
    plt.ylabel("Predicted anion conductivity")
    plt.title("AEM test prediction by conductivity range")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    error_fig_path = os.path.join(
        DATA_DIR, "aem_test_prediction_by_conductivity_range.png"
    )
    plt.savefig(error_fig_path, dpi=400, bbox_inches="tight")
    plt.show()
    print("Saved bin error plot:", error_fig_path)
    if "high" in bin_summary.index:
        print("\nHigh-conductivity bin RMSE:", float(bin_summary.loc["high", "rmse"]))
        print("Compare this with low/mid bins to see whether weighted loss helped.")
    artifact = {
        "config": cfg,
        "ae_init": ae_init,
        "joint_states": [m.state_dict() for m in models],
        "aux_cols": AUX_COLS,
        "target_col": TARGET_COL,
        "num_extra_feats": len(AUX_COLS),
        "v8_notes": "Optuna optimizer + derived AEM features + stratified split + mean/max pooling + weighted high-conductivity loss + JSON fix",
        "seed": SEED,
        "node_in": ae_init["node_in"],
        "edge_in": ae_init["edge_in"],
    }
    torch.save(artifact, ARTIFACT_PATH)
    joblib.dump(aux_scaler, AUX_SCALER_PATH)
    joblib.dump(target_transformer, TARGET_TRANSFORMER_PATH)
    print("Saved model artifact:", ARTIFACT_PATH)
    print("Saved aux scaler:", AUX_SCALER_PATH)
    print("Saved target transformer:", TARGET_TRANSFORMER_PATH)
    import selfies as sf
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

    GEN_PROJECT_DIR = str(root / "Generative_AI_surfactants_molecules")
    VAE_CKPT_PATH = os.path.join(GEN_PROJECT_DIR, "models", "pretrained_vae.pt")
    POLYMERS_CSV_PATH = os.path.join(GEN_PROJECT_DIR, "data", "polymers_dataset.csv")
    VAE_RETRAIN_CKPT_PATH = os.path.join(DATA_DIR, "aem_combined_vae.pt")
    for _p in (VAE_CKPT_PATH, POLYMERS_CSV_PATH):
        if not os.path.exists(_p):
            raise FileNotFoundError(f"Required generative-model file not found: {_p}")
    from latent_materials.teacher import VAE

    _pretrain_smiles_df = pd.read_csv(POLYMERS_CSV_PATH, header=None, skiprows=2)
    _pretrain_smiles_list = _pretrain_smiles_df.iloc[:, 1].tolist()
    _aem_smiles_list = df["SMILES"].astype(str).tolist()
    _combined_smiles_list = _pretrain_smiles_list + _aem_smiles_list
    _combined_source_list = ["polymer"] * len(_pretrain_smiles_list) + ["aem"] * len(
        _aem_smiles_list
    )
    _combined_selfies, _combined_source = ([], [])
    for _smi, _src in zip(_combined_smiles_list, _combined_source_list):
        try:
            _s = sf.encoder(str(_smi))
        except Exception:
            _s = None
        if _s:
            _combined_selfies.append(_s)
            _combined_source.append(_src)
    _lens = np.array([sf.len_selfies(s) for s in _combined_selfies])
    _source_arr = np.array(_combined_source)
    VAE_MAX_LEN = 400
    _keep_mask = _lens <= VAE_MAX_LEN
    n_aem_total = int((_source_arr == "aem").sum())
    n_aem_excluded = int(((_source_arr == "aem") & ~_keep_mask).sum())
    n_poly_excluded = int(((_source_arr == "polymer") & ~_keep_mask).sum())
    print(
        f"[VOCAB] VAE_MAX_LEN={VAE_MAX_LEN} | AEM molecules excluded (too long): {n_aem_excluded}/{n_aem_total} | polymer molecules excluded: {n_poly_excluded}"
    )
    _kept_selfies = [s for s, k in zip(_combined_selfies, _keep_mask) if k]
    VAE_ALPHABET = sf.get_alphabet_from_selfies(_kept_selfies)
    VAE_ALPHABET.add("[nop]")
    VAE_ALPHABET = list(sorted(VAE_ALPHABET))
    VAE_VOCAB_STOI = {symbol: idx for idx, symbol in enumerate(VAE_ALPHABET)}
    VAE_VOCAB_ITOS = {idx: symbol for symbol, idx in VAE_VOCAB_STOI.items()}
    VAE_LATENT_DIM = 32
    VAE_INPUT_DIM = VAE_MAX_LEN * len(VAE_ALPHABET)
    print(
        f"[VOCAB] alphabet_size={len(VAE_ALPHABET)} max_len={VAE_MAX_LEN} -> input_dim={VAE_INPUT_DIM}"
    )

    def _encode_fixed_vocab(selfies_str):
        _, one_hot = sf.selfies_to_encoding(
            selfies_str, VAE_VOCAB_STOI, pad_to_len=VAE_MAX_LEN
        )
        return np.asarray(one_hot, dtype=np.float32).reshape(-1)

    _train_vecs = np.vstack([_encode_fixed_vocab(s) for s in _kept_selfies]).astype(
        np.float32
    )
    print(
        f"[VOCAB] Built {_train_vecs.shape[0]} training vectors of dim {_train_vecs.shape[1]}."
    )
    vae = VAE(input_dim=VAE_INPUT_DIM, latent_dim=VAE_LATENT_DIM).to(device)
    _old_state = torch.load(VAE_CKPT_PATH, map_location=device)
    _new_state = vae.state_dict()
    _warm_started = []
    for _key in (
        "encoder.2.weight",
        "encoder.2.bias",
        "decoder.0.weight",
        "decoder.0.bias",
    ):
        if _key in _old_state and _old_state[_key].shape == _new_state[_key].shape:
            _new_state[_key] = _old_state[_key]
            _warm_started.append(_key)
    vae.load_state_dict(_new_state)
    print(f"[OK] Warm-started layers from the original checkpoint: {_warm_started}")

    def _vae_loss_beta(recon_x, x, mu, log_var, beta):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return (recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item())

    vae_train_tensor = torch.tensor(_train_vecs, dtype=torch.float32)
    vae_train_loader = TorchDataLoader(
        TensorDataset(vae_train_tensor), batch_size=64, shuffle=True
    )
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    BETA_TARGET = 0.05
    BETA_WARMUP_EPOCHS = 150
    print(
        f"[VAE] beta_target={BETA_TARGET} (empirically calibrated to latent std~1), warmup over {BETA_WARMUP_EPOCHS} epochs"
    )
    vae.train()
    VAE_TRAIN_EPOCHS = 500
    for _epoch in range(VAE_TRAIN_EPOCHS):
        _beta = BETA_TARGET * min(1.0, _epoch / BETA_WARMUP_EPOCHS)
        _total_loss, _total_recon, _total_kl = (0.0, 0.0, 0.0)
        for (_batch,) in vae_train_loader:
            _batch = _batch.to(device)
            vae_optimizer.zero_grad()
            _recon, _mu, _logvar = vae(_batch)
            _loss, _r, _k = _vae_loss_beta(_recon, _batch, _mu, _logvar, _beta)
            _loss.backward()
            vae_optimizer.step()
            _total_loss += _loss.item()
            _total_recon += _r
            _total_kl += _k
        if (_epoch + 1) % 25 == 0 or _epoch == 0:
            _n = len(vae_train_loader)
            print(
                f"[VAE] epoch {_epoch + 1:04d}/{VAE_TRAIN_EPOCHS} | beta={_beta:.2f} | loss={_total_loss / _n:.2f} | recon={_total_recon / _n:.2f} | kl={_total_kl / _n:.2f}"
            )
    vae.eval()
    torch.save(vae.state_dict(), VAE_RETRAIN_CKPT_PATH)
    print(f"[OK] Trained combined-corpus VAE, saved -> {VAE_RETRAIN_CKPT_PATH}")
    with torch.no_grad():
        _check_batch = vae_train_tensor[: min(1024, vae_train_tensor.size(0))].to(
            device
        )
        _enc_check = vae.encoder(_check_batch)
        _mu_check, _ = _enc_check.chunk(2, dim=-1)
        _mu_mean = _mu_check.mean(dim=0).cpu().numpy()
        _mu_std = _mu_check.std(dim=0).cpu().numpy()
    print(
        f"[VAE CHECK] latent mu: mean(|per-dim mean|)={np.abs(_mu_mean).mean():.3f} (target ~0), mean(per-dim std)={_mu_std.mean():.3f} (target ~1)"
    )

    def encode_smiles_to_vae_input(smi):
        """Return (flat_one_hot_vector, ok). ok=False if any SELFIES token falls
        outside the VAE's training-time alphabet or the encoded length exceeds
        VAE_MAX_LEN (both signal this molecule can't be represented in that latent
        space)."""
        try:
            selfies_str = sf.encoder(str(smi))
        except Exception:
            return (None, False)
        if not selfies_str:
            return (None, False)
        if sf.len_selfies(selfies_str) > VAE_MAX_LEN:
            return (None, False)
        tokens = list(sf.split_selfies(selfies_str))
        if any((tok not in VAE_VOCAB_STOI for tok in tokens)):
            return (None, False)
        try:
            _, one_hot = sf.selfies_to_encoding(
                selfies_str, VAE_VOCAB_STOI, pad_to_len=VAE_MAX_LEN
            )
        except Exception:
            return (None, False)
        return (np.asarray(one_hot, dtype=np.float32).reshape(-1), True)

    aem_smiles_all = df["SMILES"].astype(str).tolist()
    _vecs, _ok_mask = ([], [])
    for smi in aem_smiles_all:
        vec, ok = encode_smiles_to_vae_input(smi)
        _ok_mask.append(ok)
        _vecs.append(vec if ok else np.zeros(VAE_INPUT_DIM, dtype=np.float32))
    vae_encode_ok = np.array(_ok_mask, dtype=bool)
    coverage_pct = 100.0 * vae_encode_ok.mean()
    print(
        f"[COVERAGE] {int(vae_encode_ok.sum())}/{len(aem_smiles_all)} AEM molecules ({coverage_pct:.1f}%) encode within the pretrained VAE's SELFIES vocabulary."
    )
    if coverage_pct < 70.0:
        print(
            "[WARN] Coverage is low -- the surfactant/polymer pretraining alphabet may not cover AEM-specific chemistry well. Consider continuing to train the VAE on polymers_dataset.csv + AEM SMILES before relying on it further."
        )
    aem_input_matrix = np.vstack(_vecs)
    with torch.no_grad():
        _enc_out = vae.encoder(
            torch.tensor(aem_input_matrix, dtype=torch.float32, device=device)
        )
        _mu_all, _logvar_all = _enc_out.chunk(2, dim=-1)
    teacher_latents_df_idx = np.where(vae_encode_ok)[0]
    teacher_latents = _mu_all[teacher_latents_df_idx].detach().cpu()
    teacher_latents_smiles = [aem_smiles_all[i] for i in teacher_latents_df_idx]
    print(
        f"[OK] Extracted {teacher_latents.shape[0]} real teacher latents, dim={teacher_latents.shape[1]}."
    )
    from sklearn.decomposition import PCA
    from latent_materials.adapter_model import ConditionalAdapter

    adapter_cond_cols = AUX_COLS + [TARGET_COL]
    adapter_df = df.iloc[teacher_latents_df_idx].reset_index(drop=True)
    adapter_cond_scaler = StandardScaler()
    Xc_adapter = adapter_cond_scaler.fit_transform(
        adapter_df[adapter_cond_cols].values.astype(np.float32)
    )
    X_tensor = torch.tensor(Xc_adapter, dtype=torch.float32, device=device)
    Y_tensor = teacher_latents.to(device)
    adapter = ConditionalAdapter(
        input_dim=X_tensor.shape[1], latent_dim=VAE_LATENT_DIM
    ).to(device)
    adapter_opt = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=0.0001)
    adapter_loss_fn = nn.MSELoss()
    adapter_epochs, adapter_bs = (1000, 32)
    adapter_loss_trace = []
    for epoch in range(1, adapter_epochs + 1):
        perm = torch.randperm(len(X_tensor))
        total_loss = 0.0
        for i in range(0, len(X_tensor), adapter_bs):
            idx = perm[i : i + adapter_bs]
            xb, yb = (X_tensor[idx], Y_tensor[idx])
            pred = adapter(xb)
            loss = adapter_loss_fn(pred, yb)
            adapter_opt.zero_grad()
            loss.backward()
            adapter_opt.step()
            total_loss += loss.item()
        adapter_loss_trace.append(total_loss / max(1, len(X_tensor) / adapter_bs))
        if epoch % 100 == 0 or epoch == 1:
            print(
                f"[Adapter] epoch {epoch:04d} | train_loss={adapter_loss_trace[-1]:.6f}"
            )
    adapter.eval()
    with torch.no_grad():
        z_pred_check = adapter(X_tensor)
        adapter_mse = torch.mean((z_pred_check - Y_tensor) ** 2).item()
        adapter_corr = float(
            np.corrcoef(
                z_pred_check.cpu().numpy().ravel(), Y_tensor.cpu().numpy().ravel()
            )[0, 1]
        )
    print(
        f"\n[Adapter] alignment -> MSE: {adapter_mse:.5f} | Pearson corr: {adapter_corr:.4f}"
    )
    _pca_check = PCA(n_components=2, random_state=SEED)
    _zt = Y_tensor.cpu().numpy()
    _zp = z_pred_check.cpu().numpy()
    _proj_t = _pca_check.fit_transform(_zt)
    _proj_p = _pca_check.transform(_zp)
    plt.figure(figsize=(5, 5))
    plt.scatter(
        _proj_t[:, 0],
        _proj_t[:, 1],
        s=14,
        c="#0072B2",
        alpha=0.5,
        label="Real teacher latent",
    )
    plt.scatter(
        _proj_p[:, 0],
        _proj_p[:, 1],
        s=14,
        c="#D55E00",
        alpha=0.5,
        label="Adapter prediction",
    )
    plt.title("AEM <-> pretrained-VAE latent alignment (corrected encoding)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    import re
    from sklearn.decomposition import PCA
    from latent_materials.aems.mutations import (
        add_imidazolium,
        add_phosphonium,
        add_quatN,
        extend_chain,
        insert_ether,
    )

    MUTATION_OPS = [
        add_quatN,
        add_imidazolium,
        add_phosphonium,
        extend_chain,
        insert_ether,
    ]
    HEAD_REGEX = re.compile(
        "\\[n\\+\\]|n\\+|\\[p\\+\\]|p\\+|\\[s\\+\\]|s\\+|imid|pyrid|guanid",
        re.IGNORECASE,
    )

    def has_cationic_head(smi):
        return bool(smi) and bool(HEAD_REGEX.search(smi.lower()))

    def has_peroxy_smi(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return True
        return m.HasSubstructMatch(Chem.MolFromSmarts("[O;X2]-[O;X2]"))

    def longest_aliphatic_chain(smi):
        matches = re.findall("C+", smi)
        return max((len(c) for c in matches), default=0)

    def structure_pass_aem(smi, hydrophobe_min=4):
        if not smi:
            return False
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        if has_peroxy_smi(smi):
            return False
        if not has_cationic_head(smi):
            return False
        if longest_aliphatic_chain(smi) < hydrophobe_min:
            return False
        return True

    def canon_smiles(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        try:
            Chem.SanitizeMol(m)
        except Exception:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    def try_decode_selfies_vector(vec):
        """Genuine VAE decode attempt: one-hot argmax over (max_len, alphabet) -> SELFIES -> SMILES."""
        try:
            arr = vec.reshape(VAE_MAX_LEN, len(VAE_ALPHABET))
            one_hot = np.zeros_like(arr)
            one_hot[np.arange(VAE_MAX_LEN), arr.argmax(axis=1)] = 1.0
            selfies_str = sf.encoding_to_selfies(
                one_hot.tolist(), VAE_VOCAB_ITOS, enc_type="one_hot"
            )
            smi = sf.decoder(selfies_str)
            return canon_smiles(smi)
        except Exception:
            return None

    _hc_quantile = adapter_df[TARGET_COL].quantile(0.75)
    high_cond_mask = adapter_df[TARGET_COL] >= _hc_quantile
    seed_rows = adapter_df[high_cond_mask].reset_index(drop=True)
    if len(seed_rows) == 0:
        seed_rows = adapter_df.reset_index(drop=True)
    print(
        f"[GEN] Conditioning generation on {len(seed_rows)} high-conductivity seed rows (>= {_hc_quantile:.4f} S/cm)."
    )
    seed_cond = adapter_cond_scaler.transform(
        seed_rows[adapter_cond_cols].values.astype(np.float32)
    )
    seed_cond_t = torch.tensor(seed_cond, dtype=torch.float32, device=device)
    with torch.no_grad():
        seed_mu = adapter(seed_cond_t)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    known_smiles = set(
        (c for c in (canon_smiles(s) for s in df["SMILES"].astype(str)) if c)
    )
    anchor_pool_smiles = teacher_latents_smiles
    anchor_pool_latents = teacher_latents.numpy()
    SIGMA_CHOICES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    direct_decoded, mutated = (set(), set())
    TARGET_NOVEL = 250
    MAX_ATTEMPTS = 15000
    attempts = 0
    while len(direct_decoded) + len(mutated) < TARGET_NOVEL and attempts < MAX_ATTEMPTS:
        sigma = float(np.random.choice(SIGMA_CHOICES))
        with torch.no_grad():
            z_batch = seed_mu + torch.randn_like(seed_mu) * sigma
            decoded_batch = vae.decoder(z_batch).cpu().numpy()
        z_batch_np = z_batch.detach().cpu().numpy()
        for i in range(decoded_batch.shape[0]):
            if (
                len(direct_decoded) + len(mutated) >= TARGET_NOVEL
                or attempts >= MAX_ATTEMPTS
            ):
                break
            attempts += 1
            direct_smi = try_decode_selfies_vector(decoded_batch[i])
            if (
                direct_smi
                and direct_smi not in known_smiles
                and structure_pass_aem(direct_smi)
            ):
                direct_decoded.add(direct_smi)
                continue
            z_i = z_batch_np[i]
            dists = np.linalg.norm(anchor_pool_latents - z_i[None, :], axis=1)
            anchor_smi = anchor_pool_smiles[int(np.argmin(dists))]
            for op in random.sample(MUTATION_OPS, k=2):
                cand = canon_smiles(op(anchor_smi))
                if cand and cand not in known_smiles and structure_pass_aem(cand):
                    mutated.add(cand)
                    break
    novel_candidates = sorted(direct_decoded | mutated)
    print(
        f"[GEN] attempts={attempts} | direct-decode hits={len(direct_decoded)} | mutation hits={len(mutated)} | total novel candidates={len(novel_candidates)}"
    )
    novel_candidates_df = pd.DataFrame(
        {
            "SMILES": novel_candidates,
            "source": [
                "direct_decode" if s in direct_decoded else "anchor_mutation"
                for s in novel_candidates
            ],
        }
    )
    novel_candidates_path = os.path.join(DATA_DIR, "aem_novel_candidates_clean.csv")
    novel_candidates_df.to_csv(novel_candidates_path, index=False)
    print(f"[OK] Saved novel candidates -> {novel_candidates_path}")
    if len(novel_candidates) > 0:
        _preview_mols = [
            m
            for m in (Chem.MolFromSmiles(s) for s in novel_candidates[:12])
            if m is not None
        ]
        if _preview_mols:
            display(
                Draw.MolsToGridImage(_preview_mols, molsPerRow=4, subImgSize=(260, 260))
            )

    def build_candidate_graphs(smiles_list, cfg, aux_scaler, aux_fill_row):
        """Inference-only graphs for candidate SMILES, using the mean AEM aux-feature
        vector as a stand-in (candidates have no measured block-ratio composition)."""
        aux_scaled_fill = aux_scaler.transform(
            aux_fill_row.reshape(1, -1).astype(np.float32)
        )[0]
        graphs = []
        for smi in smiles_list:
            g = smiles_to_pyg(smi, 0.0, aux_scaled_fill.astype(np.float32), cfg)
            if g is not None:
                graphs.append(g)
        return graphs

    _aux_fill = df[AUX_COLS].values.astype(np.float32).mean(axis=0)
    candidate_graphs = build_candidate_graphs(
        novel_candidates, cfg, aux_scaler, _aux_fill
    )
    print(
        f"[SCORE] Built {len(candidate_graphs)}/{len(novel_candidates)} valid candidate graphs."
    )
    if len(candidate_graphs) > 0:
        cand_loader = make_loader(candidate_graphs, cfg["batch_size"], shuffle=False)
        member_preds = []
        for m in models:
            m.eval()
            preds = []
            with torch.no_grad():
                for data in cand_loader:
                    data = data.to(device)
                    y_hat, _, _, _ = m(data)
                    preds.append(y_hat.detach().cpu().numpy().reshape(-1))
            member_preds.append(np.concatenate(preds))
        member_preds = np.vstack(member_preds)
        pred_mean_t = member_preds.mean(axis=0)
        pred_std_t = member_preds.std(axis=0)
        pred_mean = inverse_target(pred_mean_t)
        ranked_df = (
            pd.DataFrame(
                {
                    "SMILES": [g.smiles for g in candidate_graphs],
                    "Predicted_conductivity": pred_mean,
                    "Predicted_conductivity_std_transformed": pred_std_t,
                }
            )
            .sort_values("Predicted_conductivity", ascending=False)
            .reset_index(drop=True)
        )
        ranked_path = os.path.join(DATA_DIR, "aem_augmented_ranked_candidates_NN.csv")
        ranked_df.to_csv(ranked_path, index=False)
        print(f"[OK] Saved ranked candidates -> {ranked_path}")
        print(ranked_df.head(10).to_string(index=False))
    else:
        ranked_df = pd.DataFrame(columns=["SMILES", "Predicted_conductivity"])
        print("[WARN] No valid candidate graphs to score.")
    pretrain_candidate_graphs = build_candidate_graphs(
        novel_candidates, cfg, aux_scaler, _aux_fill
    )
    expanded_pretrain_graphs = train_graphs + pretrain_candidate_graphs
    print(
        f"[PRETRAIN] Expanded AE-pretraining pool: {len(train_graphs)} labeled + {len(pretrain_candidate_graphs)} generated-candidate graphs = {len(expanded_pretrain_graphs)} total."
    )
    ae_init_expanded = pretrain_encoder(
        expanded_pretrain_graphs, cfg, bin_dim, cont_dim, seed=SEED
    )
    print("[OK] Re-ran AE pretraining on the expanded pool -> ae_init_expanded.")
    if len(ranked_df) > 0:
        lo_q, hi_q = ranked_df["Predicted_conductivity"].quantile([0.01, 0.99])
        pseudo_df = ranked_df[
            (ranked_df["Predicted_conductivity"] >= lo_q)
            & (ranked_df["Predicted_conductivity"] <= hi_q)
        ].reset_index(drop=True)
        print(
            f"[PSEUDO] Kept {len(pseudo_df)}/{len(ranked_df)} candidates after dropping the extreme 1% prediction-quantile tails."
        )
        pseudo_y_transformed = target_transformer.transform(
            pseudo_df["Predicted_conductivity"].values.reshape(-1, 1).astype(np.float32)
        ).reshape(-1)
        _aux_scaled_fill_pseudo = aux_scaler.transform(_aux_fill.reshape(1, -1))[
            0
        ].astype(np.float32)
        pseudo_graphs = []
        for smi, y_t in zip(pseudo_df["SMILES"], pseudo_y_transformed):
            g = smiles_to_pyg(smi, float(y_t), _aux_scaled_fill_pseudo, cfg)
            if g is not None:
                g.sample_weight = torch.tensor([0.5], dtype=torch.float32)
                pseudo_graphs.append(g)
        print(f"[PSEUDO] Built {len(pseudo_graphs)} pseudo-labeled training graphs.")
        finetune_train_graphs = train_graphs + pseudo_graphs
    else:
        pseudo_graphs = []
        finetune_train_graphs = train_graphs
        print(
            "[PSEUDO] No ranked candidates available -- skipping pseudo-label augmentation."
        )

    def finetune_with_pseudo_labels(
        base_model,
        ae_init_for_shape,
        finetune_graphs,
        val_graphs,
        cfg,
        bin_dim,
        cont_dim,
        freeze_epochs=40,
        total_epochs=150,
        swa_start_frac=0.8,
        finetune_lr=0.0005,
        seed=SEED,
    ):
        torch.manual_seed(seed)
        model = base_model
        for p in model.encoder.parameters():
            p.requires_grad = False
        train_loader = make_loader(finetune_graphs, cfg["batch_size"], shuffle=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=finetune_lr,
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        swa_start_epoch = int(swa_start_frac * total_epochs)
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=finetune_lr * 0.1)
        bce = nn.BCEWithLogitsLoss()
        mse_recon = nn.MSELoss()
        best_rmse, best_state = (float("inf"), None)
        unfrozen = False
        for epoch in range(total_epochs):
            if epoch == freeze_epochs and (not unfrozen):
                for p in model.encoder.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=finetune_lr, weight_decay=cfg["weight_decay"]
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=10
                )
                swa_scheduler = SWALR(optimizer, swa_lr=finetune_lr * 0.1)
                unfrozen = True
                print(
                    f"  [unfreeze] epoch {epoch}: encoder unfrozen, lr reset to {finetune_lr}"
                )
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                data = augment_train_batch(data, cfg)
                y_hat, x_logits, _, _ = model(data)
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
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            val_metrics = eval_model(model, val_graphs, cfg, label="ft_val")
            if epoch < swa_start_epoch:
                scheduler.step(val_metrics["rmse"])
            else:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            if val_metrics["rmse"] < best_rmse:
                best_rmse = val_metrics["rmse"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if (epoch + 1) % 25 == 0 or epoch == 0 or epoch == total_epochs - 1:
                print(
                    f"  [FT] epoch {epoch + 1:04d}/{total_epochs} | val RMSE={val_metrics['rmse']:.6f} | val R2={val_metrics['r2']:.4f}"
                )
        best_model = JointGraphAER(
            ae_init_for_shape["node_in"],
            ae_init_for_shape["edge_in"],
            cfg,
            num_extra_feats=len(AUX_COLS),
        ).to(device)
        best_model.load_state_dict(
            best_state if best_state is not None else model.state_dict()
        )
        update_bn(train_loader, swa_model, device=device)
        best_metrics = eval_model(best_model, val_graphs, cfg, label="ft_best_val")
        swa_metrics = eval_model(swa_model, val_graphs, cfg, label="ft_swa_val")
        if best_metrics["rmse"] <= swa_metrics["rmse"]:
            return (best_model, best_metrics)
        return (swa_model, swa_metrics)

    print(
        "[Phase 3] Fine-tuning each ensemble member with pseudo-labeled candidates..."
    )
    models_ft = []
    for i, m in enumerate(models):
        print(f"\n-- Fine-tuning ensemble member {i + 1}/{len(models)} --")
        m_base = m.module if isinstance(m, AveragedModel) else m
        m_ft = copy.deepcopy(m_base)
        m_ft.encoder.load_state_dict(ae_init_expanded["encoder"])
        m_ft.decoder.load_state_dict(ae_init_expanded["decoder"])
        ft_model, ft_val_metrics = finetune_with_pseudo_labels(
            m_ft,
            ae_init_expanded,
            finetune_train_graphs,
            val_graphs,
            cfg,
            bin_dim,
            cont_dim,
            seed=SEED + i * 123,
        )
        models_ft.append(ft_model)
    print("\n[OK] Finished pseudo-label fine-tuning for all ensemble members.")
    from latent_materials.validation import select_by_validation_rmse

    models, selected_index, selection_evaluations = select_by_validation_rmse(
        [models, models_ft], val_graphs, cfg, eval_ensemble
    )
    train_metrics = eval_ensemble(models, train_graphs, cfg, label="train")
    val_metrics = selection_evaluations[selected_index]
    test_metrics = eval_ensemble(models, test_graphs, cfg, label="final_test")
    for metrics in [train_metrics, val_metrics, test_metrics]:
        print_metrics(metrics)

    def _safe_metrics_dict(m):
        return {k: float(m[k]) for k in ["mae", "mse", "rmse", "r2"]}

    with open(
        os.path.join(DATA_DIR, "aem_validation_selected_metrics.json"), "w"
    ) as handle:
        json.dump(
            {
                "train": _safe_metrics_dict(train_metrics),
                "val": _safe_metrics_dict(val_metrics),
                "test": _safe_metrics_dict(test_metrics),
            },
            handle,
            indent=2,
        )
    from latent_materials.exports import export_ensemble_snapshot

    export_ensemble_snapshot(
        Path(DATA_DIR) / "validation_selected_ensemble",
        models=models,
        config=cfg,
        aux_columns=AUX_COLS,
        aux_scaler=aux_scaler,
        target_transformer=target_transformer,
        evaluations={
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        metadata={
            "seed": SEED,
            "target_col": TARGET_COL,
            "selection_protocol": "minimum validation RMSE; exact ties retain phase1",
            "selected_candidate": int(selected_index),
            "scope": "new realization; not historical manuscript metrics",
        },
    )
    available = locals().copy()
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    try:
        import torch
    except Exception:
        torch = None
    if "DATA_DIR" not in available:
        DATA_DIR = os.getcwd()
    AEM_FIG_DIR = os.path.join(DATA_DIR, "aem_publication_figures")
    os.makedirs(AEM_FIG_DIR, exist_ok=True)
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
            "legend.fontsize": 9.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    curve_color = "#0072B2"
    scatter_color = "#D55E00"
    green_color = "#009E73"
    purple_color = "#CC79A7"
    sky_color = "#56B4E9"
    orange_color = "#E69F00"
    edge_color = "#1A1A1A"
    identity_color = "#333333"
    grid_color = "#B8B8B8"

    def _require_metric_dicts():
        required = ["train_metrics", "val_metrics", "test_metrics"]
        missing = [v for v in required if v not in available]
        if missing:
            raise ValueError(
                f"Missing metric variables: {missing}. Evaluate the selected model before plotting."
            )

    def _safe_arr(x):
        return np.asarray(x, dtype=float).reshape(-1)

    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    def _r2(y_true, y_pred):
        y_true = _safe_arr(y_true)
        y_pred = _safe_arr(y_pred)
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(r2_score(y_true, y_pred))

    def _get_history():
        hist = available.get("hist")
        "Use the first ensemble member history from the AEM notebook."
        if (
            "histories" in available
            and isinstance(histories, list)
            and (len(histories) > 0)
        ):
            return histories[0]
        if "hist" in available and isinstance(hist, dict):
            return hist
        raise ValueError("Could not find histories[0] or hist. Run training first.")

    def _metric_table_from_dicts():
        _require_metric_dicts()
        rows = []
        for split_name, m in [
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Test", test_metrics),
        ]:
            y_true = _safe_arr(m["y_true"])
            y_pred = _safe_arr(m["y_pred"])
            rows.append(
                {
                    "Split": split_name,
                    "MAE": _mae(y_true, y_pred),
                    "RMSE": _rmse(y_true, y_pred),
                    "R2": _r2(y_true, y_pred),
                    "N": len(y_true),
                }
            )
        return pd.DataFrame(rows)

    def _test_residual_dataframe():
        _require_metric_dicts()
        y_true = _safe_arr(test_metrics["y_true"])
        y_pred = _safe_arr(test_metrics["y_pred"])
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        out["residual"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["residual"])
        out["squared_error"] = out["residual"] ** 2
        return out

    def _conductivity_bin_error_table(q=4):
        """Make conductivity-bin summary for held-out test set."""
        df_err = _test_residual_dataframe()
        y = df_err["y_true"].values
        q_eff = min(q, len(np.unique(y)), len(y))
        if q_eff < 2:
            df_err["conductivity_bin"] = "all"
        else:
            codes = pd.qcut(df_err["y_true"], q=q_eff, labels=False, duplicates="drop")
            n_bins = int(np.nanmax(codes)) + 1 if np.any(pd.notna(codes)) else 1
            names = (
                ["low", "mid-low", "mid-high", "high"]
                if n_bins == 4
                else [f"bin {i + 1}" for i in range(n_bins)]
            )
            df_err["conductivity_bin"] = [
                names[int(c)] if pd.notna(c) else "unknown" for c in codes
            ]
        summary = (
            df_err.groupby("conductivity_bin", observed=False)
            .agg(
                n=("y_true", "count"),
                y_true_min=("y_true", "min"),
                y_true_max=("y_true", "max"),
                MAE=("abs_error", "mean"),
                RMSE=("squared_error", lambda x: np.sqrt(np.mean(x))),
            )
            .reset_index()
        )
        return (df_err, summary)

    def _panel_label(ax, label, x=-0.14, y=1.05, fontsize=16):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    def _extract_embeddings(model, graph_list, batch_size=64):
        """Extract graph-level embeddings z_graph from trained JointGraphAER model."""
        if torch is None:
            raise ValueError("torch is not available.")
        if "make_loader" not in available:
            raise ValueError(
                "make_loader is missing. Initialize the graph loaders before plotting."
            )
        if "device" not in available:
            raise ValueError(
                "device is missing. Initialize the compute device before plotting."
            )
        model.eval()
        zs = []
        loader = make_loader(graph_list, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                _, _, _, z_graph = model(data)
                zs.append(z_graph.detach().cpu().numpy())
        return np.vstack(zs)

    def _score_bo_candidates_for_figure3():
        """Build + score the BO-generated candidate pool with the trained ensemble,
        for Figure 3's NN-vs-BO overlay. Mirrors Section 9E's NN scoring. Gracefully
        returns (None, []) if the BO candidates file or trained models aren't available.
        """
        bo_path = os.path.join(DATA_DIR, "aem_novel_candidates_clean_BO.csv")
        if (
            not os.path.exists(bo_path)
            or "models" not in available
            or "cfg" not in available
        ):
            return (None, [])
        bo_smiles_df = pd.read_csv(bo_path)
        aux_fill_bo = df[AUX_COLS].values.astype(np.float32).mean(axis=0)
        graphs_bo = build_candidate_graphs(
            bo_smiles_df["SMILES"].tolist(), cfg, aux_scaler, aux_fill_bo
        )
        if len(graphs_bo) == 0:
            return (None, [])
        loader_bo = make_loader(graphs_bo, cfg["batch_size"], shuffle=False)
        member_preds_bo = []
        for m in models:
            m.eval()
            preds = []
            with torch.no_grad():
                for data in loader_bo:
                    data = data.to(device)
                    y_hat, _, _, _ = m(data)
                    preds.append(y_hat.detach().cpu().numpy().reshape(-1))
            member_preds_bo.append(np.concatenate(preds))
        member_preds_bo = np.vstack(member_preds_bo)
        pred_mean_bo = inverse_target(member_preds_bo.mean(axis=0))
        pred_std_bo = member_preds_bo.std(axis=0)
        ranked_bo = (
            pd.DataFrame(
                {
                    "SMILES": [g.smiles for g in graphs_bo],
                    "Predicted_conductivity": pred_mean_bo,
                    "Predicted_conductivity_std_transformed": pred_std_bo,
                }
            )
            .sort_values("Predicted_conductivity", ascending=False)
            .reset_index(drop=True)
        )
        ranked_bo_path = os.path.join(
            DATA_DIR, "aem_augmented_ranked_candidates_BO.csv"
        )
        ranked_bo.to_csv(ranked_bo_path, index=False)
        print(
            f"[Figure 3] Scored {len(graphs_bo)} BO candidates with the trained ensemble -> {ranked_bo_path}"
        )
        return (ranked_bo, graphs_bo)

    def _load_ranked_candidates_if_available():
        pred_df = available.get("pred_df")
        "Return ranked candidate dataframe if pred_df exists or a saved candidate file exists."
        if (
            "pred_df" in available
            and isinstance(pred_df, pd.DataFrame)
            and (len(pred_df) > 0)
        ):
            return pred_df.copy()
        possible_paths = [
            os.path.join(DATA_DIR, "aem_augmented_ranked_candidates_NN.csv"),
            os.path.join(DATA_DIR, "aem_augmented_predicted_conductivity_NN.csv"),
            os.path.join(DATA_DIR, "aem_novel_candidates_clean_2_final.csv"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return pd.read_csv(p)
        return None

    def make_aem_figure_1_conductivity_parity():
        _require_metric_dicts()
        y_true = _safe_arr(test_metrics["y_true"])
        y_pred = _safe_arr(test_metrics["y_pred"])
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.8))
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        pad = 0.07 * (max_val - min_val + 1e-09)
        lims = [min_val - pad, max_val + pad]
        ax.scatter(
            y_true,
            y_pred,
            s=48,
            alpha=0.78,
            color=scatter_color,
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
        ax.set_xlabel("Experimental AEM conductivity")
        ax.set_ylabel("Predicted AEM conductivity")
        ax.set_title("Held-out test-set parity plot", pad=8)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3, color=grid_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best")
        fig.suptitle(
            "Test-set prediction accuracy of the AEM conductivity GNN surrogate",
            fontsize=15.5,
            y=0.98,
        )
        fig.text(
            0.5,
            -0.02,
            "The dashed line represents perfect agreement between predicted and experimental conductivity values.",
            ha="center",
            va="bottom",
            fontsize=10.2,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
        out_png = os.path.join(AEM_FIG_DIR, "figure1_aem_conductivity_parity.png")
        out_pdf = os.path.join(AEM_FIG_DIR, "figure1_aem_conductivity_parity.pdf")
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.show()
        print(f"Saved Figure 1 PNG: {out_png}")
        print(f"Saved Figure 1 PDF: {out_pdf}")

    def make_aem_figure_2_performance_summary():
        _require_metric_dicts()
        hist = _get_history()
        metric_df = _metric_table_from_dicts()
        err_df, bin_summary = _conductivity_bin_error_table(q=4)
        member_rows = []
        if (
            "models" in available
            and "test_graphs" in available
            and ("eval_model" in available)
        ):
            for i, model_i in enumerate(models):
                try:
                    mi = eval_model(model_i, test_graphs, cfg, label=f"member_{i + 1}")
                    yt = _safe_arr(mi["y_true"])
                    yp = _safe_arr(mi["y_pred"])
                    member_rows.append(
                        {
                            "Member": i + 1,
                            "RMSE": _rmse(yt, yp),
                            "MAE": _mae(yt, yp),
                            "R2": _r2(yt, yp),
                        }
                    )
                except Exception as e:
                    print(f"[WARN] Could not evaluate ensemble member {i + 1}: {e}")
        member_df = pd.DataFrame(member_rows)
        fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.4))
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.91, bottom=0.1, wspace=0.3, hspace=0.36
        )
        ax = axes[0, 0]
        epoch = np.asarray(
            hist.get("epoch", np.arange(1, len(hist.get("train_loss", [])) + 1))
        )
        train_loss = np.asarray(hist.get("train_loss", []), dtype=float)
        if len(epoch) > 0 and len(train_loss) > 0:
            ax.plot(
                epoch,
                train_loss,
                color=curve_color,
                linewidth=2.3,
                label="Training loss",
            )
        if "val_rmse" in hist and len(hist["val_rmse"]) == len(epoch):
            ax.plot(
                epoch,
                np.asarray(hist["val_rmse"], dtype=float),
                color=scatter_color,
                linewidth=2.3,
                label="Validation RMSE",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / RMSE")
        ax.set_title("Training and validation history", pad=9)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32, color=grid_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax_r2 = ax.twinx()
        if "val_r2" in hist and len(hist["val_r2"]) == len(epoch):
            ax_r2.plot(
                epoch,
                np.asarray(hist["val_r2"], dtype=float),
                color=green_color,
                linewidth=2.0,
                linestyle="-.",
                label="Validation $R^2$",
            )
            ax_r2.set_ylabel("Validation $R^2$")
            ax_r2.spines["top"].set_visible(False)
        else:
            print("[WARN] No validation R² history found in hist.")
            ax_r2.set_yticks([])
            ax_r2.spines["right"].set_visible(False)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_r2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        _panel_label(ax, "A")
        ax = axes[0, 1]
        residuals = err_df["residual"].values
        bins = min(12, max(5, int(np.sqrt(len(residuals)))))
        ax.hist(
            residuals,
            bins=bins,
            alpha=0.78,
            color=scatter_color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.axvline(
            0,
            linestyle="--",
            linewidth=1.5,
            color=identity_color,
            label="Zero residual",
        )
        ax.axvline(
            np.mean(residuals),
            linestyle="-",
            linewidth=1.5,
            color=curve_color,
            label="Mean residual",
        )
        ax.set_xlabel("Prediction residual: predicted − experimental")
        ax.set_ylabel("Count")
        ax.set_title("Held-out test residual distribution", pad=9)
        ax.grid(
            True, axis="y", linestyle="--", linewidth=0.7, alpha=0.32, color=grid_color
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best")
        ax.text(
            0.03,
            0.95,
            f"Mean: {np.mean(residuals):.3f}\nSD: {np.std(residuals, ddof=1):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.95,
            ),
        )
        _panel_label(ax, "B")
        ax = axes[1, 0]
        x = np.arange(len(bin_summary))
        width = 0.35
        ax.bar(
            x - width / 2,
            bin_summary["RMSE"],
            width,
            color=curve_color,
            edgecolor="black",
            linewidth=0.7,
            label="RMSE",
        )
        ax.bar(
            x + width / 2,
            bin_summary["MAE"],
            width,
            color=scatter_color,
            edgecolor="black",
            linewidth=0.7,
            label="MAE",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bin_summary["conductivity_bin"], rotation=0)
        ax.set_ylabel("Error")
        ax.set_title("Prediction error by conductivity range", pad=9)
        ax.grid(
            True, axis="y", linestyle="--", linewidth=0.7, alpha=0.32, color=grid_color
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best")
        _panel_label(ax, "C")
        ax = axes[1, 1]
        if len(member_df) > 0:
            ax.plot(
                member_df["Member"],
                member_df["RMSE"],
                marker="o",
                linewidth=2.1,
                color=curve_color,
                label="Test RMSE",
            )
            ax.set_xlabel("Ensemble member")
            ax.set_ylabel("RMSE")
            ax2 = ax.twinx()
            ax2.plot(
                member_df["Member"],
                member_df["R2"],
                marker="s",
                linestyle="--",
                linewidth=2.0,
                color=scatter_color,
                label="Test $R^2$",
            )
            ax2.set_ylabel("$R^2$")
            ax2.spines["top"].set_visible(False)
            ax.set_title("Ensemble-member test performance", pad=9)
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32, color=grid_color)
            ax.spines["top"].set_visible(False)
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        else:
            x = np.arange(len(metric_df))
            ax.bar(
                x, metric_df["R2"], color=green_color, edgecolor="black", linewidth=0.7
            )
            ax.axhline(0, linestyle="--", linewidth=1.1, color=identity_color)
            ax.set_xticks(x)
            ax.set_xticklabels(metric_df["Split"])
            ax.set_ylabel("$R^2$")
            ax.set_title("Explained variance by data split", pad=9)
            ax.grid(
                True,
                axis="y",
                linestyle="--",
                linewidth=0.7,
                alpha=0.32,
                color=grid_color,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        _panel_label(ax, "D")
        fig.suptitle(
            "Model diagnostics for the AEM conductivity prediction surrogate",
            fontsize=16.0,
            y=0.98,
        )
        fig.text(
            0.5,
            0.025,
            "Errors are reported in the original conductivity scale after inverse target transformation.",
            ha="center",
            va="bottom",
            fontsize=9.8,
        )
        out_png = os.path.join(AEM_FIG_DIR, "figure2_aem_model_diagnostics_2x2.png")
        out_pdf = os.path.join(AEM_FIG_DIR, "figure2_aem_model_diagnostics_2x2.pdf")
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.show()
        print(f"Saved Figure 2 PNG: {out_png}")
        print(f"Saved Figure 2 PDF: {out_pdf}")

    def make_aem_figure_3_latent_candidate_space():
        if "models" not in available or len(models) == 0:
            raise ValueError(
                "Missing trained models. Run the final ensemble training section first."
            )
        if "train_graphs" not in available or "test_graphs" not in available:
            raise ValueError(
                "Missing train_graphs/test_graphs. Build the training and test graphs before plotting."
            )
        model = models[0]
        z_train = _extract_embeddings(
            model, train_graphs, batch_size=cfg.get("batch_size", 64)
        )
        z_test = _extract_embeddings(
            model, test_graphs, batch_size=cfg.get("batch_size", 64)
        )
        if "val_graphs" in available:
            z_val = _extract_embeddings(
                model, val_graphs, batch_size=cfg.get("batch_size", 64)
            )
        else:
            z_val = np.empty((0, z_train.shape[1]))
        if "candidate_graphs" in available and len(candidate_graphs) > 0:
            z_gen = _extract_embeddings(
                model, candidate_graphs, batch_size=cfg.get("batch_size", 64)
            )
        else:
            z_gen = np.empty((0, z_train.shape[1]))
        ranked_bo_candidates, graphs_bo = _score_bo_candidates_for_figure3()
        if len(graphs_bo) > 0:
            z_gen_bo = _extract_embeddings(
                model, graphs_bo, batch_size=cfg.get("batch_size", 64)
            )
        else:
            z_gen_bo = np.empty((0, z_train.shape[1]))
        labels = (
            ["Train"] * len(z_train)
            + ["Validation"] * len(z_val)
            + ["Test"] * len(z_test)
            + ["Generated (NN)"] * len(z_gen)
            + ["Generated (BO)"] * len(z_gen_bo)
        )
        z_parts = [z_train, z_val, z_test]
        if len(z_gen) > 0:
            z_parts.append(z_gen)
        if len(z_gen_bo) > 0:
            z_parts.append(z_gen_bo)
        z_all = np.vstack(z_parts)
        z_scaler = StandardScaler()
        z_all_scaled = z_scaler.fit_transform(z_all)
        pca = PCA(n_components=2, random_state=SEED if "SEED" in available else 42)
        proj = pca.fit_transform(z_all_scaled)
        plot_df = pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "Split": labels})
        ranked_candidates = _load_ranked_candidates_if_available()
        fig, axes = plt.subplots(
            1, 2, figsize=(12.8, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0]}
        )
        ax = axes[0]
        for split, color, marker, alpha in [
            ("Train", curve_color, "o", 0.58),
            ("Validation", green_color, "^", 0.7),
            ("Test", scatter_color, "s", 0.78),
            ("Generated (NN)", purple_color, "D", 0.85),
            ("Generated (BO)", orange_color, "P", 0.8),
        ]:
            sub = plot_df[plot_df["Split"] == split]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=34,
                alpha=alpha,
                color=color,
                marker=marker,
                edgecolors="none",
                label=split,
            )
        lo1, hi1 = np.percentile(proj[:, 0], [2, 98])
        lo2, hi2 = np.percentile(proj[:, 1], [2, 98])
        pad1 = 0.15 * (hi1 - lo1 + 1e-09)
        pad2 = 0.15 * (hi2 - lo2 + 1e-09)
        xlim = (lo1 - pad1, hi1 + pad1)
        ylim = (lo2 - pad2, hi2 + pad2)
        n_outside = int(
            (
                (proj[:, 0] < xlim[0])
                | (proj[:, 0] > xlim[1])
                | (proj[:, 1] < ylim[0])
                | (proj[:, 1] > ylim[1])
            ).sum()
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if n_outside > 0:
            print(
                f"[Figure 3] {n_outside}/{len(proj)} point(s) fall outside the robust display window (still fit by PCA, just not shown in this view)."
            )
        evr = pca.explained_variance_ratio_
        ax.set_xlabel(f"Principal component 1 ({100 * evr[0]:.1f}% variance)")
        ax.set_ylabel(f"Principal component 2 ({100 * evr[1]:.1f}% variance)")
        ax.set_title("AEM graph-embedding space (standardized before PCA)", pad=8)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3, color=grid_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best")
        _panel_label(ax, "A", fontsize=14)
        ax = axes[1]
        candidate_col_candidates = [
            "Predicted_Anion_Conductivity",
            "Predicted_conductivity",
            "Predicted_Conductivity",
            "predicted_conductivity",
            "AnionConductivity_pred",
        ]
        candidate_col = None
        if ranked_candidates is not None:
            for c in candidate_col_candidates:
                if c in ranked_candidates.columns:
                    candidate_col = c
                    break
        if (
            ranked_candidates is not None
            and candidate_col is not None
            and (len(ranked_candidates) > 0)
            and (ranked_bo_candidates is not None)
        ):
            pred_nn = pd.to_numeric(
                ranked_candidates[candidate_col], errors="coerce"
            ).dropna()
            pred_bo = pd.to_numeric(
                ranked_bo_candidates["Predicted_conductivity"], errors="coerce"
            ).dropna()
            bins = np.linspace(
                min(pred_nn.min(), pred_bo.min()), max(pred_nn.max(), pred_bo.max()), 24
            )
            ax.hist(
                pred_nn,
                bins=bins,
                alpha=0.55,
                color=purple_color,
                edgecolor="black",
                linewidth=0.6,
                label=f"NN (n={len(pred_nn)})",
            )
            ax.hist(
                pred_bo,
                bins=bins,
                alpha=0.55,
                color=orange_color,
                edgecolor="black",
                linewidth=0.6,
                label=f"BO (n={len(pred_bo)})",
            )
            ax.set_xlabel("Predicted AEM conductivity")
            ax.set_ylabel("Candidate count")
            ax.set_title("Generated-candidate conductivity distribution", pad=8)
            ax.legend(loc="best")
            note = f"Predicted with the trained ensemble. NN: n={len(pred_nn)}. BO: n={len(pred_bo)}."
        elif (
            ranked_candidates is not None
            and candidate_col is not None
            and (len(ranked_candidates) > 0)
        ):
            pred_cond = pd.to_numeric(
                ranked_candidates[candidate_col], errors="coerce"
            ).dropna()
            bins = min(18, max(6, int(np.sqrt(len(pred_cond)))))
            ax.hist(
                pred_cond,
                bins=bins,
                alpha=0.78,
                color=green_color,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.set_xlabel("Predicted AEM conductivity")
            ax.set_ylabel("Candidate count")
            ax.set_title("Generated-candidate conductivity distribution", pad=8)
            note = f"Candidate predictions available: n={len(pred_cond)}. Use the ranked candidate CSV for final screening."
        else:
            _require_metric_dicts()
            y_train = _safe_arr(train_metrics["y_true"])
            y_val = _safe_arr(val_metrics["y_true"])
            y_test = _safe_arr(test_metrics["y_true"])
            all_y = np.concatenate([y_train, y_val, y_test])
            bins = min(14, max(6, int(np.sqrt(len(all_y)))))
            ax.hist(
                y_train,
                bins=bins,
                alpha=0.48,
                color=curve_color,
                edgecolor="black",
                linewidth=0.8,
                label="Train",
            )
            ax.hist(
                y_val,
                bins=bins,
                alpha=0.48,
                color=green_color,
                edgecolor="black",
                linewidth=0.8,
                label="Validation",
            )
            ax.hist(
                y_test,
                bins=bins,
                alpha=0.55,
                color=scatter_color,
                edgecolor="black",
                linewidth=0.8,
                label="Test",
            )
            ax.set_xlabel("Experimental AEM conductivity")
            ax.set_ylabel("Count")
            ax.set_title("Conductivity distribution in model data", pad=8)
            ax.legend(loc="best")
            note = "Panel B shows the experimental conductivity distribution because no ranked candidate prediction file was found."
        ax.grid(
            True, axis="y", linestyle="--", linewidth=0.7, alpha=0.3, color=grid_color
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _panel_label(ax, "B", fontsize=14)
        fig.suptitle(
            "AEM graph latent-space structure and conductivity-based screening output (NN vs. BO)",
            fontsize=15.5,
            y=1.0,
        )
        fig.text(0.5, -0.02, note, ha="center", va="bottom", fontsize=10.0)
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        out_png = os.path.join(AEM_FIG_DIR, "figure3_aem_latent_candidate_space.png")
        out_pdf = os.path.join(AEM_FIG_DIR, "figure3_aem_latent_candidate_space.pdf")
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.show()
        print(f"Saved Figure 3 PNG: {out_png}")
        print(f"Saved Figure 3 PDF: {out_pdf}")

    make_aem_figure_1_conductivity_parity()
    make_aem_figure_2_performance_summary()
    make_aem_figure_3_latent_candidate_space()
    print("\nDone. Figures saved in:")
    print(AEM_FIG_DIR)
    from sklearn.model_selection import KFold

    CV_N_FOLDS = 5
    cv_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    kf = KFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=SEED)
    cv_fold_records = []
    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(cv_full_df)):
        fold_train_df = cv_full_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = cv_full_df.iloc[va_idx].reset_index(drop=True)
        fold_train_graphs = build_graph_list(
            fold_train_df,
            cfg,
            aux_scaler,
            target_transformer,
            transform_y=True,
            aux_columns=AUX_COLS,
            target_column=TARGET_COL,
        )
        fold_val_graphs = build_graph_list(
            fold_val_df,
            cfg,
            aux_scaler,
            target_transformer,
            transform_y=True,
            aux_columns=AUX_COLS,
            target_column=TARGET_COL,
        )
        fold_ae_init = pretrain_encoder(
            fold_train_graphs, cfg, bin_dim, cont_dim, seed=SEED + fold_i
        )
        fold_model, fold_hist, fold_val_metrics = train_single_joint(
            fold_train_graphs,
            fold_val_graphs,
            fold_ae_init,
            cfg,
            bin_dim,
            cont_dim,
            seed_offset=fold_i * 777,
        )
        cv_fold_records.append(
            {
                "fold": fold_i,
                "val_rmse": fold_val_metrics["rmse"],
                "val_mae": fold_val_metrics["mae"],
                "val_r2": fold_val_metrics["r2"],
            }
        )
        print(
            f"[CV] fold {fold_i + 1}/{CV_N_FOLDS} | val RMSE={fold_val_metrics['rmse']:.6f} | val MAE={fold_val_metrics['mae']:.6f} | val R2={fold_val_metrics['r2']:.4f}"
        )
    cv_summary_df = pd.DataFrame(cv_fold_records)
    cv_summary_path = os.path.join(DATA_DIR, "aem_gnn_5fold_summary.csv")
    cv_summary_df.to_csv(cv_summary_path, index=False)
    print(f"\n[OK] Saved 5-fold CV summary -> {cv_summary_path}")
    print(cv_summary_df.describe())
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
    _tp_color = "#0072B2"
    _tp_edge_color = "#1A1A1A"
    _tp_identity_color = "#333333"
    y_true_aem = _safe_arr(test_metrics["y_true"])
    y_pred_aem = _safe_arr(test_metrics["y_pred"])
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.8))
    min_val = min(y_true_aem.min(), y_pred_aem.min())
    max_val = max(y_true_aem.max(), y_pred_aem.max())
    pad = 0.07 * (max_val - min_val + 1e-09)
    lims = [min_val - pad, max_val + pad]
    ax.scatter(
        y_true_aem,
        y_pred_aem,
        s=42,
        alpha=0.78,
        color=_tp_color,
        edgecolor=_tp_edge_color,
        linewidth=0.35,
    )
    ax.plot(
        lims,
        lims,
        linestyle="--",
        linewidth=1.5,
        color=_tp_identity_color,
        label="Ideal prediction",
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Anion conductivity (S cm$^{-1}$)", pad=8)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.14, 1.05, "A", transform=ax.transAxes, fontsize=17, ha="left", va="bottom"
    )
    ax.legend(loc="best")
    fig.suptitle(
        "Test-set prediction accuracy of the AEM conductivity GNN surrogate",
        x=0.5,
        y=0.98,
    )
    fig.text(
        0.5,
        -0.025,
        "The dashed line represents perfect agreement between predicted and true values.",
        ha="center",
        va="bottom",
        fontsize=10.2,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    _out_png = os.path.join(AEM_FIG_DIR, "aem_true_vs_pred_publication.png")
    _out_pdf = os.path.join(AEM_FIG_DIR, "aem_true_vs_pred_publication.pdf")
    fig.savefig(_out_png, bbox_inches="tight")
    fig.savefig(_out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG: {_out_png}")
    print(f"Saved PDF: {_out_pdf}")
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.4))
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.91, bottom=0.1, wspace=0.3, hspace=0.36
    )
    axA, axB, axC, axD = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
    _h = histories[0]
    _epoch = np.asarray(_h.get("epoch", []), dtype=float)
    if len(_epoch) > 0:
        axA.plot(
            _epoch,
            _h["train_loss"],
            color=curve_color,
            linewidth=2.3,
            label="Training loss",
        )
        axA.plot(
            _epoch,
            _h["val_rmse"],
            color=scatter_color,
            linewidth=2.3,
            label="Validation RMSE",
        )
    axA.set_xlabel("Epoch")
    axA.set_ylabel("Loss / RMSE")
    axA.set_title(
        "Best-model training and validation history", pad=10, fontweight="semibold"
    )
    axA.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA_r2 = axA.twinx()
    if len(_epoch) > 0 and "val_r2" in _h:
        axA_r2.plot(
            _epoch,
            _h["val_r2"],
            color=green_color,
            linewidth=2.3,
            linestyle="-.",
            label="Validation $R^2$",
        )
        axA_r2.set_ylabel("Validation $R^2$")
        axA_r2.spines["top"].set_visible(False)
    lines_1, labels_1 = axA.get_legend_handles_labels()
    lines_2, labels_2 = axA_r2.get_legend_handles_labels()
    axA.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    axA.text(
        -0.08,
        1.04,
        "A",
        transform=axA.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    def _cv_hist_panel(ax, values, xlabel, color, letter):
        values = np.asarray(values, dtype=float)
        vmin, vmax = (values.min(), values.max())
        span = vmax - vmin
        pad = 0.12 * (span + 1e-09)
        bins = np.linspace(vmin - pad, vmax + pad, 7)
        ax.hist(
            values, bins=bins, alpha=0.68, edgecolor="black", linewidth=0.8, color=color
        )
        mean_v = values.mean()
        std_v = values.std(ddof=1) if len(values) > 1 else 0.0
        ax.axvline(mean_v, color=color, linestyle="--", linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fold count")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.03,
            0.95,
            f"{mean_v:.4f} $\\pm$ {std_v:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.95,
            ),
        )
        ax.text(
            -0.16,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    _cv_hist_panel(
        axB, cv_summary_df["val_rmse"], "Conductivity RMSE (S/cm)", "#0072B2", "B"
    )
    axB.set_title("Conductivity cross-validation RMSE", pad=10, fontweight="semibold")
    _cv_hist_panel(
        axC, cv_summary_df["val_mae"], "Conductivity MAE (S/cm)", "#D55E00", "C"
    )
    axC.set_title("Conductivity cross-validation MAE", pad=10, fontweight="semibold")
    _cv_hist_panel(axD, cv_summary_df["val_r2"], "$R^2$", "#009E73", "D")
    axD.set_title("Cross-validation $R^2$ distribution", pad=10, fontweight="semibold")
    fig.suptitle(
        "Training behavior and five-fold cross-validation performance of the AEM conductivity GNN surrogate",
        fontsize=16.0,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        "Conductivity RMSE and MAE retain the units of the target (S/cm); R2 is unitless.",
        ha="center",
        va="bottom",
        fontsize=9.4,
    )
    _out_png = os.path.join(AEM_FIG_DIR, "aem_training_performance_2x2_publication.png")
    _out_pdf = os.path.join(AEM_FIG_DIR, "aem_training_performance_2x2_publication.pdf")
    fig.savefig(_out_png, bbox_inches="tight")
    fig.savefig(_out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG: {_out_png}")
    print(f"Saved PDF: {_out_pdf}")
    return root
