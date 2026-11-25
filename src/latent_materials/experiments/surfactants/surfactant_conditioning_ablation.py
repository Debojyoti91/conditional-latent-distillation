"""Final five-variant, five-seed conditioning ablation and 20 partitionwise shuffles (Tables S7/S8).

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from latent_materials.surfactants.config import AblationConfig
from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace, config=None):
    """Execute the surfactant conditioning ablation experiment in a prepared workspace."""
    config = config or AblationConfig()
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = (
        str(root / "IEMS_GEN_files/AEM/CVAE") if "surfactant" == "aem" else str(root)
    )
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    import re
    import json
    import random
    from pathlib import Path

    try:
        from rdkit import Chem
    except ImportError:
        _external_setup()
        from rdkit import Chem
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    GLOBAL_SEED = config.global_seed
    OPTIMIZATION_SEEDS = config.optimization_seeds
    N_PROPERTY_SHUFFLES = config.property_shuffles
    BOOTSTRAP_REPLICATES = config.bootstrap_replicates
    MAX_EPOCHS = config.max_epochs
    PATIENCE = config.patience
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.learning_rate
    WEIGHT_DECAY = config.weight_decay
    LATENT_DIM = 32
    HIDDEN_DIM = 128
    EXPECTED_N = 881
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)

    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    set_global_seed(GLOBAL_SEED)
    DRIVE_ROOT = "/content/drive"
    pass
    pass
    BASE_DIR = Path(str(root))
    DATA_DIR = BASE_DIR / "data"
    OUT_DIR = BASE_DIR / "outputs"
    ABLATION_DIR = BASE_DIR / "conditioning_ablation_CORRECTED"
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_FILE = OUT_DIR / "genlib_candidates_v1.csv"
    PROPERTY_FILE = OUT_DIR / "candidates_with_predictions.csv"
    LATENT_FILE = DATA_DIR / "figure2_teacher_student_latents.npz"
    for path in [CANDIDATE_FILE, PROPERTY_FILE, LATENT_FILE]:
        if not path.is_file():
            raise FileNotFoundError(f"Required original artifact missing:\n{path}")
    print("\nBASE_DIR:", BASE_DIR)
    print("ABLATION_DIR:", ABLATION_DIR)
    print("\n[PASS] Required original artifacts found.")
    library = pd.read_csv(CANDIDATE_FILE)
    properties = pd.read_csv(PROPERTY_FILE)
    latent_data = np.load(LATENT_FILE, allow_pickle=False)
    if "teacher" not in latent_data.files:
        raise RuntimeError("Latent NPZ does not contain 'teacher'.")
    teacher = np.asarray(latent_data["teacher"], dtype=np.float32)
    historical_student = None
    if "student" in latent_data.files:
        historical_student = np.asarray(latent_data["student"], dtype=np.float32)
    print("\nLoaded artifacts")
    print("----------------")
    print("Candidate library :", library.shape)
    print("Property table    :", properties.shape)
    print("Teacher latent    :", teacher.shape)
    if historical_student is not None:
        print("Historical student:", historical_student.shape)
    if len(library) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} candidates; found {len(library)}.")
    if len(properties) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} property rows; found {len(properties)}."
        )
    if teacher.shape != (EXPECTED_N, LATENT_DIM):
        raise RuntimeError(
            f"Unexpected teacher shape. Expected ({EXPECTED_N}, {LATENT_DIM}), got {teacher.shape}."
        )
    if not np.isfinite(teacher).all():
        raise RuntimeError("Teacher latent contains NaN/Inf values.")
    for table_name, df in [
        ("candidate library", library),
        ("property table", properties),
    ]:
        if "SMILES" not in df.columns:
            raise RuntimeError(f"SMILES missing from {table_name}.")
    library_smiles = library["SMILES"].astype(str).to_numpy()
    property_smiles = properties["SMILES"].astype(str).to_numpy()
    if not np.array_equal(library_smiles, property_smiles):
        mismatch = np.where(library_smiles != property_smiles)[0]
        first = int(mismatch[0])
        raise RuntimeError(
            f"Candidate/property SMILES order mismatch.\nFirst mismatch at row {first}:\nlibrary   = {library_smiles[first]}\nproperties= {property_smiles[first]}"
        )
    print("\n[PASS] Exact 881-molecule candidate/property order verified.")
    PROPERTY_COLUMNS = ["AW_ST_CMC_pred", "pCMC_pred"]
    for column in PROPERTY_COLUMNS:
        if column not in properties.columns:
            raise RuntimeError(
                f"Required saved predicted-property column missing: {column}"
            )
    property_df = properties[PROPERTY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if property_df.isna().any().any():
        bad = property_df.isna().sum()
        raise RuntimeError(f"Missing/non-numeric property predictions:\n{bad}")
    property_matrix = property_df.to_numpy(dtype=np.float32)
    if not np.isfinite(property_matrix).all():
        raise RuntimeError("Property matrix contains NaN/Inf.")
    print("\nSaved predicted-property matrix:", property_matrix.shape)
    print(property_df.describe().T)

    def longest_chain(smiles):
        try:
            runs = re.findall("C+", str(smiles))
            if len(runs) == 0:
                return 0.0
            return float(max((len(x) for x in runs)))
        except Exception:
            return np.nan

    def count_EO(smiles):
        """
        Historical structure-only proxy used in the original pipeline.
        """
        return float(len(re.findall("CCO", str(smiles))))

    def aliphatic_fraction(mol):
        try:
            carbons = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            aliphatic = sum((1 for atom in carbons if not atom.GetIsAromatic()))
            return float(aliphatic / len(carbons))
        except Exception:
            return 0.0

    def ring_count(mol):
        try:
            return float(mol.GetRingInfo().NumRings())
        except Exception:
            return 0.0

    def branching_index(mol):
        try:
            carbons = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            return float(np.mean([atom.GetTotalDegree() for atom in carbons]))
        except Exception:
            return 0.0

    from latent_materials.surfactants.descriptors import detect_head

    STRUCT_NUMERIC = [
        "longest_chain",
        "EO_count",
        "aliphatic_fraction",
        "ring_count",
        "branching_index",
    ]
    HEAD_CATEGORIES = [
        "sulfate",
        "sulfonate",
        "carboxylate",
        "quat_ammonium",
        "ethoxylated",
        "other",
    ]
    structure_records = []
    for row_index, smiles in enumerate(library_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(
                f"Invalid candidate SMILES at row {row_index}:\n{smiles}"
            )
        head = detect_head(smiles)
        record = {
            "longest_chain": longest_chain(smiles),
            "EO_count": count_EO(smiles),
            "aliphatic_fraction": aliphatic_fraction(mol),
            "ring_count": ring_count(mol),
            "branching_index": branching_index(mol),
        }
        for category in HEAD_CATEGORIES:
            record[f"head_{category}"] = float(head == category)
        structure_records.append(record)
    structure_original_df = pd.DataFrame(structure_records)
    ORIGINAL_STRUCTURE_COLUMNS = STRUCT_NUMERIC + [f"head_{x}" for x in HEAD_CATEGORIES]
    structure_original_df = structure_original_df[ORIGINAL_STRUCTURE_COLUMNS]
    head_columns = [f"head_{x}" for x in HEAD_CATEGORIES]
    head_sum = structure_original_df[head_columns].sum(axis=1).to_numpy()
    if not np.allclose(head_sum, 1.0):
        raise RuntimeError("Head-group encoding is not exactly one-hot.")
    structure_original = structure_original_df.to_numpy(dtype=np.float32)
    print("\nOriginal structural descriptor matrix:", structure_original.shape)
    print("\nHead-group counts:")
    head_names_detected = [detect_head(s) for s in library_smiles]
    print(pd.Series(head_names_detected).value_counts())
    for column in ["PO_block_count", "order"]:
        if column not in library.columns:
            raise RuntimeError(f"Required candidate-library column missing: {column}")
    po_count_series = pd.to_numeric(library["PO_block_count"], errors="coerce")
    if po_count_series.isna().any():
        raise RuntimeError("PO_block_count contains missing/non-numeric values.")
    po_count = po_count_series.to_numpy(dtype=np.float32).reshape(-1, 1)
    order_clean = library["order"].astype(str).str.strip()
    order_po_eo = (order_clean == "PO-EO").astype(np.float32).to_numpy().reshape(-1, 1)
    structure_complete = np.concatenate(
        [structure_original, po_count, order_po_eo], axis=1
    )
    COMPLETE_STRUCTURE_COLUMNS = ORIGINAL_STRUCTURE_COLUMNS + [
        "PO_block_count",
        "order_PO_EO",
    ]
    print("\nComplete structural matrix:", structure_complete.shape)
    print("PO block-count range:", float(po_count.min()), "to", float(po_count.max()))
    print("PO-EO count:", int(order_po_eo.sum()))
    MAIN_ABLATIONS = {
        "Property_only": property_matrix,
        "Structure_original": structure_original,
        "Full_original": np.concatenate([structure_original, property_matrix], axis=1),
        "Structure_complete": structure_complete,
        "Full_complete": np.concatenate([structure_complete, property_matrix], axis=1),
    }
    print("\nMain ablation dimensions")
    for name, matrix in MAIN_ABLATIONS.items():
        print(f"{name:25s}: {matrix.shape}")
        if matrix.shape[0] != EXPECTED_N:
            raise RuntimeError(f"Row-count error in {name}")
    all_indices = np.arange(EXPECTED_N)
    dev_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=42, shuffle=True
    )
    fit_idx, val_idx = train_test_split(
        dev_idx, test_size=0.2, random_state=42, shuffle=True
    )
    fit_idx = np.asarray(fit_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    print("\nSplit sizes")
    print("-----------")
    print("Fit :", len(fit_idx))
    print("Val :", len(val_idx))
    print("Test:", len(test_idx))
    if (len(fit_idx), len(val_idx), len(test_idx)) != (563, 141, 177):
        raise RuntimeError("Unexpected split sizes.")
    if len(set(fit_idx) & set(val_idx)):
        raise RuntimeError("Fit/validation overlap detected.")
    if len(set(fit_idx) & set(test_idx)):
        raise RuntimeError("Fit/test overlap detected.")
    if len(set(val_idx) & set(test_idx)):
        raise RuntimeError("Validation/test overlap detected.")

    def preprocess_features(feature_matrix):
        """
        Fit every preprocessing operation exclusively on FIT rows.

        Returns scaled fit/validation/test matrices.
        """
        feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
        fit_raw = feature_matrix[fit_idx].copy()
        val_raw = feature_matrix[val_idx].copy()
        test_raw = feature_matrix[test_idx].copy()
        medians = np.nanmedian(fit_raw, axis=0)
        if np.isnan(medians).any():
            bad_cols = np.where(np.isnan(medians))[0]
            raise RuntimeError(
                f"Feature column(s) entirely missing in fit subset: {bad_cols.tolist()}"
            )

        def apply_median(X):
            X = X.copy()
            rows, cols = np.where(np.isnan(X))
            if len(rows):
                X[rows, cols] = medians[cols]
            return X

        fit_raw = apply_median(fit_raw)
        val_raw = apply_median(val_raw)
        test_raw = apply_median(test_raw)
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(fit_raw)
        X_val = scaler.transform(val_raw)
        X_test = scaler.transform(test_raw)
        return (
            X_fit.astype(np.float32),
            X_val.astype(np.float32),
            X_test.astype(np.float32),
            scaler,
            medians,
        )

    from latent_materials.adapter_model import ConditionalAdapter

    def safe_pearson(y_true, y_pred):
        a = np.asarray(y_true, dtype=np.float64).ravel()
        b = np.asarray(y_pred, dtype=np.float64).ravel()
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    from latent_materials.statistics import latent_metrics

    Y_FIT = teacher[fit_idx].astype(np.float32)
    Y_VAL = teacher[val_idx].astype(np.float32)
    Y_TEST = teacher[test_idx].astype(np.float32)

    def train_one(feature_matrix, seed):
        set_global_seed(seed)
        X_fit, X_val, X_test, _, _ = preprocess_features(feature_matrix)
        x_fit = torch.as_tensor(X_fit, dtype=torch.float32, device=DEVICE)
        y_fit = torch.as_tensor(Y_FIT, dtype=torch.float32, device=DEVICE)
        x_val = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
        y_val = torch.as_tensor(Y_VAL, dtype=torch.float32, device=DEVICE)
        x_test = torch.as_tensor(X_test, dtype=torch.float32, device=DEVICE)
        model = ConditionalAdapter(input_dim=X_fit.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        loss_fn = nn.MSELoss()
        best_validation = np.inf
        best_epoch = None
        best_state = None
        wait = 0
        batch_generator = torch.Generator(device="cpu")
        batch_generator.manual_seed(seed)
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            permutation = torch.randperm(len(X_fit), generator=batch_generator)
            for start in range(0, len(X_fit), BATCH_SIZE):
                batch_indices_cpu = permutation[start : start + BATCH_SIZE]
                batch_indices = batch_indices_cpu.to(DEVICE)
                xb = x_fit[batch_indices]
                yb = y_fit[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb)
                loss = loss_fn(prediction, yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_prediction = model(x_val)
                validation_mse = float(torch.mean((val_prediction - y_val) ** 2).item())
            if validation_mse < best_validation - 1e-10:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                break
        if best_state is None:
            raise RuntimeError("Training failed to create a valid checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_prediction = model(x_test).detach().cpu().numpy().astype(np.float32)
        metrics = latent_metrics(Y_TEST, test_prediction)
        return {
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "validation_mse": float(best_validation),
            "test_prediction": test_prediction,
            **{k: v for k, v in metrics.items() if k != "molecule_rmse"},
        }

    print("\n" + "=" * 100)
    print("RUNNING MAIN REPRESENTATION ABLATIONS")
    print("=" * 100)
    main_rows = []
    prediction_store = {}
    for ablation_name, matrix in MAIN_ABLATIONS.items():
        print(f"\n>>> {ablation_name}")
        prediction_store[ablation_name] = []
        for seed in OPTIMIZATION_SEEDS:
            result = train_one(matrix, seed)
            prediction_store[ablation_name].append(result["test_prediction"])
            main_rows.append(
                {
                    "ablation": ablation_name,
                    "seed": seed,
                    "n_features": matrix.shape[1],
                    "best_epoch": result["best_epoch"],
                    "validation_mse": result["validation_mse"],
                    "test_mse": result["test_mse"],
                    "test_rmse": result["test_rmse"],
                    "pearson_r": result["pearson_r"],
                    "molecule_rmse_mean": result["molecule_rmse_mean"],
                }
            )
            print(
                f"seed={seed:2d} | epoch={result['best_epoch']:4d} | MSE={result['test_mse']:.8f} | RMSE={result['test_rmse']:.6f} | r={result['pearson_r']:.6f} | mol-RMSE={result['molecule_rmse_mean']:.6f}"
            )
    main_results = pd.DataFrame(main_rows)
    main_summary = (
        main_results.groupby("ablation", sort=False)
        .agg(
            n_features=("n_features", "first"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_sd=("test_mse", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_sd=("test_rmse", "std"),
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_sd=("pearson_r", "std"),
            molecule_rmse_mean=("molecule_rmse_mean", "mean"),
            molecule_rmse_sd=("molecule_rmse_mean", "std"),
        )
        .reset_index()
    )
    print("\n" + "=" * 100)
    print("MAIN ABLATION SUMMARY")
    print("=" * 100)
    print(main_summary.to_string(index=False))
    mean_test_predictions = {}
    for name, predictions in prediction_store.items():
        stacked = np.stack(predictions, axis=0)
        mean_test_predictions[name] = np.mean(stacked, axis=0)
    per_molecule_rmse = {}
    for name, prediction in mean_test_predictions.items():
        diff = prediction.astype(np.float64) - Y_TEST.astype(np.float64)
        per_molecule_rmse[name] = np.sqrt(np.mean(diff**2, axis=1))
    PRIMARY_COMPARISONS = [
        ("Full_original vs Structure_original", "Full_original", "Structure_original"),
        ("Full_complete vs Structure_complete", "Full_complete", "Structure_complete"),
        ("Full_complete vs Property_only", "Full_complete", "Property_only"),
        ("Full_complete vs Full_original", "Full_complete", "Full_original"),
    ]

    def paired_bootstrap_improvement(
        better_errors, comparator_errors, n_bootstrap=BOOTSTRAP_REPLICATES, seed=12345
    ):
        """
        Positive improvement means the first model
        has LOWER per-molecule RMSE.
        """
        better_errors = np.asarray(better_errors, dtype=np.float64)
        comparator_errors = np.asarray(comparator_errors, dtype=np.float64)
        paired_improvement = comparator_errors - better_errors
        observed = float(paired_improvement.mean())
        n = len(paired_improvement)
        rng = np.random.default_rng(seed)
        bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bootstrap_means[i] = np.mean(paired_improvement[idx])
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        bootstrap_fraction_nonpositive = float(np.mean(bootstrap_means <= 0))
        return {
            "mean_RMSE_improvement": observed,
            "CI95_low": float(ci_low),
            "CI95_high": float(ci_high),
            "bootstrap_fraction_improvement_le_0": bootstrap_fraction_nonpositive,
        }

    bootstrap_rows = []
    for comparison_index, (label, better, comparator) in enumerate(PRIMARY_COMPARISONS):
        stats = paired_bootstrap_improvement(
            better_errors=per_molecule_rmse[better],
            comparator_errors=per_molecule_rmse[comparator],
            seed=12345 + comparison_index,
        )
        bootstrap_rows.append(
            {
                "comparison": label,
                "better_model": better,
                "comparator": comparator,
                **stats,
            }
        )
    bootstrap_results = pd.DataFrame(bootstrap_rows)
    print("\n" + "=" * 100)
    print("PAIRED MOLECULE BOOTSTRAP")
    print("=" * 100)
    print(bootstrap_results.to_string(index=False))

    def make_partitionwise_shuffled_properties(shuffle_seed):
        rng = np.random.default_rng(shuffle_seed)
        shuffled = property_matrix.copy()
        for indices in [fit_idx, val_idx, test_idx]:
            permuted_indices = rng.permutation(indices)
            shuffled[indices] = property_matrix[permuted_indices]
        return shuffled

    shuffle_rows = []
    shuffle_ensemble_predictions = {}
    print("\n" + "=" * 100)
    print("RUNNING REPEATED SHUFFLED-PROPERTY CONTROLS")
    print("=" * 100)
    for shuffle_id in range(N_PROPERTY_SHUFFLES):
        shuffle_seed = 20260926 + shuffle_id
        shuffled_properties = make_partitionwise_shuffled_properties(shuffle_seed)
        shuffled_matrix = np.concatenate(
            [structure_complete, shuffled_properties], axis=1
        )
        shuffle_predictions = []
        print(f"\nShuffle {shuffle_id + 1}/{N_PROPERTY_SHUFFLES}")
        for seed in OPTIMIZATION_SEEDS:
            result = train_one(shuffled_matrix, seed)
            shuffle_predictions.append(result["test_prediction"])
            shuffle_rows.append(
                {
                    "shuffle_id": shuffle_id,
                    "shuffle_seed": shuffle_seed,
                    "optimization_seed": seed,
                    "test_mse": result["test_mse"],
                    "test_rmse": result["test_rmse"],
                    "pearson_r": result["pearson_r"],
                    "molecule_rmse_mean": result["molecule_rmse_mean"],
                }
            )
        shuffle_ensemble_predictions[shuffle_id] = np.mean(
            np.stack(shuffle_predictions, axis=0), axis=0
        )
    shuffle_results = pd.DataFrame(shuffle_rows)
    real_full_prediction = mean_test_predictions["Full_complete"]
    real_full_metrics = latent_metrics(Y_TEST, real_full_prediction)
    shuffle_summary_rows = []
    for shuffle_id, prediction in shuffle_ensemble_predictions.items():
        metrics = latent_metrics(Y_TEST, prediction)
        shuffle_errors = metrics["molecule_rmse"]
        full_errors = real_full_metrics["molecule_rmse"]
        observed_improvement = float(np.mean(shuffle_errors - full_errors))
        shuffle_summary_rows.append(
            {
                "shuffle_id": shuffle_id,
                "ensemble_test_mse": metrics["test_mse"],
                "ensemble_test_rmse": metrics["test_rmse"],
                "ensemble_pearson_r": metrics["pearson_r"],
                "ensemble_molecule_rmse": metrics["molecule_rmse_mean"],
                "real_full_RMSE_improvement": observed_improvement,
            }
        )
    shuffle_summary = pd.DataFrame(shuffle_summary_rows)
    print("\nRepeated shuffle summary")
    print(shuffle_summary.describe().T)
    real_mse = float(real_full_metrics["test_mse"])
    real_mol_rmse = float(real_full_metrics["molecule_rmse_mean"])
    shuffle_mses = shuffle_summary["ensemble_test_mse"].to_numpy()
    shuffle_mol_rmse = shuffle_summary["ensemble_molecule_rmse"].to_numpy()
    empirical_null_fraction_mse_as_good_or_better = (
        1 + np.sum(shuffle_mses <= real_mse)
    ) / (N_PROPERTY_SHUFFLES + 1)
    empirical_null_fraction_molrmse_as_good_or_better = (
        1 + np.sum(shuffle_mol_rmse <= real_mol_rmse)
    ) / (N_PROPERTY_SHUFFLES + 1)
    shuffle_control_stats = pd.DataFrame(
        [
            {
                "n_property_shuffles": N_PROPERTY_SHUFFLES,
                "real_full_complete_test_mse": real_mse,
                "mean_shuffled_test_mse": float(np.mean(shuffle_mses)),
                "median_shuffled_test_mse": float(np.median(shuffle_mses)),
                "null_fraction_MSE_as_good_or_better_than_real": float(
                    empirical_null_fraction_mse_as_good_or_better
                ),
                "real_full_complete_molecule_rmse": real_mol_rmse,
                "mean_shuffled_molecule_rmse": float(np.mean(shuffle_mol_rmse)),
                "median_shuffled_molecule_rmse": float(np.median(shuffle_mol_rmse)),
                "null_fraction_molRMSE_as_good_or_better_than_real": float(
                    empirical_null_fraction_molrmse_as_good_or_better
                ),
            }
        ]
    )
    print("\n" + "=" * 100)
    print("SHUFFLED-PROPERTY NEGATIVE CONTROL")
    print("=" * 100)
    print(shuffle_control_stats.to_string(index=False))
    summary_indexed = main_summary.set_index("ablation")

    def percent_mse_reduction(improved, reference):
        improved_mse = float(summary_indexed.loc[improved, "test_mse_mean"])
        reference_mse = float(summary_indexed.loc[reference, "test_mse_mean"])
        return float(100.0 * (reference_mse - improved_mse) / reference_mse)

    effect_rows = [
        {
            "comparison": "Full_original vs Structure_original",
            "percent_test_MSE_reduction": percent_mse_reduction(
                "Full_original", "Structure_original"
            ),
        },
        {
            "comparison": "Full_complete vs Structure_complete",
            "percent_test_MSE_reduction": percent_mse_reduction(
                "Full_complete", "Structure_complete"
            ),
        },
        {
            "comparison": "Full_complete vs Property_only",
            "percent_test_MSE_reduction": percent_mse_reduction(
                "Full_complete", "Property_only"
            ),
        },
        {
            "comparison": "Full_complete vs Full_original",
            "percent_test_MSE_reduction": percent_mse_reduction(
                "Full_complete", "Full_original"
            ),
        },
    ]
    effect_summary = pd.DataFrame(effect_rows)
    print("\n" + "=" * 100)
    print("EFFECT SIZES")
    print("=" * 100)
    print(effect_summary.to_string(index=False))
    historical_student_reference = None
    if historical_student is not None and historical_student.shape == teacher.shape:
        hist_metrics = latent_metrics(teacher[test_idx], historical_student[test_idx])
        historical_student_reference = pd.DataFrame(
            [
                {
                    "test_mse": hist_metrics["test_mse"],
                    "test_rmse": hist_metrics["test_rmse"],
                    "pearson_r": hist_metrics["pearson_r"],
                    "molecule_rmse_mean": hist_metrics["molecule_rmse_mean"],
                }
            ]
        )
        print("\nHistorical student (descriptive reference only)")
        print(historical_student_reference.to_string(index=False))
    split_labels = np.full(EXPECTED_N, "", dtype=object)
    split_labels[fit_idx] = "fit"
    split_labels[val_idx] = "validation"
    split_labels[test_idx] = "test"
    split_df = pd.DataFrame(
        {
            "original_index": np.arange(EXPECTED_N),
            "SMILES": library_smiles,
            "split": split_labels,
        }
    )
    feature_manifest = {
        "interpretation": "Representation-level conditioning-information ablation against authenticated saved teacher latent coordinates; not a generator retraining ablation.",
        "n_candidates": EXPECTED_N,
        "latent_dim": LATENT_DIM,
        "predicted_property_columns": PROPERTY_COLUMNS,
        "original_structure_columns": ORIGINAL_STRUCTURE_COLUMNS,
        "complete_structure_columns": COMPLETE_STRUCTURE_COLUMNS,
        "optimization_seeds": OPTIMIZATION_SEEDS,
        "n_property_shuffles": N_PROPERTY_SHUFFLES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "fit_n": len(fit_idx),
        "validation_n": len(val_idx),
        "test_n": len(test_idx),
        "split_random_state": 42,
        "eo_definition": 'len(re.findall(r"CCO", smiles))',
        "properties_are": "saved predicted properties, not experimental labels",
    }
    main_results_path = ABLATION_DIR / "main_ablation_by_seed.csv"
    main_summary_path = ABLATION_DIR / "main_ablation_summary.csv"
    bootstrap_path = ABLATION_DIR / "paired_molecule_bootstrap.csv"
    effect_path = ABLATION_DIR / "effect_sizes.csv"
    shuffle_seed_path = ABLATION_DIR / "shuffled_property_by_seed.csv"
    shuffle_summary_path = ABLATION_DIR / "shuffled_property_summary.csv"
    shuffle_control_stats_path = (
        ABLATION_DIR / "shuffled_property_control_statistics.csv"
    )
    split_path = ABLATION_DIR / "fixed_data_split.csv"
    feature_path = ABLATION_DIR / "candidate_ablation_features.csv"
    manifest_path = ABLATION_DIR / "ablation_manifest.json"
    main_results.to_csv(main_results_path, index=False)
    main_summary.to_csv(main_summary_path, index=False)
    bootstrap_results.to_csv(bootstrap_path, index=False)
    effect_summary.to_csv(effect_path, index=False)
    shuffle_results.to_csv(shuffle_seed_path, index=False)
    shuffle_summary.to_csv(shuffle_summary_path, index=False)
    shuffle_control_stats.to_csv(shuffle_control_stats_path, index=False)
    split_df.to_csv(split_path, index=False)
    candidate_feature_df = pd.DataFrame(
        structure_complete, columns=COMPLETE_STRUCTURE_COLUMNS
    )
    candidate_feature_df.insert(0, "SMILES", library_smiles)
    candidate_feature_df["AW_ST_CMC_pred"] = property_matrix[:, 0]
    candidate_feature_df["pCMC_pred"] = property_matrix[:, 1]
    candidate_feature_df.to_csv(feature_path, index=False)
    with open(manifest_path, "w") as f:
        json.dump(feature_manifest, f, indent=2)
    if historical_student_reference is not None:
        historical_student_reference.to_csv(
            ABLATION_DIR / "historical_student_reference.csv", index=False
        )
    prediction_payload = {"teacher_test": teacher[test_idx], "test_idx": test_idx}
    for name, predictions in prediction_store.items():
        prediction_payload[f"{name}_all_seeds"] = np.stack(predictions, axis=0)
        prediction_payload[f"{name}_seed_mean"] = mean_test_predictions[name]
    for shuffle_id, prediction in shuffle_ensemble_predictions.items():
        prediction_payload[f"shuffle_{shuffle_id:03d}_seed_mean"] = prediction
    np.savez_compressed(
        ABLATION_DIR / "ablation_test_predictions.npz", **prediction_payload
    )
    PLOT_ORDER = [
        "Property_only",
        "Structure_original",
        "Full_original",
        "Structure_complete",
        "Full_complete",
    ]
    DISPLAY_LABELS = [
        "Property\nonly",
        "Structure\n(original)",
        "Full\n(original)",
        "Structure\n(complete)",
        "Full\n(complete)",
    ]
    plot_summary = main_summary.set_index("ablation").loc[PLOT_ORDER].reset_index()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    ax1, ax2 = axes
    x = np.arange(len(PLOT_ORDER))
    ax1.bar(
        x,
        plot_summary["test_mse_mean"],
        yerr=plot_summary["test_mse_sd"],
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(DISPLAY_LABELS)
    ax1.set_ylabel("Untouched-test latent MSE")
    ax1.set_title("a  Conditioning-information ablation", loc="left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax1.set_axisbelow(True)
    effect_plot = bootstrap_results.copy()
    effect_labels = [
        "Properties added\n(original)",
        "Properties added\n(complete)",
        "Full vs\nproperty only",
        "PO information\nadded",
    ]
    means = effect_plot["mean_RMSE_improvement"].to_numpy()
    low = effect_plot["CI95_low"].to_numpy()
    high = effect_plot["CI95_high"].to_numpy()
    y = np.arange(len(effect_plot))
    xerr = np.vstack([means - low, high - means])
    ax2.errorbar(means, y, xerr=xerr, fmt="o", markersize=5, capsize=3, linewidth=1.0)
    ax2.axvline(0, linestyle="--", linewidth=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(effect_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Improvement in per-molecule latent RMSE")
    ax2.set_title("b  Paired test-molecule bootstrap", loc="left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.25)
    ax2.set_axisbelow(True)
    fig.suptitle("Surfactant representation-level conditioning ablation", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.84, wspace=0.38)
    figure_pdf = ABLATION_DIR / "conditioning_ablation_corrected.pdf"
    figure_png = ABLATION_DIR / "conditioning_ablation_corrected.png"
    figure_svg = ABLATION_DIR / "conditioning_ablation_corrected.svg"
    fig.savefig(figure_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(figure_png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(figure_svg, bbox_inches="tight", pad_inches=0.04)
    plt.show()
    print("\n" + "=" * 100)
    print("CORRECTED CONDITIONING ABLATION COMPLETE")
    print("=" * 100)
    print("\nIMPORTANT INTERPRETATION:")
    print(
        "This experiment measures recoverability of the historical teacher latent representation from conditioning variables."
    )
    print(
        "It is NOT a direct CVAE/generation-performance ablation because the original generative model was not retrained under each condition."
    )
    print("\nProperty inputs are saved PREDICTED properties:")
    print(PROPERTY_COLUMNS)
    print("\nMain ablation summary:")
    print(main_summary.to_string(index=False))
    print("\nPaired molecule bootstrap:")
    print(bootstrap_results.to_string(index=False))
    print("\nRepeated shuffled-property control:")
    print(shuffle_control_stats.to_string(index=False))
    print("\nSaved files:")
    for path in [
        main_results_path,
        main_summary_path,
        bootstrap_path,
        effect_path,
        shuffle_seed_path,
        shuffle_summary_path,
        shuffle_control_stats_path,
        split_path,
        feature_path,
        manifest_path,
        ABLATION_DIR / "ablation_test_predictions.npz",
        figure_pdf,
        figure_png,
        figure_svg,
    ]:
        print(" -", path)
    print("\n[PASS] Finished.")
    return root
