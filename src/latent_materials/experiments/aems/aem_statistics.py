"""Final canonical-SMILES conductivity alignment and statistics only; aligned by canonical SMILES.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the aem statistics experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from scipy.stats import mannwhitneyu

    ROOT = Path(str(root / "IEMS_GEN_files"))
    DATA = ROOT / "AEM" / "CVAE" / "data"
    COMP = DATA / "nn_bo_comparison"
    METRICS_FILE = COMP / "AEM_NN_BO_candidate_metrics.csv"
    NN_FILE = DATA / "aem_augmented_ranked_candidates_NN.csv"
    BO_FILE = DATA / "aem_augmented_ranked_candidates_BO.csv"
    OUT = ROOT / "reviewer_defense_AEM_NN_vs_BO"
    OUT.mkdir(exist_ok=True)
    SEED = 20260911
    N_BOOT = 10000
    N_PERM = 10000
    rng = np.random.default_rng(SEED)

    def canon(s):
        try:
            m = Chem.MolFromSmiles(str(s))
            return Chem.MolToSmiles(m, canonical=True) if m else None
        except:
            return None

    metrics = pd.read_csv(METRICS_FILE)
    nn = pd.read_csv(NN_FILE)
    bo = pd.read_csv(BO_FILE)
    metrics["branch"] = metrics["branch"].astype(str).str.upper().str.strip()
    if "canonical_SMILES" not in metrics.columns:
        metrics["canonical_SMILES"] = metrics["SMILES"].map(canon)
    else:
        metrics["canonical_SMILES"] = metrics["canonical_SMILES"].map(canon)
    nn["canonical_SMILES"] = nn["SMILES"].map(canon)
    bo["canonical_SMILES"] = bo["SMILES"].map(canon)
    nn = nn.dropna(subset=["canonical_SMILES"]).drop_duplicates("canonical_SMILES")
    bo = bo.dropna(subset=["canonical_SMILES"]).drop_duplicates("canonical_SMILES")
    metrics = (
        metrics.dropna(subset=["canonical_SMILES"])
        .drop_duplicates(["branch", "canonical_SMILES"])
        .copy()
    )
    print("=" * 100)
    print("SOURCE POPULATIONS")
    print("=" * 100)
    print("NN saved population:", len(nn))
    print("BO saved population:", len(bo))
    print("\nRetained comparison:")
    print(metrics["branch"].value_counts())
    nn_map = dict(zip(nn["canonical_SMILES"], nn["Predicted_conductivity"]))
    bo_map = dict(zip(bo["canonical_SMILES"], bo["Predicted_conductivity"]))

    def recover_cond(row):
        smi = row["canonical_SMILES"]
        if row["branch"] == "NN":
            return nn_map.get(smi, np.nan)
        if row["branch"] == "BO":
            return bo_map.get(smi, np.nan)
        return np.nan

    metrics["Predicted_conductivity_recovered"] = metrics.apply(recover_cond, axis=1)
    print("\n" + "=" * 100)
    print("ALIGNMENT RESULTS")
    print("=" * 100)
    for branch in ["NN", "BO"]:
        sub = metrics[metrics["branch"] == branch]
        n_total = len(sub)
        n_found = sub["Predicted_conductivity_recovered"].notna().sum()
        print(
            f"{branch}: recovered {n_found}/{n_total} ({100 * n_found / n_total:.1f}%)"
        )
    missing = metrics[metrics["Predicted_conductivity_recovered"].isna()][
        ["branch", "SMILES", "canonical_SMILES"]
    ]
    if len(missing):
        print("\nMissing alignments:")
        print(missing.head(20).to_string(index=False))
    else:
        print(
            "\n✅ All retained NN/BO molecules matched their source prediction files."
        )
    x_nn = (
        metrics.loc[metrics["branch"] == "NN", "Predicted_conductivity_recovered"]
        .dropna()
        .astype(float)
        .values
    )
    x_bo = (
        metrics.loc[metrics["branch"] == "BO", "Predicted_conductivity_recovered"]
        .dropna()
        .astype(float)
        .values
    )
    print("\n" + "=" * 100)
    print("RECOVERED CONDUCTIVITY COMPARISON")
    print("=" * 100)
    print("NN n =", len(x_nn))
    print("BO n =", len(x_bo))
    if len(x_nn) >= 2 and len(x_bo) >= 2:
        observed = x_bo.mean() - x_nn.mean()
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            boot[i] = (
                rng.choice(x_bo, len(x_bo), replace=True).mean()
                - rng.choice(x_nn, len(x_nn), replace=True).mean()
            )
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        pooled = np.concatenate([x_bo, x_nn])
        n_bo = len(x_bo)
        extreme = 0
        for i in range(N_PERM):
            z = rng.permutation(pooled)
            d = z[:n_bo].mean() - z[n_bo:].mean()
            if abs(d) >= abs(observed):
                extreme += 1
        p_perm = (extreme + 1) / (N_PERM + 1)
        U, p_mw = mannwhitneyu(x_bo, x_nn, alternative="two-sided")
        cliff = 2 * U / (len(x_bo) * len(x_nn)) - 1
        print(f"NN mean       = {x_nn.mean():.8f}")
        print(f"BO mean       = {x_bo.mean():.8f}")
        print(f"NN median     = {np.median(x_nn):.8f}")
        print(f"BO median     = {np.median(x_bo):.8f}")
        print(f"BO - NN       = {observed:.8f}")
        print(f"Bootstrap 95% = [{ci_lo:.8f}, {ci_hi:.8f}]")
        print(f"Permutation p = {p_perm:.8g}")
        print(f"Mann-Whitney p= {p_mw:.8g}")
        print(f"Cliff's delta = {cliff:.6f}")
        result = pd.DataFrame(
            [
                {
                    "NN_n": len(x_nn),
                    "BO_n": len(x_bo),
                    "NN_mean": x_nn.mean(),
                    "BO_mean": x_bo.mean(),
                    "NN_median": np.median(x_nn),
                    "BO_median": np.median(x_bo),
                    "BO_minus_NN_mean": observed,
                    "bootstrap_95CI_low": ci_lo,
                    "bootstrap_95CI_high": ci_hi,
                    "permutation_p": p_perm,
                    "mannwhitney_p": p_mw,
                    "cliffs_delta_BO_vs_NN": cliff,
                }
            ]
        )
        result.to_csv(
            OUT / "AEM_NN_BO_recovered_conductivity_statistics.csv", index=False
        )
    else:
        print(
            "❌ Insufficient aligned conductivity values; do not perform a conductivity significance test."
        )
    repaired = OUT / "AEM_NN_BO_candidate_metrics_with_recovered_conductivity.csv"
    metrics.to_csv(repaired, index=False)
    print("\nSaved repaired table:")
    print(repaired)
    return root
