"""Matched AEM NN/BO selection and six-panel comparison recovered from the supplied HGARE notebook, cells48–53. Uses frozen inputs, not retrained surrogate predictions.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the figure 4 aem experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from rdkit import Chem, DataStructs

    def canon_smiles(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        try:
            Chem.SanitizeMol(m)
        except Exception:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    import re
    from scipy.spatial import ConvexHull, Delaunay
    from matplotlib.patches import Polygon
    from rdkit.Chem import AllChem, Descriptors
    from sklearn.decomposition import PCA

    NNBO_FIG_DIR = os.path.join(DATA_DIR, "nn_bo_comparison")
    os.makedirs(NNBO_FIG_DIR, exist_ok=True)
    NN_PATH = os.path.join(DATA_DIR, "aem_augmented_ranked_candidates_NN.csv")
    BO_PATH = os.path.join(DATA_DIR, "aem_novel_candidates_clean_BO.csv")
    TRAIN80_PATH = os.path.join(DATA_DIR, "aem_train80.csv")
    TEST20_PATH = os.path.join(DATA_DIR, "aem_test20.csv")
    COND_COL80 = "Conductivity"
    nn_raw = pd.read_csv(NN_PATH).dropna(subset=["SMILES"]).copy()
    bo_raw = pd.read_csv(BO_PATH).dropna(subset=["SMILES"]).copy()
    bo_train_df = pd.read_csv(TRAIN80_PATH).dropna(subset=["SMILES"]).copy()
    bo_test_df = pd.read_csv(TEST20_PATH).dropna(subset=["SMILES"]).copy()
    nn_raw["branch"] = "NN"
    bo_raw["branch"] = "BO"
    print("Loaded:")
    print(f"NN cleaned candidates: {len(nn_raw)}")
    print(f"BO cleaned candidates: {len(bo_raw)}")
    print(f"Train molecules (80-split): {len(bo_train_df)}")
    print(f"Test molecules (20-split): {len(bo_test_df)}")

    def longest_chain(smiles):
        runs = re.findall("C+", str(smiles))
        return max((len(c) for c in runs), default=0)

    def ring_count(mol):
        try:
            return mol.GetRingInfo().NumRings()
        except Exception:
            return np.nan

    def branching_index(mol):
        try:
            carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            return float(np.mean([a.GetTotalDegree() for a in carbons]))
        except Exception:
            return np.nan

    def aliphatic_fraction_mol(mol):
        try:
            carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            alip = sum((1 for a in carbons if not a.GetIsAromatic()))
            return alip / len(carbons)
        except Exception:
            return np.nan

    def aromatic_fraction_mol(mol):
        try:
            atoms = list(mol.GetAtoms())
            if not atoms:
                return np.nan
            return sum((1 for a in atoms if a.GetIsAromatic())) / len(atoms)
        except Exception:
            return np.nan

    QUAT_N_SMARTS = Chem.MolFromSmarts("[N+](C)(C)(C)C")
    IMIDAZ_SMARTS = Chem.MolFromSmarts("c1[n+](C)cnc1")
    PHOS_SMARTS = Chem.MolFromSmarts("[P+](C)(C)(C)C")

    def detect_head_aem(mol):
        if mol is None:
            return "invalid"
        if mol.HasSubstructMatch(IMIDAZ_SMARTS):
            return "imidazolium"
        if mol.HasSubstructMatch(PHOS_SMARTS):
            return "phosphonium"
        if mol.HasSubstructMatch(QUAT_N_SMARTS):
            return "quat_ammonium"
        return "other"

    def sa_score_simple(mol):
        """Same simplified heuristic used in the surfactant paper's post-processing
        comparison, kept identical here for cross-section methodological consistency."""
        rings = mol.GetRingInfo().NumRings()
        chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        mw = Descriptors.MolWt(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        return rings * 0.5 + chiral * 1.0 + mw / 100.0 * 0.3 + rot_bonds / 10.0 * 0.2

    def annotate_candidates(cand_df):
        rows = []
        for smi in cand_df["SMILES"]:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                rows.append(
                    {
                        "canonical_SMILES": None,
                        "valid": False,
                        "MW": np.nan,
                        "LogP": np.nan,
                        "TPSA": np.nan,
                        "HBD": np.nan,
                        "HBA": np.nan,
                        "RotBonds": np.nan,
                        "longest_chain": np.nan,
                        "ring_count": np.nan,
                        "branching_index": np.nan,
                        "aliphatic_fraction": np.nan,
                        "aromatic_fraction": np.nan,
                        "head_type": "invalid",
                        "SA_score": np.nan,
                    }
                )
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            rows.append(
                {
                    "canonical_SMILES": Chem.MolToSmiles(mol, canonical=True),
                    "valid": True,
                    "MW": Descriptors.MolWt(mol),
                    "LogP": Descriptors.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                    "HBD": Descriptors.NumHDonors(mol),
                    "HBA": Descriptors.NumHAcceptors(mol),
                    "RotBonds": Descriptors.NumRotatableBonds(mol),
                    "longest_chain": longest_chain(smi),
                    "ring_count": ring_count(mol),
                    "branching_index": branching_index(mol),
                    "aliphatic_fraction": aliphatic_fraction_mol(mol),
                    "aromatic_fraction": aromatic_fraction_mol(mol),
                    "head_type": detect_head_aem(mol),
                    "SA_score": sa_score_simple(mol),
                }
            )
        ann = pd.DataFrame(rows)
        out = pd.concat([cand_df.reset_index(drop=True), ann], axis=1)
        out = (
            out[out["valid"]].drop_duplicates("canonical_SMILES").reset_index(drop=True)
        )
        return out

    bo_train_ann = annotate_candidates(bo_train_df[["SMILES", COND_COL80]])
    hc_quantile80 = bo_train_ann[COND_COL80].quantile(0.75)
    hc_subset80 = bo_train_ann[bo_train_ann[COND_COL80] >= hc_quantile80]
    print(
        f"High-conductivity training subset for ideal targets: {len(hc_subset80)}/{len(bo_train_ann)} molecules (>= {hc_quantile80:.4f} S/cm)"
    )
    AEM_SCORE_IDEAL = {
        "longest_chain": hc_subset80["longest_chain"].median(),
        "aromatic_fraction": hc_subset80["aromatic_fraction"].median(),
        "LogP": hc_subset80["LogP"].median(),
        "MW": hc_subset80["MW"].median(),
    }
    AEM_SCORE_SCALE = {
        "longest_chain": max(1.0, 1.5 * bo_train_ann["longest_chain"].std()),
        "aromatic_fraction": max(0.05, 1.5 * bo_train_ann["aromatic_fraction"].std()),
        "LogP": max(0.5, 1.5 * bo_train_ann["LogP"].std()),
        "MW": max(50.0, 1.5 * bo_train_ann["MW"].std()),
    }
    print(
        "\nData-driven composite-score targets (from top-quartile-conductivity training molecules):"
    )
    for k, v in AEM_SCORE_IDEAL.items():
        print(f"  {k}: ideal={v:.3f}  scale={AEM_SCORE_SCALE[k]:.3f}")

    def composite_score_aem(row):
        """AEM-quality score: same 5-term weighted structure as the surfactant
        paper's composite_score (chain 30% / secondary-descriptor 20% / SA 20% /
        LogP 15% / MW 15%), with targets grounded in this dataset instead of
        surfactant HLB intuition."""
        score = 0.0
        chain_score = (
            1.0
            - abs(row["longest_chain"] - AEM_SCORE_IDEAL["longest_chain"])
            / AEM_SCORE_SCALE["longest_chain"]
        )
        score += max(0, chain_score) * 0.3
        arom_score = (
            1.0
            - abs(row["aromatic_fraction"] - AEM_SCORE_IDEAL["aromatic_fraction"])
            / AEM_SCORE_SCALE["aromatic_fraction"]
        )
        score += max(0, arom_score) * 0.2
        sa_term = 1.0 - row["SA_score"] / 10.0
        score += max(0, sa_term) * 0.2
        logp_score = (
            1.0 - abs(row["LogP"] - AEM_SCORE_IDEAL["LogP"]) / AEM_SCORE_SCALE["LogP"]
        )
        score += max(0, logp_score) * 0.15
        mw_score = 1.0 - abs(row["MW"] - AEM_SCORE_IDEAL["MW"]) / AEM_SCORE_SCALE["MW"]
        score += max(0, mw_score) * 0.15
        return score

    NNBO_FP_RADIUS, NNBO_FP_BITS = (2, 2048)
    nnbo_mgen = AllChem.GetMorganGenerator(radius=NNBO_FP_RADIUS, fpSize=NNBO_FP_BITS)

    def nnbo_smiles_to_fp(smi):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return nnbo_mgen.GetFingerprint(mol)

    def nnbo_fp_to_np(fp):
        arr = np.zeros((NNBO_FP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def nnbo_max_tanimoto_to_reference(query_smi_list, ref_fps):
        out = []
        for smi in query_smi_list:
            fp = nnbo_smiles_to_fp(smi)
            if fp is None or len(ref_fps) == 0:
                out.append(np.nan)
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            out.append(float(max(sims)) if sims else np.nan)
        return np.array(out, dtype=float)

    def nnbo_novelty_category(sim):
        if pd.isna(sim):
            return "Invalid"
        if sim >= 0.85:
            return "Very similar"
        if sim >= 0.7:
            return "Moderately similar"
        if sim >= 0.5:
            return "Moderately novel"
        return "Highly novel"

    def nnbo_diversity_filter(cand_df, threshold=0.8, max_keep=500):
        fps, valid_indices = ([], [])
        for idx, smi in enumerate(cand_df["canonical_SMILES"]):
            fp = nnbo_smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
                valid_indices.append(idx)
        if not fps:
            return cand_df.iloc[:0].copy()
        keep_indices, keep_fps = ([valid_indices[0]], [fps[0]])
        for i in range(1, len(fps)):
            max_sim = max(
                (DataStructs.TanimotoSimilarity(fps[i], kfp) for kfp in keep_fps)
            )
            if max_sim < threshold:
                keep_indices.append(valid_indices[i])
                keep_fps.append(fps[i])
                if len(keep_indices) >= max_keep:
                    break
        return cand_df.iloc[keep_indices].copy().reset_index(drop=True)

    def nnbo_fps_for_smiles(smiles_list):
        arrs = [
            nnbo_fp_to_np(f)
            for f in (nnbo_smiles_to_fp(s) for s in smiles_list)
            if f is not None
        ]
        return np.asarray(arrs) if arrs else np.empty((0, NNBO_FP_BITS), dtype=np.int8)

    def nnbo_safe_bins(values, n_bins=28, pad_fraction=0.03):
        values = pd.Series(values).dropna().astype(float)
        if len(values) == 0:
            return np.linspace(0, 1, n_bins)
        vmin, vmax = (values.min(), values.max())
        if np.isclose(vmin, vmax):
            pad = abs(vmin) * pad_fraction + 0.001
            return np.linspace(vmin - pad, vmax + pad, n_bins)
        pad = (vmax - vmin) * pad_fraction
        return np.linspace(vmin - pad, vmax + pad, n_bins)

    nn_ann = annotate_candidates(nn_raw)
    bo_ann = annotate_candidates(bo_raw)
    nn_ann["composite_score"] = nn_ann.apply(composite_score_aem, axis=1)
    bo_ann["composite_score"] = bo_ann.apply(composite_score_aem, axis=1)
    SA_THRESHOLD = 8.0
    DIV_THRESHOLD = 0.8
    MAX_KEEP = 500
    nn_sa = nn_ann[nn_ann["SA_score"] < SA_THRESHOLD].copy()
    bo_sa = bo_ann[bo_ann["SA_score"] < SA_THRESHOLD].copy()
    nn_div = nnbo_diversity_filter(nn_sa, threshold=DIV_THRESHOLD, max_keep=MAX_KEEP)
    bo_div = nnbo_diversity_filter(bo_sa, threshold=DIV_THRESHOLD, max_keep=MAX_KEEP)
    print("Post-processing counts:")
    print(
        f"NN cleaned: {len(nn_ann)} | after SA: {len(nn_sa)} | after diversity: {len(nn_div)}"
    )
    print(
        f"BO cleaned: {len(bo_ann)} | after SA: {len(bo_sa)} | after diversity: {len(bo_div)}"
    )
    bo_train_smiles = (
        bo_train_df["SMILES"]
        .dropna()
        .map(canon_smiles)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    bo_train_fps = [
        f for f in (nnbo_smiles_to_fp(s) for s in bo_train_smiles) if f is not None
    ]
    print(f"\nUnique training reference molecules for novelty: {len(bo_train_fps)}")
    nn_div["max_tanimoto_to_train"] = nnbo_max_tanimoto_to_reference(
        nn_div["canonical_SMILES"], bo_train_fps
    )
    bo_div["max_tanimoto_to_train"] = nnbo_max_tanimoto_to_reference(
        bo_div["canonical_SMILES"], bo_train_fps
    )
    nn_div["novelty_category"] = nn_div["max_tanimoto_to_train"].apply(
        nnbo_novelty_category
    )
    bo_div["novelty_category"] = bo_div["max_tanimoto_to_train"].apply(
        nnbo_novelty_category
    )
    candidate_metrics = pd.concat([nn_div, bo_div], ignore_index=True)
    candidate_metrics_path = os.path.join(
        NNBO_FIG_DIR, "AEM_NN_BO_candidate_metrics.csv"
    )
    candidate_metrics.to_csv(candidate_metrics_path, index=False)
    print(f"Saved candidate metrics: {candidate_metrics_path}")

    def frac_moderately_or_highly_novel(cand_df):
        if len(cand_df) == 0:
            return np.nan
        return (
            100.0
            * cand_df["novelty_category"]
            .isin(["Moderately novel", "Highly novel"])
            .mean()
        )

    def high_score_fraction(cand_df, threshold):
        if len(cand_df) == 0:
            return np.nan
        return 100.0 * (cand_df["composite_score"] >= threshold).mean()

    HIGH_SCORE_THRESHOLD = candidate_metrics["composite_score"].quantile(0.75)
    nnbo_summary_rows = []
    for label, full_df, sa_df, div_df in [
        ("NN", nn_ann, nn_sa, nn_div),
        ("BO", bo_ann, bo_sa, bo_div),
    ]:
        nnbo_summary_rows.append(
            {
                "branch": label,
                "cleaned_candidates": len(full_df),
                "after_SA_filter": len(sa_df),
                "after_diversity_filter": len(div_df),
                "SA_retention_percent": (
                    100.0 * len(sa_df) / len(full_df) if len(full_df) else np.nan
                ),
                "diversity_retention_percent_of_cleaned": (
                    100.0 * len(div_df) / len(full_df) if len(full_df) else np.nan
                ),
                "mean_max_tanimoto_to_train": div_df["max_tanimoto_to_train"].mean(),
                "std_max_tanimoto_to_train": div_df["max_tanimoto_to_train"].std(),
                "median_max_tanimoto_to_train": div_df[
                    "max_tanimoto_to_train"
                ].median(),
                "moderately_or_highly_novel_percent": frac_moderately_or_highly_novel(
                    div_df
                ),
                "highly_novel_percent": (
                    100.0 * (div_df["novelty_category"] == "Highly novel").mean()
                    if len(div_df)
                    else np.nan
                ),
                "mean_composite_score": div_df["composite_score"].mean(),
                "std_composite_score": div_df["composite_score"].std(),
                "median_composite_score": div_df["composite_score"].median(),
                "high_score_fraction_percent": high_score_fraction(
                    div_df, HIGH_SCORE_THRESHOLD
                ),
                "mean_SA_score": div_df["SA_score"].mean(),
                "std_SA_score": div_df["SA_score"].std(),
                "median_SA_score": div_df["SA_score"].median(),
                "mean_MW": div_df["MW"].mean(),
                "std_MW": div_df["MW"].std(),
                "mean_LogP": div_df["LogP"].mean(),
                "std_LogP": div_df["LogP"].std(),
                "mean_longest_chain": div_df["longest_chain"].mean(),
                "std_longest_chain": div_df["longest_chain"].std(),
                "mean_aromatic_fraction": div_df["aromatic_fraction"].mean(),
                "std_aromatic_fraction": div_df["aromatic_fraction"].std(),
            }
        )
    nnbo_summary = pd.DataFrame(nnbo_summary_rows)
    nnbo_summary_path = os.path.join(
        NNBO_FIG_DIR, "AEM_NN_BO_postprocessing_summary_table.csv"
    )
    nnbo_summary.to_csv(nnbo_summary_path, index=False)
    print(f"\nSaved summary table: {nnbo_summary_path}")
    print(nnbo_summary.to_string(index=False))
    cat_order = [
        "Very similar",
        "Moderately similar",
        "Moderately novel",
        "Highly novel",
    ]
    nn_cat = (
        nn_div["novelty_category"]
        .value_counts(normalize=True)
        .reindex(cat_order)
        .fillna(0.0)
        * 100.0
    )
    bo_cat = (
        bo_div["novelty_category"]
        .value_counts(normalize=True)
        .reindex(cat_order)
        .fillna(0.0)
        * 100.0
    )
    novelty_category_df = pd.DataFrame(
        {
            "category": cat_order,
            "NN_percent": nn_cat.values,
            "BO_percent": bo_cat.values,
        }
    )
    novelty_category_path = os.path.join(
        NNBO_FIG_DIR, "AEM_NN_BO_novelty_category_percentages.csv"
    )
    novelty_category_df.to_csv(novelty_category_path, index=False)
    print(f"\nSaved novelty category percentages: {novelty_category_path}")
    print(novelty_category_df.to_string(index=False))
    bo_train_canon = pd.Series(
        bo_train_df["SMILES"].dropna().map(canon_smiles).dropna().unique()
    ).tolist()
    bo_test_canon = pd.Series(
        bo_test_df["SMILES"].dropna().map(canon_smiles).dropna().unique()
    ).tolist()
    nn_plot_smiles = nn_div["canonical_SMILES"].tolist()
    bo_plot_smiles = bo_div["canonical_SMILES"].tolist()
    fps_train80 = nnbo_fps_for_smiles(bo_train_canon)
    fps_test20 = nnbo_fps_for_smiles(bo_test_canon)
    fps_nn = nnbo_fps_for_smiles(nn_plot_smiles)
    fps_bo = nnbo_fps_for_smiles(bo_plot_smiles)
    fps_all = np.vstack([fps_train80, fps_test20, fps_nn, fps_bo])
    labels_all = (
        ["Train"] * len(fps_train80)
        + ["Test"] * len(fps_test20)
        + ["NN"] * len(fps_nn)
        + ["BO"] * len(fps_bo)
    )
    nnbo_pca = PCA(n_components=2, random_state=123)
    coords_all = nnbo_pca.fit_transform(fps_all)
    pca_df = pd.DataFrame(
        {"PC1": coords_all[:, 0], "PC2": coords_all[:, 1], "set": labels_all}
    )
    pca_path = os.path.join(NNBO_FIG_DIR, "AEM_NN_BO_joint_PCA_coordinates.csv")
    pca_df.to_csv(pca_path, index=False)
    print(f"Saved PCA coordinates: {pca_path}")

    def get_points(pdf, label):
        return pdf[pdf["set"] == label][["PC1", "PC2"]].values

    def safe_hull(points):
        if points is None or len(points) < 3:
            return None
        try:
            return ConvexHull(points)
        except Exception:
            return None

    def points_inside_hull(points, hull_ref_points):
        if len(hull_ref_points) < 3 or len(points) == 0:
            return np.zeros(len(points), dtype=bool)
        try:
            tri = Delaunay(hull_ref_points)
            return tri.find_simplex(points) >= 0
        except Exception:
            return np.zeros(len(points), dtype=bool)

    pts_train80, pts_test20, pts_nn, pts_bo = (
        get_points(pca_df, l) for l in ("Train", "Test", "NN", "BO")
    )
    hull_train80, hull_test20, hull_nn, hull_bo = (
        safe_hull(p) for p in (pts_train80, pts_test20, pts_nn, pts_bo)
    )
    train80_hull_ref = (
        pts_train80[hull_train80.vertices] if hull_train80 is not None else pts_train80
    )
    test20_inside_train = points_inside_hull(pts_test20, train80_hull_ref)
    nn_inside_train = points_inside_hull(pts_nn, train80_hull_ref)
    bo_inside_train = points_inside_hull(pts_bo, train80_hull_ref)
    coverage_summary = {
        "test_inside_train_hull_percent": (
            100.0 * np.mean(test20_inside_train) if len(test20_inside_train) else np.nan
        ),
        "NN_inside_train_hull_percent": (
            100.0 * np.mean(nn_inside_train) if len(nn_inside_train) else np.nan
        ),
        "BO_inside_train_hull_percent": (
            100.0 * np.mean(bo_inside_train) if len(bo_inside_train) else np.nan
        ),
        "train_hull_area": (
            float(hull_train80.volume) if hull_train80 is not None else np.nan
        ),
        "test_hull_area": (
            float(hull_test20.volume) if hull_test20 is not None else np.nan
        ),
        "NN_hull_area": float(hull_nn.volume) if hull_nn is not None else np.nan,
        "BO_hull_area": float(hull_bo.volume) if hull_bo is not None else np.nan,
    }
    coverage_df = pd.DataFrame([coverage_summary])
    coverage_path = os.path.join(
        NNBO_FIG_DIR, "AEM_NN_BO_convex_hull_coverage_summary.csv"
    )
    coverage_df.to_csv(coverage_path, index=False)
    print("Saved convex-hull coverage summary:", coverage_path)
    print(coverage_df.to_string(index=False))
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.linewidth": 1.15,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.fontsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    NNBO_COLORS = {
        "NN": "#0077BB",
        "BO": "#CC3311",
        "Train": "#6C757D",
        "Test": "#009988",
        "HullTrain": "#ADB5BD",
        "HighScore": "#332288",
        "Cutoff": "#111111",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.2))
    axes = axes.ravel()
    fig.suptitle(
        "Post-generation comparison of NN- and BO-generated AEM candidates",
        fontsize=16.5,
        fontweight="semibold",
        y=0.995,
    )
    tan_bins = np.linspace(0, 1, 31)
    axes[0].hist(
        nn_div["max_tanimoto_to_train"].dropna(),
        bins=tan_bins,
        alpha=0.42,
        density=True,
        color=NNBO_COLORS["NN"],
        edgecolor=NNBO_COLORS["NN"],
        linewidth=1.0,
        label="NN",
    )
    axes[0].hist(
        bo_div["max_tanimoto_to_train"].dropna(),
        bins=tan_bins,
        alpha=0.42,
        density=True,
        color=NNBO_COLORS["BO"],
        edgecolor=NNBO_COLORS["BO"],
        linewidth=1.0,
        label="BO",
    )
    axes[0].axvline(
        nn_div["max_tanimoto_to_train"].median(),
        color=NNBO_COLORS["NN"],
        linestyle="--",
        linewidth=1.6,
    )
    axes[0].axvline(
        bo_div["max_tanimoto_to_train"].median(),
        color=NNBO_COLORS["BO"],
        linestyle="--",
        linewidth=1.6,
    )
    axes[0].set_xlabel("Maximum Tanimoto similarity to training")
    axes[0].set_ylabel("Density")
    axes[0].set_title("A. Novelty relative to training")
    axes[0].legend(frameon=False)
    x = np.arange(len(cat_order))
    w = 0.38
    axes[1].bar(
        x - w / 2,
        nn_cat.values,
        width=w,
        color=NNBO_COLORS["NN"],
        alpha=0.9,
        label="NN",
    )
    axes[1].bar(
        x + w / 2,
        bo_cat.values,
        width=w,
        color=NNBO_COLORS["BO"],
        alpha=0.9,
        label="BO",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["Very\nsimilar", "Moderately\nsimilar", "Moderately\nnovel", "Highly\nnovel"]
    )
    axes[1].set_ylabel("Candidates (%)")
    axes[1].set_title("B. Novelty-category distribution")
    axes[1].legend(frameon=False)
    stages = ["After SA", "After diversity"]
    nn_counts = [len(nn_sa), len(nn_div)]
    bo_counts = [len(bo_sa), len(bo_div)]
    x2 = np.arange(len(stages))
    axes[2].bar(
        x2 - w / 2, nn_counts, width=w, color=NNBO_COLORS["NN"], alpha=0.9, label="NN"
    )
    axes[2].bar(
        x2 + w / 2, bo_counts, width=w, color=NNBO_COLORS["BO"], alpha=0.9, label="BO"
    )
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(stages)
    axes[2].set_ylabel("Candidate count")
    axes[2].set_title("C. Post-screening retention")
    axes[2].legend(frameon=False)
    count_offset = max(bo_counts + nn_counts) * 0.025
    for i, val in enumerate(nn_counts):
        axes[2].text(
            i - w / 2,
            val + count_offset,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for i, val in enumerate(bo_counts):
        axes[2].text(
            i + w / 2,
            val + count_offset,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    score_bins = nnbo_safe_bins(candidate_metrics["composite_score"], n_bins=28)
    axes[3].hist(
        nn_div["composite_score"].dropna(),
        bins=score_bins,
        alpha=0.42,
        density=True,
        color=NNBO_COLORS["NN"],
        edgecolor=NNBO_COLORS["NN"],
        linewidth=1.0,
        label="NN",
    )
    axes[3].hist(
        bo_div["composite_score"].dropna(),
        bins=score_bins,
        alpha=0.42,
        density=True,
        color=NNBO_COLORS["BO"],
        edgecolor=NNBO_COLORS["BO"],
        linewidth=1.0,
        label="BO",
    )
    axes[3].axvline(
        HIGH_SCORE_THRESHOLD,
        color=NNBO_COLORS["HighScore"],
        linestyle="--",
        linewidth=1.5,
        label="Upper-quartile threshold",
    )
    axes[3].set_xlabel("Composite AEM-quality score")
    axes[3].set_ylabel("Density")
    axes[3].set_title("D. AEM-quality-score distribution")
    axes[3].legend(frameon=False)
    sa_bins = nnbo_safe_bins(candidate_metrics["SA_score"], n_bins=28)
    axes[4].hist(
        nn_div["SA_score"].dropna(),
        bins=sa_bins,
        alpha=0.42,
        density=True,
        color=NNBO_COLORS["NN"],
        edgecolor=NNBO_COLORS["NN"],
        linewidth=1.0,
        label="NN",
    )
    axes[4].hist(
        bo_div["SA_score"].dropna(),
        bins=sa_bins,
        alpha=0.42,
        density=True,
        color=NNBO_COLORS["BO"],
        edgecolor=NNBO_COLORS["BO"],
        linewidth=1.0,
        label="BO",
    )
    axes[4].axvline(
        SA_THRESHOLD,
        color=NNBO_COLORS["Cutoff"],
        linestyle="--",
        linewidth=1.3,
        label="SA cutoff",
    )
    axes[4].set_xlabel("SA score")
    axes[4].set_ylabel("Density")
    axes[4].set_title("E. Synthetic-accessibility score")
    axes[4].legend(frameon=False)
    ax = axes[5]
    if hull_train80 is not None:
        ax.add_patch(
            Polygon(
                pts_train80[hull_train80.vertices],
                closed=True,
                facecolor=NNBO_COLORS["HullTrain"],
                edgecolor=NNBO_COLORS["Train"],
                linewidth=1.3,
                alpha=0.2,
                label="Train hull",
                zorder=0,
            )
        )
    if hull_nn is not None:
        ax.add_patch(
            Polygon(
                pts_nn[hull_nn.vertices],
                closed=True,
                fill=False,
                edgecolor=NNBO_COLORS["NN"],
                linewidth=1.9,
                label="NN hull",
                zorder=5,
            )
        )
    if hull_bo is not None:
        ax.add_patch(
            Polygon(
                pts_bo[hull_bo.vertices],
                closed=True,
                fill=False,
                edgecolor=NNBO_COLORS["BO"],
                linewidth=1.9,
                label="BO hull",
                zorder=5,
            )
        )
    for label, alpha, size, color, zorder in [
        ("Train", 0.18, 10, NNBO_COLORS["Train"], 1),
        ("Test", 0.38, 14, NNBO_COLORS["Test"], 2),
        ("NN", 0.88, 36, NNBO_COLORS["NN"], 4),
        ("BO", 0.8, 30, NNBO_COLORS["BO"], 3),
    ]:
        temp = pca_df[pca_df["set"] == label]
        ax.scatter(
            temp["PC1"],
            temp["PC2"],
            s=size,
            alpha=alpha,
            color=color,
            label=label,
            linewidths=0,
            zorder=zorder,
        )
    coverage_text = f"Inside train hull\nNN: {coverage_summary['NN_inside_train_hull_percent']:.1f}%\nBO: {coverage_summary['BO_inside_train_hull_percent']:.1f}%"
    ax.text(
        0.03,
        0.97,
        coverage_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.92
        ),
    )
    ax.set_xlabel(f"PC1 ({nnbo_pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({nnbo_pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("F. Chemical-space coverage")
    ax.legend(
        frameon=False, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.7
    )
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, linestyle="--", linewidth=0.45, alpha=0.22)
    fig.tight_layout(rect=[0, 0.0, 1, 0.955])
    nnbo_main_png = os.path.join(
        NNBO_FIG_DIR, "AEM_manuscript_NN_vs_BO_postprocessing_figure.png"
    )
    nnbo_main_pdf = os.path.join(
        NNBO_FIG_DIR, "AEM_manuscript_NN_vs_BO_postprocessing_figure.pdf"
    )
    fig.savefig(nnbo_main_png, dpi=600, bbox_inches="tight")
    fig.savefig(nnbo_main_pdf, bbox_inches="tight")
    plt.show()
    print("Saved main manuscript figure:", nnbo_main_png, nnbo_main_pdf)
    return root
