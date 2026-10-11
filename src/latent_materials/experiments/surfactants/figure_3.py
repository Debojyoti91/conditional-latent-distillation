"""Surfactant NN/BO comparison. Produces manuscript Figure 3.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the figure 3 experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = (
        str(root / "IEMS_GEN_files/AEM/CVAE") if "surfactant" == "aem" else str(root)
    )
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    import os

    pass
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    warnings.filterwarnings("ignore")
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors
    except ImportError:
        _external_setup()
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors
    try:
        from scipy.spatial import ConvexHull, Delaunay
    except ImportError:
        _external_setup()
        from scipy.spatial import ConvexHull, Delaunay
    from sklearn.decomposition import PCA

    SEED = 123
    rng = np.random.default_rng(SEED)
    BASE_DIR = str(root)
    DATA_DIR = f"{BASE_DIR}/data"
    OUT_DIR = f"{BASE_DIR}/outputs"
    FIG_DIR = f"{BASE_DIR}/postprocessing_NN_BO_comparison"
    os.makedirs(FIG_DIR, exist_ok=True)
    NN_PATH = f"{OUT_DIR}/novel_candidates_property_guided_structural_clean.csv"
    BO_PATH = f"{OUT_DIR}/novel_candidates_property_guided_structural_clean_bo_qlogehvi_fast.csv"
    TRAIN_CANDIDATES = [
        f"{DATA_DIR}/surfpro_train80.csv",
        f"{DATA_DIR}/surfpro_train80_PO_balanced.csv",
        f"{DATA_DIR}/X_train_scaled.csv",
    ]
    TEST_CANDIDATES = [
        f"{DATA_DIR}/surfpro_test20_real.csv",
        f"{DATA_DIR}/surfpro_test20.csv",
        f"{DATA_DIR}/X_test_scaled.csv",
    ]
    OUT_STEM = os.path.join(FIG_DIR, "Figure4_surfactant_search_policy_NCS")

    def require_file(path, label):
        if not os.path.exists(path):
            raise FileNotFoundError(f"\nMissing {label}:\n{path}")
        print(f"[FOUND] {label}: {path}")
        return path

    def first_existing(paths, label):
        for path in paths:
            if os.path.exists(path):
                print(f"[FOUND] {label}: {path}")
                return path
        raise FileNotFoundError(f"\nCould not find {label}.\n" + "\n".join(paths))

    NN_PATH = require_file(NN_PATH, "NN cleaned candidates")
    BO_PATH = require_file(BO_PATH, "BO cleaned candidates")
    TRAIN_PATH = first_existing(TRAIN_CANDIDATES, "training reference set")
    TEST_PATH = first_existing(TEST_CANDIDATES, "test reference set")
    nn_raw = pd.read_csv(NN_PATH).dropna(subset=["SMILES"]).copy()
    bo_raw = pd.read_csv(BO_PATH).dropna(subset=["SMILES"]).copy()
    train_df = pd.read_csv(TRAIN_PATH).dropna(subset=["SMILES"]).copy()
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["SMILES"]).copy()
    nn_raw["branch"] = "NN"
    bo_raw["branch"] = "BO"
    print("\n============================================================")
    print("INPUT DATA")
    print("============================================================")
    print(f"NN cleaned candidates : {len(nn_raw)}")
    print(f"BO cleaned candidates : {len(bo_raw)}")
    print(f"Training molecules     : {len(train_df)}")
    print(f"Test molecules         : {len(test_df)}")
    from latent_materials.surfactants.descriptors import canon_smiles
    from latent_materials.surfactants.descriptors import longest_chain
    from latent_materials.surfactants.descriptors import count_EO
    from latent_materials.surfactants.descriptors import count_PO
    from latent_materials.surfactants.descriptors import ring_count
    from latent_materials.surfactants.descriptors import branching_index
    from latent_materials.surfactants.descriptors import aliphatic_fraction
    from latent_materials.surfactants.descriptors import detect_head
    from latent_materials.surfactants.descriptors import heuristic_accessibility_score

    def annotate_candidates(df):
        rows = []
        for smi in df["SMILES"]:
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
                        "EO_count": np.nan,
                        "PO_count": np.nan,
                        "ring_count": np.nan,
                        "branching_index": np.nan,
                        "aliphatic_fraction": np.nan,
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
                    "EO_count": count_EO(smi),
                    "PO_count": count_PO(smi),
                    "ring_count": ring_count(mol),
                    "branching_index": branching_index(mol),
                    "aliphatic_fraction": aliphatic_fraction(mol),
                    "head_type": detect_head(smi),
                    "SA_score": heuristic_accessibility_score(mol),
                }
            )
        annotation_df = pd.DataFrame(rows)
        out = pd.concat([df.reset_index(drop=True), annotation_df], axis=1)
        out = (
            out[out["valid"]].drop_duplicates("canonical_SMILES").reset_index(drop=True)
        )
        return out

    from latent_materials.surfactants.descriptors import composite_score

    FP_RADIUS = 2
    FP_BITS = 2048
    morgan_generator = AllChem.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    from latent_materials.surfactants.descriptors import smiles_to_fp
    from latent_materials.surfactants.descriptors import fps_for_smiles
    from latent_materials.surfactants.descriptors import max_tanimoto_to_reference
    from latent_materials.surfactants.descriptors import novelty_category

    def diversity_filter(df, threshold=0.8, max_keep=500):
        fps = []
        valid_indices = []
        for idx, smi in enumerate(df["canonical_SMILES"]):
            fp = smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
                valid_indices.append(idx)
        if not fps:
            return df.iloc[:0].copy()
        keep_indices = [valid_indices[0]]
        keep_fps = [fps[0]]
        for i in range(1, len(fps)):
            max_similarity = max(
                (
                    DataStructs.TanimotoSimilarity(fps[i], selected_fp)
                    for selected_fp in keep_fps
                )
            )
            if max_similarity < threshold:
                keep_indices.append(valid_indices[i])
                keep_fps.append(fps[i])
                if len(keep_indices) >= max_keep:
                    break
        return df.iloc[keep_indices].copy().reset_index(drop=True)

    print("\n[INFO] Annotating candidate chemistry...")
    nn = annotate_candidates(nn_raw)
    bo = annotate_candidates(bo_raw)
    nn["composite_score"] = nn.apply(composite_score, axis=1)
    bo["composite_score"] = bo.apply(composite_score, axis=1)
    ACCESSIBILITY_THRESHOLD = 8.0
    DIV_THRESHOLD = 0.8
    MAX_KEEP = 500
    nn_access = nn[nn["SA_score"] < ACCESSIBILITY_THRESHOLD].copy()
    bo_access = bo[bo["SA_score"] < ACCESSIBILITY_THRESHOLD].copy()
    nn_div = diversity_filter(nn_access, threshold=DIV_THRESHOLD, max_keep=MAX_KEEP)
    bo_div = diversity_filter(bo_access, threshold=DIV_THRESHOLD, max_keep=MAX_KEEP)
    print("\n============================================================")
    print("MATCHED POST-PROCESSING")
    print("============================================================")
    print(f"NN cleaned       : {len(nn)}")
    print(f"NN accessibility : {len(nn_access)}")
    print(f"NN diversity     : {len(nn_div)}")
    print(f"BO cleaned       : {len(bo)}")
    print(f"BO accessibility : {len(bo_access)}")
    print(f"BO diversity     : {len(bo_div)}")
    train_smiles = (
        train_df["SMILES"]
        .dropna()
        .map(canon_smiles)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    train_fps = [smiles_to_fp(smi) for smi in train_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]
    nn_div["max_tanimoto_to_train"] = max_tanimoto_to_reference(
        nn_div["canonical_SMILES"], train_fps
    )
    bo_div["max_tanimoto_to_train"] = max_tanimoto_to_reference(
        bo_div["canonical_SMILES"], train_fps
    )
    nn_div["novelty_category"] = nn_div["max_tanimoto_to_train"].apply(novelty_category)
    bo_div["novelty_category"] = bo_div["max_tanimoto_to_train"].apply(novelty_category)
    candidate_metrics = pd.concat([nn_div, bo_div], ignore_index=True)
    candidate_metrics_path = f"{FIG_DIR}/NN_BO_candidate_metrics.csv"
    candidate_metrics.to_csv(candidate_metrics_path, index=False)
    CATEGORY_ORDER = [
        "Very similar",
        "Moderately similar",
        "Moderately novel",
        "Highly novel",
    ]
    nn_category_pct = (
        nn_div["novelty_category"]
        .value_counts(normalize=True)
        .reindex(CATEGORY_ORDER)
        .fillna(0.0)
        * 100.0
    )
    bo_category_pct = (
        bo_div["novelty_category"]
        .value_counts(normalize=True)
        .reindex(CATEGORY_ORDER)
        .fillna(0.0)
        * 100.0
    )
    novelty_category_df = pd.DataFrame(
        {
            "category": CATEGORY_ORDER,
            "NN_percent": nn_category_pct.values,
            "BO_percent": bo_category_pct.values,
        }
    )
    novelty_category_path = f"{FIG_DIR}/NN_BO_novelty_category_percentages.csv"
    novelty_category_df.to_csv(novelty_category_path, index=False)

    def moderately_or_highly_novel_fraction(df):
        if len(df) == 0:
            return np.nan
        mask = df["novelty_category"].isin(["Moderately novel", "Highly novel"])
        return 100.0 * mask.mean()

    summary_rows = []
    for branch_name, full_df, access_df, diverse_df in [
        ("NN", nn, nn_access, nn_div),
        ("BO", bo, bo_access, bo_div),
    ]:
        summary_rows.append(
            {
                "branch": branch_name,
                "cleaned_candidates": len(full_df),
                "after_accessibility_filter": len(access_df),
                "after_diversity_filter": len(diverse_df),
                "accessibility_retention_percent": (
                    100.0 * len(access_df) / len(full_df) if len(full_df) else np.nan
                ),
                "diversity_retention_percent_of_cleaned": (
                    100.0 * len(diverse_df) / len(full_df) if len(full_df) else np.nan
                ),
                "mean_max_tanimoto_to_train": diverse_df[
                    "max_tanimoto_to_train"
                ].mean(),
                "std_max_tanimoto_to_train": diverse_df["max_tanimoto_to_train"].std(),
                "median_max_tanimoto_to_train": diverse_df[
                    "max_tanimoto_to_train"
                ].median(),
                "moderately_or_highly_novel_percent": moderately_or_highly_novel_fraction(
                    diverse_df
                ),
                "mean_composite_score": diverse_df["composite_score"].mean(),
                "std_composite_score": diverse_df["composite_score"].std(),
                "median_composite_score": diverse_df["composite_score"].median(),
                "mean_heuristic_accessibility_score": diverse_df["SA_score"].mean(),
                "std_heuristic_accessibility_score": diverse_df["SA_score"].std(),
                "median_heuristic_accessibility_score": diverse_df["SA_score"].median(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f"{FIG_DIR}/NN_BO_postprocessing_summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    train_canon = pd.Series(
        train_df["SMILES"].dropna().map(canon_smiles).dropna().unique()
    ).tolist()
    test_canon = pd.Series(
        test_df["SMILES"].dropna().map(canon_smiles).dropna().unique()
    ).tolist()
    nn_plot_smiles = nn_div["canonical_SMILES"].tolist()
    bo_plot_smiles = bo_div["canonical_SMILES"].tolist()
    fps_train = fps_for_smiles(train_canon)
    fps_test = fps_for_smiles(test_canon)
    fps_nn = fps_for_smiles(nn_plot_smiles)
    fps_bo = fps_for_smiles(bo_plot_smiles)
    if min(len(fps_train), len(fps_test), len(fps_nn), len(fps_bo)) == 0:
        raise RuntimeError("One or more fingerprint sets are empty.")
    fps_all = np.vstack([fps_train, fps_test, fps_nn, fps_bo])
    labels_all = (
        ["Training"] * len(fps_train)
        + ["Test"] * len(fps_test)
        + ["NN"] * len(fps_nn)
        + ["BO"] * len(fps_bo)
    )
    pca = PCA(n_components=2, random_state=SEED)
    coords_all = pca.fit_transform(fps_all)
    pca_df = pd.DataFrame(
        {"PC1": coords_all[:, 0], "PC2": coords_all[:, 1], "set": labels_all}
    )
    PCA_PATH = f"{FIG_DIR}/NN_BO_joint_PCA_coordinates.csv"
    pca_df.to_csv(PCA_PATH, index=False)

    def get_points(dataframe, label):
        return dataframe[dataframe["set"] == label][["PC1", "PC2"]].values

    def safe_hull(points):
        if points is None or len(points) < 3:
            return None
        try:
            return ConvexHull(points)
        except Exception:
            return None

    def points_inside_hull(points, hull_reference_points):
        if len(hull_reference_points) < 3 or len(points) == 0:
            return np.zeros(len(points), dtype=bool)
        try:
            triangulation = Delaunay(hull_reference_points)
            return triangulation.find_simplex(points) >= 0
        except Exception:
            return np.zeros(len(points), dtype=bool)

    pts_train = get_points(pca_df, "Training")
    pts_test = get_points(pca_df, "Test")
    pts_nn = get_points(pca_df, "NN")
    pts_bo = get_points(pca_df, "BO")
    hull_train = safe_hull(pts_train)
    hull_nn = safe_hull(pts_nn)
    hull_bo = safe_hull(pts_bo)
    if hull_train is not None:
        train_hull_reference_points = pts_train[hull_train.vertices]
    else:
        train_hull_reference_points = pts_train
    test_inside_train = points_inside_hull(pts_test, train_hull_reference_points)
    nn_inside_train = points_inside_hull(pts_nn, train_hull_reference_points)
    bo_inside_train = points_inside_hull(pts_bo, train_hull_reference_points)
    coverage_df = pd.DataFrame(
        [
            {
                "test_inside_train_hull_percent": 100.0 * np.mean(test_inside_train),
                "NN_inside_train_hull_percent": 100.0 * np.mean(nn_inside_train),
                "BO_inside_train_hull_percent": 100.0 * np.mean(bo_inside_train),
            }
        ]
    )
    COVERAGE_PATH = f"{FIG_DIR}/NN_BO_convex_hull_coverage_summary.csv"
    coverage_df.to_csv(COVERAGE_PATH, index=False)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.1,
            "font.weight": "normal",
            "axes.labelsize": 7.2,
            "axes.labelweight": "normal",
            "axes.titlesize": 7.2,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    NN_COLOR = "#0072B2"
    BO_COLOR = "#D55E00"
    TRAIN_COLOR = "#A6A6A6"
    TEST_COLOR = "#4D4D4D"
    HULL_FILL = "#D9D9D9"
    TEXT_COLOR = "#111111"
    BOX_EDGE = "#333333"

    def clean_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(width=0.65)
        ax.grid(False)

    def panel_label(ax, label):
        ax.text(
            -0.15,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="normal",
            ha="left",
            va="bottom",
            color=TEXT_COLOR,
        )

    def distribution_panel(ax, nn_values, bo_values, ylabel, ylim=None):
        nn_values = np.asarray(pd.Series(nn_values).dropna().astype(float))
        bo_values = np.asarray(pd.Series(bo_values).dropna().astype(float))
        positions = [0, 1]
        box = ax.boxplot(
            [nn_values, bo_values],
            positions=positions,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            whis=[5, 95],
            medianprops=dict(color=BOX_EDGE, linewidth=1.05),
            whiskerprops=dict(color=BOX_EDGE, linewidth=0.75),
            capprops=dict(color=BOX_EDGE, linewidth=0.75),
            boxprops=dict(color=BOX_EDGE, linewidth=0.75),
        )
        box["boxes"][0].set_facecolor(NN_COLOR)
        box["boxes"][0].set_alpha(0.16)
        box["boxes"][1].set_facecolor(BO_COLOR)
        box["boxes"][1].set_alpha(0.16)
        jitter_nn = rng.normal(loc=0.0, scale=0.055, size=len(nn_values))
        jitter_bo = rng.normal(loc=0.0, scale=0.055, size=len(bo_values))
        ax.scatter(
            np.zeros(len(nn_values)) + jitter_nn,
            nn_values,
            s=11,
            color=NN_COLOR,
            alpha=0.62,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )
        ax.scatter(
            np.ones(len(bo_values)) + jitter_bo,
            bo_values,
            s=9,
            color=BO_COLOR,
            alpha=0.48,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(["NN", "BO"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        clean_axis(ax)

    fig = plt.figure(figsize=(7.15, 4.95), facecolor="white")
    fig.suptitle(
        "Surfactant Search-Policy Comparison Under Matched Post-processing",
        x=0.5,
        y=0.982,
        fontsize=10.0,
        fontweight="normal",
        ha="center",
        va="top",
        color=TEXT_COLOR,
    )
    global_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=NN_COLOR,
            markeredgecolor="none",
            markersize=5.2,
            label="NN",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BO_COLOR,
            markeredgecolor="none",
            markersize=5.2,
            label="BO",
        ),
    ]
    fig.legend(
        handles=global_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.942),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.075,
        right=0.985,
        bottom=0.1,
        top=0.865,
        hspace=0.47,
        wspace=0.43,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    distribution_panel(
        axes[0],
        nn_div["max_tanimoto_to_train"],
        bo_div["max_tanimoto_to_train"],
        ylabel="Maximum Tanimoto\nsimilarity to training",
        ylim=(0.0, 1.02),
    )
    panel_label(axes[0], "a")
    ax = axes[1]
    y = np.arange(len(CATEGORY_ORDER))
    bar_height = 0.34
    ax.barh(
        y - bar_height / 2,
        nn_category_pct.values,
        height=bar_height,
        color=NN_COLOR,
        alpha=0.88,
        linewidth=0,
    )
    ax.barh(
        y + bar_height / 2,
        bo_category_pct.values,
        height=bar_height,
        color=BO_COLOR,
        alpha=0.88,
        linewidth=0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        ["Very similar", "Moderately similar", "Moderately novel", "Highly novel"],
        fontsize=5.9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Candidates (%)")
    ax.set_ylabel("")
    ax.set_xlim(left=0)
    max_category_percent = max(
        float(np.nanmax(nn_category_pct.values)),
        float(np.nanmax(bo_category_pct.values)),
    )
    ax.set_xlim(0, max_category_percent * 1.12)
    ax.tick_params(axis="y", pad=2.5)
    clean_axis(ax)
    panel_label(ax, "b")
    ax = axes[2]
    stages = ["Accessibility", "Diversity"]
    nn_counts = [len(nn_access), len(nn_div)]
    bo_counts = [len(bo_access), len(bo_div)]
    x = np.arange(len(stages))
    bar_width = 0.36
    ax.bar(
        x - bar_width / 2,
        nn_counts,
        width=bar_width,
        color=NN_COLOR,
        alpha=0.88,
        linewidth=0,
    )
    ax.bar(
        x + bar_width / 2,
        bo_counts,
        width=bar_width,
        color=BO_COLOR,
        alpha=0.88,
        linewidth=0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Retained candidates")
    ax.set_ylim(bottom=0)
    clean_axis(ax)
    panel_label(ax, "c")
    distribution_panel(
        axes[3],
        nn_div["composite_score"],
        bo_div["composite_score"],
        ylabel="Composite amphiphile score",
    )
    panel_label(axes[3], "d")
    distribution_panel(
        axes[4],
        nn_div["SA_score"],
        bo_div["SA_score"],
        ylabel="Heuristic-accessibility score",
    )
    panel_label(axes[4], "e")
    ax = axes[5]
    if hull_train is not None:
        train_polygon = Polygon(
            pts_train[hull_train.vertices],
            closed=True,
            facecolor=HULL_FILL,
            edgecolor=TRAIN_COLOR,
            linewidth=0.75,
            alpha=0.22,
            zorder=0,
        )
        ax.add_patch(train_polygon)
    training_points = pca_df[pca_df["set"] == "Training"]
    ax.scatter(
        training_points["PC1"],
        training_points["PC2"],
        s=5,
        marker="o",
        color=TRAIN_COLOR,
        alpha=0.15,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    test_points = pca_df[pca_df["set"] == "Test"]
    ax.scatter(
        test_points["PC1"],
        test_points["PC2"],
        s=10,
        marker="o",
        facecolors="none",
        edgecolors=TEST_COLOR,
        linewidths=0.45,
        alpha=0.4,
        rasterized=True,
        zorder=2,
    )
    bo_points = pca_df[pca_df["set"] == "BO"]
    ax.scatter(
        bo_points["PC1"],
        bo_points["PC2"],
        s=12,
        marker="o",
        color=BO_COLOR,
        alpha=0.58,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    nn_points = pca_df[pca_df["set"] == "NN"]
    ax.scatter(
        nn_points["PC1"],
        nn_points["PC2"],
        s=18,
        marker="^",
        color=NN_COLOR,
        alpha=0.82,
        edgecolors="none",
        rasterized=True,
        zorder=4,
    )
    if hull_nn is not None:
        nn_vertices = pts_nn[hull_nn.vertices]
        nn_loop = np.vstack([nn_vertices, nn_vertices[0]])
        ax.plot(
            nn_loop[:, 0],
            nn_loop[:, 1],
            color=NN_COLOR,
            linewidth=1.05,
            alpha=0.9,
            zorder=5,
        )
    if hull_bo is not None:
        bo_vertices = pts_bo[hull_bo.vertices]
        bo_loop = np.vstack([bo_vertices, bo_vertices[0]])
        ax.plot(
            bo_loop[:, 0],
            bo_loop[:, 1],
            color=BO_COLOR,
            linewidth=1.05,
            alpha=0.9,
            zorder=5,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    clean_axis(ax)
    panel_label(ax, "f")
    pca_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TRAIN_COLOR,
            markeredgecolor="none",
            alpha=0.55,
            markersize=4.2,
            label="Training",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=TEST_COLOR,
            markeredgewidth=0.7,
            markersize=4.2,
            label="Test",
        ),
    ]
    ax.legend(
        handles=pca_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
        fontsize=5.8,
    )
    PNG_PATH = OUT_STEM + ".png"
    PDF_PATH = OUT_STEM + ".pdf"
    SVG_PATH = OUT_STEM + ".svg"
    fig.savefig(
        PNG_PATH, dpi=600, bbox_inches="tight", pad_inches=0.045, facecolor="white"
    )
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.045, facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.045, facecolor="white")
    plt.show()
    print("\n============================================================")
    print("FIGURE 4 COMPLETE")
    print("============================================================")
    print(f"NN cleaned candidates             : {len(nn)}")
    print(f"NN after accessibility filter     : {len(nn_access)}")
    print(f"NN after diversity filter         : {len(nn_div)}")
    print()
    print(f"BO cleaned candidates             : {len(bo)}")
    print(f"BO after accessibility filter     : {len(bo_access)}")
    print(f"BO after diversity filter         : {len(bo_div)}")
    print()
    print("Mean maximum Tanimoto:")
    print(f"  NN : {nn_div['max_tanimoto_to_train'].mean():.4f}")
    print(f"  BO : {bo_div['max_tanimoto_to_train'].mean():.4f}")
    print()
    print("Median maximum Tanimoto:")
    print(f"  NN : {nn_div['max_tanimoto_to_train'].median():.4f}")
    print(f"  BO : {bo_div['max_tanimoto_to_train'].median():.4f}")
    print()
    print("Moderately/highly novel:")
    print(f"  NN : {moderately_or_highly_novel_fraction(nn_div):.1f}%")
    print(f"  BO : {moderately_or_highly_novel_fraction(bo_div):.1f}%")
    print()
    print("Inside projected training-set PCA hull:")
    print(f"  NN : {100.0 * np.mean(nn_inside_train):.1f}%")
    print(f"  BO : {100.0 * np.mean(bo_inside_train):.1f}%")
    print("\nSaved:")
    print("PNG       :", PNG_PATH)
    print("PDF       :", PDF_PATH)
    print("SVG       :", SVG_PATH)
    print("Metrics   :", candidate_metrics_path)
    print("Summary   :", summary_path)
    print("Novelty   :", novelty_category_path)
    print("PCA       :", PCA_PATH)
    print("Coverage  :", COVERAGE_PATH)
    print("============================================================")
    return root
