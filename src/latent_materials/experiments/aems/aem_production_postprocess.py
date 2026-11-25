"""482 -> 322 -> 125 production postprocessing; no conductivity term in recorded production score.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import display, require_dependencies


def run(workspace):
    """Execute the aem production postprocess experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    "\nCorrected AEM post-processing workflow\n\nInput:\n    /content/drive/MyDrive/surfactants/IEMS_GEN_files/aem_novel_candidates_clean_2.csv\n\nOutputs:\n    /content/drive/MyDrive/surfactants/IEMS_GEN_files/postprocessing_AEM_corrected/\n\nMain correction:\n    Correctly classifies aliphatic quaternary ammonium [N+](C)(C)C\n    separately from aromatic/imidazolium N+ groups.\n"
    pass
    pass
    import os
    import json
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Draw
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    warnings.filterwarnings("ignore")
    print("Dependencies loaded successfully.")
    BASE_DIR = str(root / "IEMS_GEN_files")
    CSV_PATH = f"{BASE_DIR}/aem_novel_candidates_clean_2.csv"
    POSTPROC_DIR = f"{BASE_DIR}/postprocessing_AEM_corrected"
    os.makedirs(POSTPROC_DIR, exist_ok=True)
    print("Input CSV:", CSV_PATH)
    print("Postprocessing output directory:", POSTPROC_DIR)
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} AEM-generated candidates")
    print("Columns:", df.columns.tolist())
    if "SMILES" not in df.columns:
        raise ValueError("Input file must contain a column named 'SMILES'.")
    df["SMILES"] = df["SMILES"].astype(str)
    print("\nExtracting corrected AEM structural features...")
    OO_smarts = Chem.MolFromSmarts("O-O")
    ether_smarts = Chem.MolFromSmarts("COC")
    from latent_materials.aems.descriptors import valid_mol
    from latent_materials.aems.descriptors import formal_charge
    from latent_materials.aems.descriptors import fragment_count
    from latent_materials.aems.descriptors import has_peroxy

    def count_substructure(mol, patt):
        try:
            if patt is None:
                return 0
            return len(mol.GetSubstructMatches(patt))
        except Exception:
            return 0

    from latent_materials.aems.descriptors import detect_aem_headgroup
    from latent_materials.aems.descriptors import longest_carbon_chain_graph
    from latent_materials.aems.descriptors import aromatic_fraction
    from latent_materials.aems.descriptors import aliphatic_carbon_fraction
    from latent_materials.aems.descriptors import heteroatom_count
    from latent_materials.aems.descriptors import cationic_site_count
    from latent_materials.aems.descriptors import n_cationic_nitrogens
    from latent_materials.aems.descriptors import n_cationic_phosphorus
    from latent_materials.aems.descriptors import ion_exchange_capacity_proxy
    from latent_materials.aems.descriptors import branching_index

    features = []
    for smi in df["SMILES"]:
        mol = valid_mol(smi)
        if mol is None:
            features.append(
                {
                    "canonical_SMILES": None,
                    "valid_rdkit": False,
                    "MW": np.nan,
                    "LogP": np.nan,
                    "TPSA": np.nan,
                    "HBD": np.nan,
                    "HBA": np.nan,
                    "RotBonds": np.nan,
                    "ring_count": np.nan,
                    "formal_charge": np.nan,
                    "fragment_count": np.nan,
                    "cationic_site_count": np.nan,
                    "cationic_N_count": np.nan,
                    "cationic_P_count": np.nan,
                    "aem_head_type": "invalid",
                    "has_cationic_head": False,
                    "has_peroxy": np.nan,
                    "ether_count": np.nan,
                    "aromatic_atom_fraction": np.nan,
                    "aliphatic_carbon_fraction": np.nan,
                    "heteroatom_count": np.nan,
                    "longest_carbon_chain": np.nan,
                    "branching_index": np.nan,
                    "IEC_proxy_mmol_g": np.nan,
                    "needs_counterion": np.nan,
                }
            )
            continue
        fc = formal_charge(mol)
        head = detect_aem_headgroup(mol, smi)
        n_cat = cationic_site_count(mol)
        features.append(
            {
                "canonical_SMILES": Chem.MolToSmiles(mol, canonical=True),
                "valid_rdkit": True,
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "RotBonds": Descriptors.NumRotatableBonds(mol),
                "ring_count": mol.GetRingInfo().NumRings(),
                "formal_charge": fc,
                "fragment_count": fragment_count(smi),
                "cationic_site_count": n_cat,
                "cationic_N_count": n_cationic_nitrogens(mol),
                "cationic_P_count": n_cationic_phosphorus(mol),
                "aem_head_type": head,
                "has_cationic_head": head not in ["invalid", "no_cationic_head"],
                "has_peroxy": has_peroxy(mol),
                "ether_count": count_substructure(mol, ether_smarts),
                "aromatic_atom_fraction": aromatic_fraction(mol),
                "aliphatic_carbon_fraction": aliphatic_carbon_fraction(mol),
                "heteroatom_count": heteroatom_count(mol),
                "longest_carbon_chain": longest_carbon_chain_graph(mol),
                "branching_index": branching_index(mol),
                "IEC_proxy_mmol_g": ion_exchange_capacity_proxy(mol),
                "needs_counterion": int(fc != 0),
            }
        )
    feat_df = pd.DataFrame(features)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    before = len(df)
    df = (
        df.dropna(subset=["canonical_SMILES"])
        .drop_duplicates("canonical_SMILES")
        .reset_index(drop=True)
    )
    after = len(df)
    print(f"Valid unique molecules: {after} / {before}")
    annotated_path = f"{POSTPROC_DIR}/aem_candidates_annotated.csv"
    df.to_csv(annotated_path, index=False)
    print("Saved annotated dataset:", annotated_path)
    print("\nCalculating simple SA scores...")
    from latent_materials.aems.descriptors import sa_score_simple

    df["SA_score"] = df["canonical_SMILES"].apply(
        lambda s: (
            sa_score_simple(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.nan
        )
    )
    df_aem_filtered = df[
        (df["valid_rdkit"] == True)
        & (df["has_cationic_head"] == True)
        & (df["has_peroxy"] == 0)
        & (df["SA_score"] < 10.0)
    ].copy()
    print(
        f"After AEM structural + SA filter: {len(df_aem_filtered)} / {len(df)} molecules"
    )
    filtered_path = f"{POSTPROC_DIR}/aem_candidates_filtered.csv"
    df_aem_filtered.to_csv(filtered_path, index=False)
    print("Saved filtered dataset:", filtered_path)
    print("\nApplying diversity filter...")

    def diversity_filter(
        df_in,
        smiles_col="canonical_SMILES",
        threshold=0.8,
        max_keep=500,
        fp_radius=2,
        fp_bits=2048,
    ):
        """
        Greedy Tanimoto diversity filtering.
        Keeps molecules with max similarity < threshold
        to previously selected molecules.
        """
        smiles_list = df_in[smiles_col].tolist()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps = []
        valid_idx = []
        for i, mol in enumerate(mols):
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, fp_radius, nBits=fp_bits
                )
                fps.append(fp)
                valid_idx.append(i)
        if len(fps) == 0:
            return df_in.iloc[:0].copy()
        keep_idx = [valid_idx[0]]
        keep_fps = [fps[0]]
        for i in range(1, len(fps)):
            max_sim = max(
                (DataStructs.TanimotoSimilarity(fps[i], fp) for fp in keep_fps)
            )
            if max_sim < threshold:
                keep_idx.append(valid_idx[i])
                keep_fps.append(fps[i])
                if len(keep_idx) >= max_keep:
                    break
        return df_in.iloc[keep_idx].copy().reset_index(drop=True)

    df_diverse = diversity_filter(
        df_aem_filtered, smiles_col="canonical_SMILES", threshold=0.8, max_keep=500
    )
    print(f"After diversity filter: {len(df_diverse)} molecules")
    diverse_path = f"{POSTPROC_DIR}/aem_candidates_diverse.csv"
    df_diverse.to_csv(diverse_path, index=False)
    print("Saved diverse set:", diverse_path)
    print("\nGenerating Morgan fingerprints...")
    fps_array = []
    valid_smiles = []
    for smi in df_diverse["canonical_SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.array(fp)
            fps_array.append(arr)
            valid_smiles.append(smi)
    fps_array = np.array(fps_array)
    print(f"Fingerprint matrix: {fps_array.shape}")
    df_vis = (
        df_diverse[df_diverse["canonical_SMILES"].isin(valid_smiles)]
        .copy()
        .reset_index(drop=True)
    )
    head_types = df_vis["aem_head_type"].values
    head_map = {h: i for i, h in enumerate(df_vis["aem_head_type"].unique())}
    colors_head = [head_map[h] for h in head_types]
    print("\nGenerating t-SNE visualization...")
    if len(fps_array) > 50:
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, max(5, len(fps_array) // 3)),
        )
        coords_tsne = tsne.fit_transform(fps_array)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes[0, 0].scatter(
            coords_tsne[:, 0],
            coords_tsne[:, 1],
            c=colors_head,
            cmap="tab10",
            s=40,
            alpha=0.7,
        )
        axes[0, 0].set_title("AEM Chemical Space colored by Corrected Head Type")
        axes[0, 0].set_xlabel("t-SNE 1")
        axes[0, 0].set_ylabel("t-SNE 2")
        for head, idx in head_map.items():
            axes[0, 0].scatter(
                [], [], c=plt.cm.tab10(idx / max(1, len(head_map))), label=head, s=60
            )
        axes[0, 0].legend(loc="best", fontsize=8)
        sc2 = axes[0, 1].scatter(
            coords_tsne[:, 0],
            coords_tsne[:, 1],
            c=df_vis["MW"],
            cmap="viridis",
            s=40,
            alpha=0.7,
        )
        axes[0, 1].set_title("AEM Chemical Space colored by Molecular Weight")
        axes[0, 1].set_xlabel("t-SNE 1")
        axes[0, 1].set_ylabel("t-SNE 2")
        plt.colorbar(sc2, ax=axes[0, 1], label="MW")
        sc3 = axes[1, 0].scatter(
            coords_tsne[:, 0],
            coords_tsne[:, 1],
            c=df_vis["IEC_proxy_mmol_g"],
            cmap="plasma",
            s=40,
            alpha=0.7,
        )
        axes[1, 0].set_title("AEM Chemical Space colored by IEC Proxy")
        axes[1, 0].set_xlabel("t-SNE 1")
        axes[1, 0].set_ylabel("t-SNE 2")
        plt.colorbar(sc3, ax=axes[1, 0], label="IEC proxy")
        sc4 = axes[1, 1].scatter(
            coords_tsne[:, 0],
            coords_tsne[:, 1],
            c=df_vis["LogP"],
            cmap="coolwarm",
            s=40,
            alpha=0.7,
        )
        axes[1, 1].set_title("AEM Chemical Space colored by LogP")
        axes[1, 1].set_xlabel("t-SNE 1")
        axes[1, 1].set_ylabel("t-SNE 2")
        plt.colorbar(sc4, ax=axes[1, 1], label="LogP")
        plt.tight_layout()
        tsne_path = f"{POSTPROC_DIR}/aem_chemical_space_tsne.png"
        plt.savefig(tsne_path, dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved t-SNE:", tsne_path)
    else:
        print(f"Skipping t-SNE: need >50 molecules, have {len(fps_array)}")
    print("\nGenerating PCA visualization...")
    if len(fps_array) > 2:
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(fps_array)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(
            coords_pca[:, 0],
            coords_pca[:, 1],
            c=colors_head,
            cmap="tab10",
            s=40,
            alpha=0.7,
        )
        axes[0].set_title(
            f"PCA by Corrected AEM Head Type | Var: {pca.explained_variance_ratio_.sum():.2%}"
        )
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        for head, idx in head_map.items():
            axes[0].scatter(
                [], [], c=plt.cm.tab10(idx / max(1, len(head_map))), label=head, s=60
            )
        axes[0].legend(loc="best", fontsize=8)
        sc = axes[1].scatter(
            coords_pca[:, 0],
            coords_pca[:, 1],
            c=df_vis["MW"],
            cmap="viridis",
            s=40,
            alpha=0.7,
        )
        axes[1].set_title("PCA colored by Molecular Weight")
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.colorbar(sc, ax=axes[1], label="MW")
        plt.tight_layout()
        pca_path = f"{POSTPROC_DIR}/aem_chemical_space_pca.png"
        plt.savefig(pca_path, dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved PCA:", pca_path)
    else:
        print(f"Skipping PCA: need >2 molecules, have {len(fps_array)}")
    print("\nPerforming clustering analysis...")
    if len(fps_array) >= 10:
        inertias = []
        K_range = range(2, min(11, max(3, len(fps_array) // 5)))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(fps_array)
            inertias.append(kmeans.inertia_)
        plt.figure(figsize=(8, 5))
        plt.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for AEM Candidate Clustering")
        plt.grid(alpha=0.3)
        elbow_path = f"{POSTPROC_DIR}/aem_clustering_elbow.png"
        plt.savefig(elbow_path, dpi=150, bbox_inches="tight")
        plt.show()
        optimal_k = min(5, max(2, len(fps_array) // 3))
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(fps_array)
        df_vis["cluster"] = clusters
        df_diverse = df_diverse.merge(
            df_vis[["canonical_SMILES", "cluster"]], on="canonical_SMILES", how="left"
        )
        if len(fps_array) > 50 and "coords_tsne" in locals():
            plt.figure(figsize=(10, 8))
            sc = plt.scatter(
                coords_tsne[:, 0],
                coords_tsne[:, 1],
                c=clusters,
                cmap="Set3",
                s=60,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.4,
            )
            plt.colorbar(sc, label="Cluster")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title(f"AEM Candidate Clusters on t-SNE | k={optimal_k}")
            clust_tsne_path = f"{POSTPROC_DIR}/aem_clusters_on_tsne.png"
            plt.savefig(clust_tsne_path, dpi=200, bbox_inches="tight")
            plt.show()
        print(f"\nCluster statistics | k={optimal_k}")
        for i in range(optimal_k):
            cdf = df_diverse[df_diverse["cluster"] == i]
            print(f"\nCluster {i}: {len(cdf)} molecules")
            print("  Head types:", cdf["aem_head_type"].value_counts().to_dict())
            print(f"  Avg MW: {cdf['MW'].mean():.1f}")
            print(f"  Avg LogP: {cdf['LogP'].mean():.2f}")
            print(f"  Avg IEC proxy: {cdf['IEC_proxy_mmol_g'].mean():.3f}")
        clustered_path = f"{POSTPROC_DIR}/aem_candidates_clustered.csv"
        df_diverse.to_csv(clustered_path, index=False)
        print("\nSaved clustered dataset:", clustered_path)
    else:
        print(f"Skipping clustering: need >=10 molecules, have {len(fps_array)}")
    print("\nGenerating AEM property distribution plots...")
    plot_cols = [
        ("MW", "Molecular Weight"),
        ("LogP", "LogP"),
        ("TPSA", "TPSA"),
        ("RotBonds", "Rotatable Bonds"),
        ("ring_count", "Ring Count"),
        ("formal_charge", "Formal Charge"),
        ("cationic_site_count", "Cationic Site Count"),
        ("cationic_N_count", "Cationic N Count"),
        ("cationic_P_count", "Cationic P Count"),
        ("ether_count", "Ether Count"),
        ("IEC_proxy_mmol_g", "IEC Proxy"),
        ("SA_score", "SA Score"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, plot_cols):
        values = df_diverse[col].dropna()
        if len(values) == 0:
            ax.set_title(f"{label} | no data")
            continue
        ax.hist(values, bins=30, alpha=0.75, edgecolor="black")
        ax.axvline(
            values.median(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Median: {values.median():.2f}",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    dist_path = f"{POSTPROC_DIR}/aem_property_distributions.png"
    plt.savefig(dist_path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved property distributions:", dist_path)
    plt.figure(figsize=(8, 5))
    head_counts = df_diverse["aem_head_type"].value_counts()
    plt.bar(range(len(head_counts)), head_counts.values, alpha=0.75, edgecolor="black")
    plt.xticks(range(len(head_counts)), head_counts.index, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Corrected AEM Cationic Headgroup Distribution")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    head_path = f"{POSTPROC_DIR}/aem_headgroup_distribution_corrected.png"
    plt.savefig(head_path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved corrected headgroup distribution:", head_path)
    print("\nGenerating AEM feature correlation matrix...")
    corr_features = [
        "MW",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
        "ring_count",
        "formal_charge",
        "cationic_site_count",
        "cationic_N_count",
        "cationic_P_count",
        "ether_count",
        "aromatic_atom_fraction",
        "aliphatic_carbon_fraction",
        "heteroatom_count",
        "longest_carbon_chain",
        "branching_index",
        "IEC_proxy_mmol_g",
        "SA_score",
    ]
    possible_cond_cols = [
        "predicted_conductivity_oldGNN",
        "predicted_conductivity",
        "Conductivity",
        "Anion_Conductivity_Scm",
        "Anion_Conductivity_pred_Scm",
    ]
    cond_col = None
    for c in possible_cond_cols:
        if c in df_diverse.columns:
            cond_col = c
            corr_features.append(c)
            print(f"Found conductivity-like column: {cond_col}")
            break
    corr_df = (
        df_diverse[corr_features]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=1, how="all")
    )
    if corr_df.shape[1] >= 2:
        corr_matrix = corr_df.corr()
        plt.figure(figsize=(13, 11))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("AEM Candidate Feature Correlation Matrix", fontsize=14, pad=20)
        plt.tight_layout()
        corr_path = f"{POSTPROC_DIR}/aem_correlation_matrix.png"
        plt.savefig(corr_path, dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved correlation matrix:", corr_path)
    else:
        print("Skipping correlation matrix: not enough numeric features.")
    print("\nSelecting top AEM candidates...")
    from latent_materials.aems.descriptors import normalize_clip

    if cond_col is not None:
        cond_values = pd.to_numeric(df_diverse[cond_col], errors="coerce").dropna()
        if len(cond_values) > 0:
            cond_low = float(cond_values.quantile(0.05))
            cond_high = float(cond_values.quantile(0.95))
        else:
            cond_low, cond_high = (0.0, 1.0)
    else:
        cond_low, cond_high = (0.0, 1.0)

    def aem_composite_score(row):
        """
        Corrected AEM-specific heuristic score.

        Higher = better.

        Rewarded:
          - predicted conductivity, if available
          - cationic site presence
          - IEC proxy in practical range
          - lower SA score
          - moderate MW
          - moderate hydrophobicity

        Penalized:
          - peroxide motif
          - no cationic head
          - disconnected fragments
        """
        score = 0.0
        if cond_col is not None and cond_col in row:
            cond_val = row[cond_col]
            cond_score = normalize_clip(cond_val, cond_low, cond_high)
            score += 0.35 * cond_score
        ncat = row.get("cationic_site_count", 0)
        if ncat >= 1:
            score += 0.15
        if ncat == 1:
            score += 0.05
        elif ncat == 2:
            score += 0.04
        iec = row.get("IEC_proxy_mmol_g", np.nan)
        if not pd.isna(iec):
            if 1.0 <= iec <= 4.0:
                score += 0.2
            elif 0.5 <= iec < 1.0 or 4.0 < iec <= 6.0:
                score += 0.1
        sa = row.get("SA_score", np.nan)
        if not pd.isna(sa):
            sa_quality = 1.0 - min(sa / 10.0, 1.0)
            score += 0.15 * sa_quality
        mw = row.get("MW", np.nan)
        if not pd.isna(mw):
            mw_quality = 1.0 - abs(mw - 450.0) / 400.0
            score += 0.1 * max(0.0, mw_quality)
        logp = row.get("LogP", np.nan)
        if not pd.isna(logp):
            logp_quality = 1.0 - abs(logp - 2.5) / 4.0
            score += 0.05 * max(0.0, logp_quality)
        if row.get("has_peroxy", 0) == 1:
            score -= 0.5
        if row.get("has_cationic_head", False) is False:
            score -= 0.5
        if row.get("fragment_count", 1) > 1:
            score -= 0.1
        return score

    df_diverse["aem_composite_score"] = df_diverse.apply(aem_composite_score, axis=1)
    if cond_col is not None:
        df_ranked = df_diverse.sort_values(
            [cond_col, "aem_composite_score"], ascending=[False, False]
        ).reset_index(drop=True)
    else:
        df_ranked = df_diverse.sort_values(
            "aem_composite_score", ascending=False
        ).reset_index(drop=True)
    ranked_path = f"{POSTPROC_DIR}/aem_candidates_ranked.csv"
    df_ranked.to_csv(ranked_path, index=False)
    print("Saved ranked candidates:", ranked_path)
    top_100 = df_ranked.head(100).copy()
    top_20 = df_ranked.head(20).copy()
    top_10 = df_ranked.head(10).copy()
    top_100_path = f"{POSTPROC_DIR}/AEM_TOP_100_candidates.csv"
    top_20_path = f"{POSTPROC_DIR}/AEM_TOP_20_candidates.csv"
    top_10_path = f"{POSTPROC_DIR}/AEM_TOP_10_candidates_for_MD.csv"
    top_100.to_csv(top_100_path, index=False)
    top_20.to_csv(top_20_path, index=False)
    top_10.to_csv(top_10_path, index=False)
    print("Saved top 100:", top_100_path)
    print("Saved top 20:", top_20_path)
    print("Saved top 10 for MD:", top_10_path)
    print("\nGenerating top molecule grid image...")
    top_mols = []
    legends = []
    for i, row in top_20.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_SMILES"])
        if mol is None:
            continue
        top_mols.append(mol)
        legend = f"#{i + 1}\nScore: {row['aem_composite_score']:.2f}\n"
        legend += f"Head: {row['aem_head_type']}\n"
        legend += f"MW: {row['MW']:.0f}, IEC: {row['IEC_proxy_mmol_g']:.2f}"
        if cond_col is not None:
            legend += f"\nCond: {row[cond_col]:.3g}"
        legends.append(legend)
    if len(top_mols) > 0:
        img = Draw.MolsToGridImage(
            top_mols[:20],
            molsPerRow=4,
            subImgSize=(400, 400),
            legends=legends[:20],
            returnPNG=False,
        )
        top_grid_path = f"{POSTPROC_DIR}/AEM_TOP_20_molecules.png"
        img.save(top_grid_path)
        display(img)
        print("Saved top 20 molecule grid:", top_grid_path)
    else:
        print("No valid molecules available for grid image.")
    print("\n" + "=" * 90)
    print(" CORRECTED AEM POSTPROCESSING SUMMARY REPORT ".center(90, "="))
    print("=" * 90)
    print(f"\nInput file: {CSV_PATH}")
    print(f"Input molecules loaded: {len(pd.read_csv(CSV_PATH))}")
    print(f"Valid unique molecules: {len(df)}")
    print(f"After AEM structural + SA filter: {len(df_aem_filtered)}")
    print(f"After diversity filter: {len(df_diverse)}")
    print("\n--- AEM Structural Statistics ---")
    for col, label in [
        ("MW", "Molecular Weight"),
        ("LogP", "LogP"),
        ("TPSA", "TPSA"),
        ("RotBonds", "Rotatable Bonds"),
        ("formal_charge", "Formal Charge"),
        ("cationic_site_count", "Cationic Sites"),
        ("cationic_N_count", "Cationic N"),
        ("cationic_P_count", "Cationic P"),
        ("IEC_proxy_mmol_g", "IEC Proxy"),
        ("SA_score", "SA Score"),
    ]:
        if col in df_diverse.columns:
            print(
                f"{label:25s}: {df_diverse[col].mean():.3f} ± {df_diverse[col].std():.3f}"
            )
    print("\n--- Corrected Cationic Headgroup Distribution ---")
    for head, count in df_diverse["aem_head_type"].value_counts().items():
        print(f"{head:25s}: {count:4d} ({100 * count / len(df_diverse):5.1f}%)")
    if cond_col is not None:
        print(f"\n--- Conductivity Column Used for Ranking: {cond_col} ---")
        print(df_diverse[cond_col].describe())
    else:
        print(
            "\nNo conductivity prediction column found. Ranking used structural AEM score only."
        )
    print("\n--- Top 10 AEM Candidates ---")
    for i, row in top_10.iterrows():
        print(f"\n#{i + 1}: Score = {row['aem_composite_score']:.3f}")
        if cond_col is not None:
            print(f"  Predicted conductivity: {row[cond_col]}")
        print(f"  SMILES: {row['canonical_SMILES']}")
        print(f"  Head: {row['aem_head_type']}")
        print(f"  MW: {row['MW']:.1f}")
        print(f"  LogP: {row['LogP']:.2f}")
        print(f"  TPSA: {row['TPSA']:.1f}")
        print(f"  Formal charge: {row['formal_charge']}")
        print(f"  Cationic sites: {row['cationic_site_count']}")
        print(f"  IEC proxy: {row['IEC_proxy_mmol_g']:.3f}")
        print(f"  SA score: {row['SA_score']:.3f}")
    print("\n--- Saved Files ---")
    print(f"1. Annotated dataset:       {annotated_path}")
    print(f"2. Filtered dataset:        {filtered_path}")
    print(f"3. Diverse dataset:         {diverse_path}")
    if "clustered_path" in locals():
        print(f"4. Clustered dataset:       {clustered_path}")
    print(f"5. Ranked candidates:       {ranked_path}")
    print(f"6. Top 100 candidates:      {top_100_path}")
    print(f"7. Top 20 candidates:       {top_20_path}")
    print(f"8. Top 10 candidates for MD:{top_10_path}")
    print(f"9. Visualizations in:       {POSTPROC_DIR}")
    print("\n" + "=" * 90)
    print(" CORRECTED AEM POSTPROCESSING COMPLETE ".center(90, "="))
    print("=" * 90)
    summary_stats = {
        "method": "Corrected AEM BO generated candidate postprocessing",
        "input_file": CSV_PATH,
        "total_loaded": int(len(pd.read_csv(CSV_PATH))),
        "valid_unique": int(len(df)),
        "after_aem_filter": int(len(df_aem_filtered)),
        "after_diversity_filter": int(len(df_diverse)),
        "diversity_threshold": 0.8,
        "top_candidates_exported": {
            "top_100": top_100_path,
            "top_20": top_20_path,
            "top_10_for_MD": top_10_path,
        },
        "conductivity_column_used": cond_col,
        "statistics": {
            "MW_mean": float(df_diverse["MW"].mean()),
            "MW_std": float(df_diverse["MW"].std()),
            "LogP_mean": float(df_diverse["LogP"].mean()),
            "LogP_std": float(df_diverse["LogP"].std()),
            "TPSA_mean": float(df_diverse["TPSA"].mean()),
            "RotBonds_mean": float(df_diverse["RotBonds"].mean()),
            "formal_charge_mean": float(df_diverse["formal_charge"].mean()),
            "cationic_site_count_mean": float(df_diverse["cationic_site_count"].mean()),
            "cationic_N_count_mean": float(df_diverse["cationic_N_count"].mean()),
            "cationic_P_count_mean": float(df_diverse["cationic_P_count"].mean()),
            "IEC_proxy_mean": float(df_diverse["IEC_proxy_mmol_g"].mean()),
            "SA_score_mean": float(df_diverse["SA_score"].mean()),
        },
        "corrected_headgroup_distribution": df_diverse["aem_head_type"]
        .value_counts()
        .to_dict(),
    }
    summary_path = f"{POSTPROC_DIR}/aem_summary_statistics_corrected.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print("Saved corrected summary statistics JSON:", summary_path)
    return root
