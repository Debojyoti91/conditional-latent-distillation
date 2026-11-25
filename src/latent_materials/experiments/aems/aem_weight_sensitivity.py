"""125-member production score sensitivity and Figure S8. Historical equation number 14 is manuscript Eq. 10.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the aem weight sensitivity experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    ROOT = Path(str(root / "IEMS_GEN_files"))
    INPUT = ROOT / "postprocessing_AEM_corrected" / "aem_candidates_ranked.csv"
    OUT = ROOT / "reviewer_defense_AEM_Eq14_sensitivity"
    OUT.mkdir(parents=True, exist_ok=True)
    SEED = 20260911
    N_DRAWS = 10000
    PERTURBATION = 0.2
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(INPUT).copy()
    print("=" * 100)
    print("AEM EQ. 14 WEIGHT-SENSITIVITY ANALYSIS")
    print("=" * 100)
    print("Input:", INPUT)
    print("Candidates:", len(df))
    print("Columns:", len(df.columns))
    possible_cond_cols = [
        "predicted_conductivity_oldGNN_2",
        "predicted_conductivity_oldGNN",
        "Predicted_conductivity",
        "predicted_conductivity",
        "Conductivity",
        "Anion_Conductivity_Scm",
        "Anion_Conductivity_pred_Scm",
    ]
    cond_col = None
    for c in possible_cond_cols:
        if c in df.columns:
            cond_col = c
            break
    print("\nConductivity column in production ranked table:", cond_col)
    if cond_col is None:
        print(
            "→ IMPORTANT: the executed 125-candidate production score contains NO conductivity contribution."
        )
        print(
            "→ Sensitivity below therefore tests the actual structural AEM composite score used for this production ranking."
        )

    def bool01(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, str):
            return float(x.strip().lower() in {"true", "1", "yes", "y"})
        return float(bool(x))

    ncat = (
        pd.to_numeric(df["cationic_site_count"], errors="coerce").fillna(0).to_numpy()
    )
    Q_CATION_PRESENT = (ncat >= 1).astype(float)
    Q_ONE_CATION = (ncat == 1).astype(float)
    Q_TWO_CATIONS = (ncat == 2).astype(float)
    iec = pd.to_numeric(df["IEC_proxy_mmol_g"], errors="coerce").to_numpy()
    Q_IEC = np.zeros(len(df), dtype=float)
    core = np.isfinite(iec) & (iec >= 1.0) & (iec <= 4.0)
    shoulder = np.isfinite(iec) & (
        (iec >= 0.5) & (iec < 1.0) | (iec > 4.0) & (iec <= 6.0)
    )
    Q_IEC[core] = 1.0
    Q_IEC[shoulder] = 0.5
    sa = pd.to_numeric(df["SA_score"], errors="coerce").to_numpy()
    Q_SA = np.zeros(len(df), dtype=float)
    finite = np.isfinite(sa)
    Q_SA[finite] = 1.0 - np.minimum(sa[finite] / 10.0, 1.0)
    mw = pd.to_numeric(df["MW"], errors="coerce").to_numpy()
    Q_MW = np.zeros(len(df), dtype=float)
    finite = np.isfinite(mw)
    Q_MW[finite] = np.maximum(0.0, 1.0 - np.abs(mw[finite] - 450.0) / 400.0)
    logp = pd.to_numeric(df["LogP"], errors="coerce").to_numpy()
    Q_LOGP = np.zeros(len(df), dtype=float)
    finite = np.isfinite(logp)
    Q_LOGP[finite] = np.maximum(0.0, 1.0 - np.abs(logp[finite] - 2.5) / 4.0)
    Q_COND = np.zeros(len(df), dtype=float)
    if cond_col is not None:
        cond = pd.to_numeric(df[cond_col], errors="coerce")
        valid_cond = cond.dropna()
        if len(valid_cond):
            cond_low = float(valid_cond.quantile(0.05))
            cond_high = float(valid_cond.quantile(0.95))
            if cond_high != cond_low:
                Q_COND = np.clip(
                    (cond.fillna(cond_low).to_numpy() - cond_low)
                    / (cond_high - cond_low),
                    0,
                    1,
                )
            print("Conductivity normalization:", cond_low, "to", cond_high)
    peroxy = np.array([bool01(x) for x in df["has_peroxy"]])
    cation_head = np.array([bool01(x) for x in df["has_cationic_head"]])
    fragments = (
        pd.to_numeric(df["fragment_count"], errors="coerce").fillna(1).to_numpy()
    )
    PENALTY = (
        -0.5 * (peroxy == 1) - 0.5 * (cation_head == 0) - 0.1 * (fragments > 1)
    ).astype(float)
    weights = {
        "conductivity": 0.35,
        "cation_present": 0.15,
        "one_cation": 0.05,
        "two_cations": 0.04,
        "IEC": 0.2,
        "SA": 0.15,
        "MW": 0.1,
        "LogP": 0.05,
    }
    qualities = {
        "conductivity": Q_COND,
        "cation_present": Q_CATION_PRESENT,
        "one_cation": Q_ONE_CATION,
        "two_cations": Q_TWO_CATIONS,
        "IEC": Q_IEC,
        "SA": Q_SA,
        "MW": Q_MW,
        "LogP": Q_LOGP,
    }
    active_weights = {
        k: v
        for k, v in weights.items()
        if not (k == "conductivity" and cond_col is None)
    }

    def calculate_score(weight_dict):
        score = PENALTY.copy()
        for name, w in weight_dict.items():
            score += w * qualities[name]
        return score

    baseline_score = calculate_score(active_weights)
    stored_score = pd.to_numeric(df["aem_composite_score"], errors="coerce").to_numpy()
    difference = baseline_score - stored_score
    max_abs_error = np.nanmax(np.abs(difference))
    mean_abs_error = np.nanmean(np.abs(difference))
    print("\n" + "=" * 100)
    print("BASELINE SCORE RECONSTRUCTION")
    print("=" * 100)
    print(f"Maximum absolute error: {max_abs_error:.12g}")
    print(f"Mean absolute error:    {mean_abs_error:.12g}")
    if max_abs_error > 1e-08:
        check = pd.DataFrame(
            {
                "stored": stored_score,
                "reconstructed": baseline_score,
                "difference": difference,
            }
        )
        check.to_csv(OUT / "baseline_score_reconstruction_FAILED.csv", index=False)
        raise RuntimeError(
            "\nSTOPPING: Eq. 14 reconstruction does not exactly match the saved production scores.\nDo not run sensitivity until this is resolved."
        )
    print("✅ Exact reconstruction of saved production score.")
    baseline_order = np.argsort(-baseline_score, kind="mergesort")
    baseline_rank = np.empty(len(df), dtype=int)
    baseline_rank[baseline_order] = np.arange(len(df)) + 1
    baseline_top10 = set(baseline_order[:10])
    baseline_top20 = set(baseline_order[:20])
    weight_names = list(active_weights.keys())
    n = len(df)
    spearman_values = np.empty(N_DRAWS)
    top10_overlap = np.empty(N_DRAWS, dtype=int)
    top20_overlap = np.empty(N_DRAWS, dtype=int)
    rank_sum = np.zeros(n, dtype=float)
    rank_sq_sum = np.zeros(n, dtype=float)
    top10_count = np.zeros(n, dtype=int)
    top20_count = np.zeros(n, dtype=int)
    draw_weight_records = []
    for draw in range(N_DRAWS):
        perturbed = {}
        rec = {"draw": draw + 1}
        for name in weight_names:
            multiplier = rng.uniform(1.0 - PERTURBATION, 1.0 + PERTURBATION)
            w = active_weights[name] * multiplier
            perturbed[name] = w
            rec[f"weight_{name}"] = w
        score = calculate_score(perturbed)
        order = np.argsort(-score, kind="mergesort")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n) + 1
        rho, _ = spearmanr(baseline_rank, ranks)
        spearman_values[draw] = rho
        current_top10 = set(order[:10])
        current_top20 = set(order[:20])
        top10_overlap[draw] = len(baseline_top10 & current_top10)
        top20_overlap[draw] = len(baseline_top20 & current_top20)
        rank_sum += ranks
        rank_sq_sum += ranks.astype(float) ** 2
        top10_count[list(current_top10)] += 1
        top20_count[list(current_top20)] += 1
        draw_weight_records.append(rec)
    mean_rank = rank_sum / N_DRAWS
    rank_variance = rank_sq_sum / N_DRAWS - mean_rank**2
    rank_sd = np.sqrt(np.maximum(rank_variance, 0))
    candidate_stability = df[
        ["SMILES", "canonical_SMILES", "aem_composite_score"]
    ].copy()
    candidate_stability["baseline_rank"] = baseline_rank
    candidate_stability["mean_perturbed_rank"] = mean_rank
    candidate_stability["sd_perturbed_rank"] = rank_sd
    candidate_stability["top10_probability"] = top10_count / N_DRAWS
    candidate_stability["top20_probability"] = top20_count / N_DRAWS
    candidate_stability = candidate_stability.sort_values("baseline_rank").reset_index(
        drop=True
    )
    candidate_stability.to_csv(
        OUT / "AEM_Eq14_candidate_rank_stability.csv", index=False
    )
    draw_summary = pd.DataFrame(
        {
            "draw": np.arange(1, N_DRAWS + 1),
            "spearman_rho": spearman_values,
            "top10_overlap": top10_overlap,
            "top20_overlap": top20_overlap,
        }
    )
    draw_summary.to_csv(OUT / "AEM_Eq14_perturbation_draw_summary.csv", index=False)
    pd.DataFrame(draw_weight_records).to_csv(
        OUT / "AEM_Eq14_perturbed_weights.csv", index=False
    )

    def q(x, p):
        return float(np.percentile(x, p))

    summary = pd.DataFrame(
        [
            {
                "N_candidates": n,
                "N_perturbations": N_DRAWS,
                "weight_variation_percent": 100 * PERTURBATION,
                "spearman_mean": np.mean(spearman_values),
                "spearman_median": np.median(spearman_values),
                "spearman_2.5pct": q(spearman_values, 2.5),
                "spearman_97.5pct": q(spearman_values, 97.5),
                "top10_overlap_mean": np.mean(top10_overlap),
                "top10_overlap_median": np.median(top10_overlap),
                "top10_overlap_2.5pct": q(top10_overlap, 2.5),
                "top10_overlap_97.5pct": q(top10_overlap, 97.5),
                "top20_overlap_mean": np.mean(top20_overlap),
                "top20_overlap_median": np.median(top20_overlap),
                "top20_overlap_2.5pct": q(top20_overlap, 2.5),
                "top20_overlap_97.5pct": q(top20_overlap, 97.5),
            }
        ]
    )
    summary.to_csv(OUT / "AEM_Eq14_sensitivity_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.hist(spearman_values, bins=35, edgecolor="black")
    ax.set_xlabel("Spearman rank correlation with baseline")
    ax.set_ylabel("Perturbation count")
    fig.tight_layout()
    fig.savefig(OUT / "AEM_Eq14_Spearman_stability.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "AEM_Eq14_Spearman_stability.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    bins = np.arange(top10_overlap.min() - 0.5, top10_overlap.max() + 1.5, 1)
    ax.hist(top10_overlap, bins=bins, edgecolor="black")
    ax.set_xlabel("Number of baseline top-10 candidates retained")
    ax.set_ylabel("Perturbation count")
    ax.set_xticks(np.arange(top10_overlap.min(), top10_overlap.max() + 1))
    fig.tight_layout()
    fig.savefig(OUT / "AEM_Eq14_top10_overlap.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "AEM_Eq14_top10_overlap.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(
        candidate_stability["baseline_rank"],
        candidate_stability["mean_perturbed_rank"],
        s=25,
    )
    lims = [1, n]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Baseline rank")
    ax.set_ylabel("Mean rank under weight perturbation")
    fig.tight_layout()
    fig.savefig(
        OUT / "AEM_Eq14_baseline_vs_perturbed_rank.png", dpi=600, bbox_inches="tight"
    )
    fig.savefig(OUT / "AEM_Eq14_baseline_vs_perturbed_rank.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("\n" + "=" * 100)
    print("EQ. 14 SENSITIVITY RESULTS")
    print("=" * 100)
    print("\nActive positive weights:")
    for k, v in active_weights.items():
        print(f"  {k:20s}: {v:.3f}")
    print(
        f"\nSpearman rho:\n  mean   = {np.mean(spearman_values):.4f}\n  median = {np.median(spearman_values):.4f}\n  95% interval = [{q(spearman_values, 2.5):.4f}, {q(spearman_values, 97.5):.4f}]"
    )
    print(
        f"\nTop-10 overlap:\n  mean   = {np.mean(top10_overlap):.2f}/10\n  median = {np.median(top10_overlap):.0f}/10\n  95% interval = [{q(top10_overlap, 2.5):.0f}, {q(top10_overlap, 97.5):.0f}]/10"
    )
    print(
        f"\nTop-20 overlap:\n  mean   = {np.mean(top20_overlap):.2f}/20\n  median = {np.median(top20_overlap):.0f}/20\n  95% interval = [{q(top20_overlap, 2.5):.0f}, {q(top20_overlap, 97.5):.0f}]/20"
    )
    print("\nBaseline top-10 candidate stability:")
    print(
        candidate_stability[candidate_stability["baseline_rank"] <= 10][
            [
                "baseline_rank",
                "aem_composite_score",
                "mean_perturbed_rank",
                "sd_perturbed_rank",
                "top10_probability",
                "top20_probability",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\n" + "=" * 100)
    print("SAVED FILES")
    print("=" * 100)
    for p in sorted(OUT.iterdir()):
        print(p)
    print("\nComplete.")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 9.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    TEAL = "#167D8D"
    TEAL_DARK = "#0B5964"
    TEAL_LIGHT = "#B9DDE1"
    OCHRE = "#C58B2A"
    OCHRE_DARK = "#8C6117"
    OCHRE_LIGHT = "#EAD5AC"
    GRAPHITE = "#343434"
    MID_GREY = "#777777"
    LIGHT_GREY = "#D9D9D9"
    GRID_GREY = "#E6E6E6"
    rho_median = float(np.median(spearman_values))
    rho_low = float(np.percentile(spearman_values, 2.5))
    rho_high = float(np.percentile(spearman_values, 97.5))
    top10_median = float(np.median(top10_overlap))
    top10_mean = float(np.mean(top10_overlap))
    rank_corr, _ = spearmanr(
        candidate_stability["baseline_rank"], candidate_stability["mean_perturbed_rank"]
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 4.05),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08], "wspace": 0.34},
    )
    ax_a, ax_b, ax_c = axes
    ax_a.hist(
        spearman_values,
        bins=35,
        color=TEAL,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.92,
    )
    ax_a.axvline(rho_median, color=GRAPHITE, linestyle="--", linewidth=1.15, zorder=5)
    ax_a.axvspan(rho_low, rho_high, color=TEAL_LIGHT, alpha=0.28, linewidth=0, zorder=0)
    ax_a.set_xlabel("Spearman rank correlation, $\\rho$")
    ax_a.set_ylabel("Perturbation count")
    ax_a.set_title("Global rank-order stability", pad=9)
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_a.text(
        0.04,
        0.95,
        f"Median $\\rho$ = {rho_median:.3f}\n95% interval = [{rho_low:.3f}, {rho_high:.3f}]",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=GRAPHITE,
    )
    bins = np.arange(top10_overlap.min() - 0.5, top10_overlap.max() + 1.5, 1)
    ax_b.hist(
        top10_overlap,
        bins=bins,
        color=OCHRE,
        edgecolor="white",
        linewidth=0.65,
        alpha=0.92,
    )
    ax_b.axvline(top10_median, color=GRAPHITE, linestyle="--", linewidth=1.15, zorder=5)
    ax_b.set_xlabel("Baseline top-10 candidates retained")
    ax_b.set_ylabel("Perturbation count")
    ax_b.set_title("Top-10 candidate retention", pad=9)
    ax_b.set_xticks(np.arange(top10_overlap.min(), top10_overlap.max() + 1))
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_b.text(
        0.04,
        0.95,
        f"Median retained = {top10_median:.0f}/10\nMean retained = {top10_mean:.2f}/10",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=GRAPHITE,
    )
    baseline_x = candidate_stability["baseline_rank"].to_numpy()
    perturbed_y = candidate_stability["mean_perturbed_rank"].to_numpy()
    rank_sd_plot = candidate_stability["sd_perturbed_rank"].to_numpy()
    if np.nanmax(rank_sd_plot) > 0:
        point_sizes = 18 + 42 * rank_sd_plot / np.nanmax(rank_sd_plot)
    else:
        point_sizes = np.full(len(rank_sd_plot), 24.0)
    ax_c.scatter(
        baseline_x,
        perturbed_y,
        s=point_sizes,
        color=TEAL_DARK,
        alpha=0.68,
        edgecolors="white",
        linewidths=0.35,
        rasterized=True,
        zorder=3,
    )
    lims = [1, n]
    ax_c.plot(lims, lims, linestyle="--", linewidth=1.0, color=GRAPHITE, zorder=1)
    ax_c.set_xlim(lims)
    ax_c.set_ylim(lims)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlabel("Baseline candidate rank")
    ax_c.set_ylabel("Mean rank after weight perturbation")
    ax_c.set_title("Candidate-level rank robustness", pad=9)
    ax_c.text(
        0.04,
        0.95,
        f"Rank correlation = {rank_corr:.3f}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=GRAPHITE,
    )
    top10_mask = candidate_stability["baseline_rank"].to_numpy() <= 10
    ax_c.scatter(
        baseline_x[top10_mask],
        perturbed_y[top10_mask],
        s=point_sizes[top10_mask] + 18,
        facecolors="none",
        edgecolors=OCHRE_DARK,
        linewidths=1.15,
        zorder=5,
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRAPHITE)
        ax.spines["bottom"].set_color(GRAPHITE)
        ax.tick_params(axis="both", colors=GRAPHITE, direction="out")
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
    for ax, letter in zip(axes, ["a", "b", "c"]):
        ax.text(
            -0.13,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=12.5,
            fontweight="bold",
            color=GRAPHITE,
            ha="left",
            va="top",
        )
    fig.suptitle(
        "AEM candidate rankings are robust to perturbation of composite-score weights",
        fontsize=12.5,
        fontweight="normal",
        color=GRAPHITE,
        y=1.02,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.82, wspace=0.34)
    PNG_COMPOSITE = OUT / "AEM_Eq14_weight_sensitivity_3panel.png"
    PDF_COMPOSITE = OUT / "AEM_Eq14_weight_sensitivity_3panel.pdf"
    SVG_COMPOSITE = OUT / "AEM_Eq14_weight_sensitivity_3panel.svg"
    fig.savefig(PNG_COMPOSITE, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_COMPOSITE, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_COMPOSITE, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    print("\nComposite publication figure saved:")
    print(PNG_COMPOSITE)
    print(PDF_COMPOSITE)
    print(SVG_COMPOSITE)
    return root
