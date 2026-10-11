"""Final two-panel conditioning-ablation figure.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the figure s11 experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = (
        str(root / "IEMS_GEN_files/AEM/CVAE") if "surfactant" == "aem" else str(root)
    )
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    BASE_DIR = Path(str(root))
    ABLATION_DIR = BASE_DIR / "conditioning_ablation_CORRECTED"
    FIGURE_DIR = BASE_DIR / "conditioning_ablation_TRANSFER_FINAL"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_RESULTS_FILE = ABLATION_DIR / "main_ablation_by_seed.csv"
    SHUFFLE_SUMMARY_FILE = ABLATION_DIR / "shuffled_property_summary.csv"
    PREDICTION_FILE = ABLATION_DIR / "ablation_test_predictions.npz"
    required_files = [MAIN_RESULTS_FILE, SHUFFLE_SUMMARY_FILE, PREDICTION_FILE]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing:\n{path}")
    print("[PASS] Required files found.")
    main_results = pd.read_csv(MAIN_RESULTS_FILE)
    shuffle_summary = pd.read_csv(SHUFFLE_SUMMARY_FILE)
    prediction_data = np.load(PREDICTION_FILE, allow_pickle=False)
    print("\nLoaded:")
    print("main_results:", main_results.shape)
    print("shuffle_summary:", shuffle_summary.shape)
    required_main_columns = ["ablation", "seed", "test_mse"]
    for col in required_main_columns:
        if col not in main_results.columns:
            raise RuntimeError(f"Missing main-results column: {col}")
    required_prediction_arrays = ["teacher_test", "Full_complete_seed_mean"]
    for key in required_prediction_arrays:
        if key not in prediction_data.files:
            raise RuntimeError(f"Missing prediction array: {key}")
    installed_fonts = {font.name for font in fm.fontManager.ttflist}
    preferred_fonts = [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "Nimbus Sans",
        "DejaVu Sans",
    ]
    PUB_FONT = "DejaVu Sans"
    for candidate in preferred_fonts:
        if candidate in installed_fonts:
            PUB_FONT = candidate
            break
    print("Publication font:", PUB_FONT)
    plt.rcParams.update(
        {
            "font.family": PUB_FONT,
            "font.size": 7.5,
            "font.weight": "normal",
            "axes.labelsize": 8.0,
            "axes.labelweight": "normal",
            "axes.titlesize": 8.2,
            "axes.titleweight": "normal",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    ORDER = [
        "Structure_original",
        "Full_original",
        "Structure_complete",
        "Full_complete",
        "Property_only",
    ]
    DISPLAY = {
        "Structure_original": "Original\nstructure",
        "Full_original": "Original structure\n+ properties",
        "Structure_complete": "Complete\nstructure",
        "Full_complete": "Complete structure\n+ properties",
        "Property_only": "Properties\nonly",
    }
    COLORS = {
        "Structure_original": "#9FC5B5",
        "Full_original": "#8FAFD0",
        "Structure_complete": "#DCA39C",
        "Full_complete": "#A8C895",
        "Property_only": "#B5A5C7",
    }
    DARK = "#303030"
    GRID = "#E7E7E7"
    SHUFFLE_GRAY = "#B8B8B8"
    AUTHENTIC = "#A75169"
    MSE_SCALE = 10000.0
    panel_a = main_results[main_results["ablation"].isin(ORDER)].copy()
    panel_a["scaled_mse"] = panel_a["test_mse"].astype(float) * MSE_SCALE
    summary_a = (
        panel_a.groupby("ablation")["scaled_mse"].agg(["mean", "std"]).reindex(ORDER)
    )
    teacher_test = np.asarray(prediction_data["teacher_test"], dtype=np.float64)
    full_complete_ensemble = np.asarray(
        prediction_data["Full_complete_seed_mean"], dtype=np.float64
    )
    authentic_mse = float(np.mean((full_complete_ensemble - teacher_test) ** 2))
    authentic_mse_scaled = authentic_mse * MSE_SCALE
    shuffle_mse_scaled = (
        shuffle_summary["ensemble_test_mse"].astype(float).to_numpy() * MSE_SCALE
    )
    print("\nAuthentic ensemble MSE:", authentic_mse_scaled, "×10^-4")
    print("Mean shuffled ensemble MSE:", float(np.mean(shuffle_mse_scaled)), "×10^-4")
    MM_TO_IN = 1.0 / 25.4
    FIG_WIDTH_MM = 180
    FIG_HEIGHT_MM = 94
    fig = plt.figure(
        figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN), facecolor="white"
    )
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.7, 1.0],
        left=0.075,
        right=0.985,
        bottom=0.22,
        top=0.8,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    fig.suptitle(
        "Latent-reconstruction performance under surfactant conditioning ablations",
        x=0.53,
        y=0.965,
        fontsize=10.5,
        fontweight="normal",
        ha="center",
        va="top",
    )
    x = np.arange(len(ORDER))
    means = summary_a["mean"].to_numpy()
    sds = summary_a["std"].to_numpy()
    bar_colors = [COLORS[name] for name in ORDER]
    ax_a.bar(
        x,
        means,
        width=0.66,
        color=bar_colors,
        edgecolor="#8C8C8C",
        linewidth=0.65,
        zorder=2,
    )
    ax_a.errorbar(
        x,
        means,
        yerr=sds,
        fmt="none",
        ecolor=DARK,
        elinewidth=0.9,
        capsize=2.6,
        capthick=0.8,
        zorder=4,
    )
    seed_values = sorted(panel_a["seed"].unique())
    seed_offsets = np.linspace(-0.09, 0.09, len(seed_values))
    for model_index, name in enumerate(ORDER):
        subset = panel_a[panel_a["ablation"] == name].sort_values("seed")
        values = subset["scaled_mse"].to_numpy()
        ax_a.scatter(
            np.full(len(values), model_index) + seed_offsets[: len(values)],
            values,
            s=12,
            facecolor="#444444",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.72,
            zorder=5,
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([DISPLAY[name] for name in ORDER])
    ax_a.set_ylabel("Test latent reconstruction MSE ($\\times10^{-4}$)")
    ax_a.set_ylim(0, (means + sds).max() * 1.14)
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_a.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax_a.grid(axis="x", visible=False)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.text(
        -0.055,
        1.055,
        "a",
        transform=ax_a.transAxes,
        fontsize=9.5,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    ax_a.text(
        0.0,
        1.055,
        "Conditioning representations",
        transform=ax_a.transAxes,
        fontsize=8.4,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    rng = np.random.default_rng(20260926)
    shuffle_x = rng.uniform(-0.07, 0.07, len(shuffle_mse_scaled))
    ax_b.scatter(
        shuffle_x,
        shuffle_mse_scaled,
        s=22,
        facecolor=SHUFFLE_GRAY,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
        zorder=3,
    )
    shuffle_mean = float(np.mean(shuffle_mse_scaled))
    shuffle_sd = float(np.std(shuffle_mse_scaled, ddof=1))
    ax_b.errorbar(
        0,
        shuffle_mean,
        yerr=shuffle_sd,
        fmt="_",
        markersize=14,
        markeredgewidth=1.4,
        color=DARK,
        ecolor=DARK,
        elinewidth=1.0,
        capsize=2.7,
        capthick=0.8,
        zorder=5,
    )
    ax_b.scatter(
        1,
        authentic_mse_scaled,
        s=43,
        facecolor=AUTHENTIC,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ax_b.plot(
        [0.89, 1.11],
        [authentic_mse_scaled, authentic_mse_scaled],
        color=AUTHENTIC,
        linewidth=1.1,
        zorder=4,
    )
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Shuffled\nproperty pairing", "Authentic\nproperty pairing"])
    ax_b.set_ylabel("Ensemble test MSE ($\\times10^{-4}$)")
    ax_b.set_xlim(-0.34, 1.34)
    b_all = np.concatenate([shuffle_mse_scaled, np.array([authentic_mse_scaled])])
    b_min = float(b_all.min())
    b_max = float(b_all.max())
    b_span = b_max - b_min
    b_pad = 0.15 * b_span
    ax_b.set_ylim(b_min - b_pad, b_max + b_pad)
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_b.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_b.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax_b.grid(axis="x", visible=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.text(
        -0.055,
        1.055,
        "b",
        transform=ax_b.transAxes,
        fontsize=9.5,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    ax_b.text(
        0.0,
        1.055,
        "Property–molecule correspondence",
        transform=ax_b.transAxes,
        fontsize=8.4,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    ax_b.text(
        0.02,
        0.96,
        f"{len(shuffle_mse_scaled)} shuffled assignments",
        transform=ax_b.transAxes,
        fontsize=5.8,
        fontweight="normal",
        color="#666666",
        ha="left",
        va="top",
    )
    for ax in [ax_a, ax_b]:
        ax.tick_params(direction="out")
        ax.set_axisbelow(True)
        ax.spines["left"].set_color(DARK)
        ax.spines["bottom"].set_color(DARK)
    PDF_OUT = FIGURE_DIR / "conditioning_ablation_NATURE_FINAL_v2.pdf"
    SVG_OUT = FIGURE_DIR / "conditioning_ablation_NATURE_FINAL_v2.svg"
    PNG_OUT = FIGURE_DIR / "conditioning_ablation_NATURE_FINAL_v2_600dpi.png"
    TIFF_OUT = FIGURE_DIR / "conditioning_ablation_NATURE_FINAL_v2_600dpi.tiff"
    fig.savefig(
        PDF_OUT, format="pdf", bbox_inches="tight", pad_inches=0.035, facecolor="white"
    )
    fig.savefig(
        SVG_OUT, format="svg", bbox_inches="tight", pad_inches=0.035, facecolor="white"
    )
    fig.savefig(
        PNG_OUT,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
    )
    fig.savefig(
        TIFF_OUT,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.show()
    print("\nSaved:")
    print(PDF_OUT)
    print(SVG_OUT)
    print(PNG_OUT)
    print(TIFF_OUT)
    return root
