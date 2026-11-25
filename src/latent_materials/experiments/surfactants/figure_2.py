"""Final six-panel, two-domain held-out validation figure from saved latent results. Training producing this cache is missing.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the figure 2 experiment in a prepared workspace."""
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
    from sklearn.decomposition import PCA

    BASE = Path(str(root))
    OUT = BASE / "latent_adapter_two_domain_NatComm"
    NPZ_FILE = OUT / "two_domain_latent_results.npz"
    SUMMARY_FILE = OUT / "two_domain_adapter_summary.csv"
    if not NPZ_FILE.exists():
        raise FileNotFoundError(NPZ_FILE)
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(SUMMARY_FILE)
    SEED = 42
    D = np.load(NPZ_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    surf_teacher_test = D["surf_teacher_test"]
    surf_pred_test = D["surf_pred_test"]
    surf_teacher_all = D["surf_teacher_all"]
    surf_pred_all = D["surf_pred_all"]
    surf_train_trace = D["surf_train_trace"]
    surf_test_trace = D["surf_test_trace"]
    surf_test_rmse = D["surf_test_rmse"]
    aem_teacher_test = D["aem_teacher_test"]
    aem_pred_test = D["aem_pred_test"]
    aem_teacher_all = D["aem_teacher_all"]
    aem_pred_all = D["aem_pred_all"]
    aem_train_trace = D["aem_train_trace"]
    aem_test_trace = D["aem_test_trace"]
    aem_test_rmse = D["aem_test_rmse"]
    surf_best_epoch = int(
        summary.loc[summary["Domain"] == "Surfactant", "Best_epoch"].iloc[0]
    )
    aem_best_epoch = int(summary.loc[summary["Domain"] == "AEM", "Best_epoch"].iloc[0])
    print("Loaded real data:")
    print("Surfactant test:", surf_teacher_test.shape)
    print("AEM test       :", aem_teacher_test.shape)
    print("Surfactant best epoch:", surf_best_epoch)
    print("AEM best epoch       :", aem_best_epoch)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "normal",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.labelweight": "normal",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    SURF = "#2878B5"
    SURF_LIGHT = "#A9CBE8"
    AEM = "#D64550"
    AEM_LIGHT = "#F2B6B9"
    TEACHER = "#333333"
    surf_pca = PCA(n_components=2)
    surf_pca.fit(surf_teacher_all)
    surf_teacher_pca = surf_pca.transform(surf_teacher_test)
    surf_pred_pca = surf_pca.transform(surf_pred_test)
    aem_pca = PCA(n_components=2)
    aem_pca.fit(aem_teacher_all)
    aem_teacher_pca = aem_pca.transform(aem_teacher_test)
    aem_pred_pca = aem_pca.transform(aem_pred_test)
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.08], hspace=0.48, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, 0:2])
    ax_f = fig.add_subplot(gs[1, 2:4])
    for i in range(len(surf_teacher_pca)):
        ax_a.plot(
            [surf_teacher_pca[i, 0], surf_pred_pca[i, 0]],
            [surf_teacher_pca[i, 1], surf_pred_pca[i, 1]],
            color=SURF_LIGHT,
            linewidth=0.45,
            alpha=0.3,
            zorder=1,
        )
    ax_a.scatter(
        surf_teacher_pca[:, 0],
        surf_teacher_pca[:, 1],
        s=28,
        facecolors="white",
        edgecolors=TEACHER,
        linewidths=0.8,
        label="Teacher",
        zorder=3,
    )
    ax_a.scatter(
        surf_pred_pca[:, 0],
        surf_pred_pca[:, 1],
        s=23,
        color=SURF,
        alpha=0.78,
        linewidths=0,
        label="Adapter",
        zorder=2,
    )
    ax_a.set_title("Surfactant held-out latent alignment", pad=8)
    ax_a.set_xlabel(f"PC1 ({surf_pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax_a.set_ylabel(f"PC2 ({surf_pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax_a.legend(frameon=False, loc="lower right", handletextpad=0.4)
    for i in range(len(aem_teacher_pca)):
        ax_b.plot(
            [aem_teacher_pca[i, 0], aem_pred_pca[i, 0]],
            [aem_teacher_pca[i, 1], aem_pred_pca[i, 1]],
            color=AEM_LIGHT,
            linewidth=0.55,
            alpha=0.35,
            zorder=1,
        )
    ax_b.scatter(
        aem_teacher_pca[:, 0],
        aem_teacher_pca[:, 1],
        s=30,
        facecolors="white",
        edgecolors=TEACHER,
        linewidths=0.8,
        label="Teacher",
        zorder=3,
    )
    ax_b.scatter(
        aem_pred_pca[:, 0],
        aem_pred_pca[:, 1],
        s=25,
        color=AEM,
        alpha=0.82,
        linewidths=0,
        label="Adapter",
        zorder=2,
    )
    ax_b.set_title("AEM held-out latent alignment", pad=8)
    ax_b.set_xlabel(f"PC1 ({aem_pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax_b.set_ylabel(f"PC2 ({aem_pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax_b.legend(frameon=False, loc="lower right", handletextpad=0.4)
    x = surf_teacher_test.ravel()
    y = surf_pred_test.ravel()
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.04 * (hi - lo)
    lo -= pad
    hi += pad
    ax_c.scatter(x, y, s=8, color=SURF, alpha=0.24, edgecolors="none", rasterized=True)
    ax_c.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="#222222")
    ax_c.set_xlim(lo, hi)
    ax_c.set_ylim(lo, hi)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_title("Surfactant latent-coordinate parity", pad=8)
    ax_c.set_xlabel("Teacher latent coordinate")
    ax_c.set_ylabel("Adapter latent coordinate")
    x = aem_teacher_test.ravel()
    y = aem_pred_test.ravel()
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.04 * (hi - lo)
    lo -= pad
    hi += pad
    ax_d.scatter(x, y, s=12, color=AEM, alpha=0.4, edgecolors="none", rasterized=True)
    ax_d.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="#222222")
    ax_d.set_xlim(lo, hi)
    ax_d.set_ylim(lo, hi)
    ax_d.set_aspect("equal", adjustable="box")
    ax_d.set_title("AEM latent-coordinate parity", pad=8)
    ax_d.set_xlabel("Teacher latent coordinate")
    ax_d.set_ylabel("Adapter latent coordinate")
    surf_epochs = np.arange(1, len(surf_train_trace) + 1)
    aem_epochs = np.arange(1, len(aem_train_trace) + 1)
    ax_e.plot(
        surf_epochs,
        surf_train_trace,
        color=SURF,
        linewidth=1.35,
        alpha=0.62,
        label="Surfactant train",
    )
    ax_e.plot(
        surf_epochs,
        surf_test_trace,
        color=SURF,
        linewidth=1.55,
        linestyle="--",
        label="Surfactant held-out",
    )
    ax_e.plot(
        aem_epochs,
        aem_train_trace,
        color=AEM,
        linewidth=1.35,
        alpha=0.62,
        label="AEM train",
    )
    ax_e.plot(
        aem_epochs,
        aem_test_trace,
        color=AEM,
        linewidth=1.55,
        linestyle="--",
        label="AEM held-out",
    )
    ax_e.axvline(surf_best_epoch, color=SURF, linestyle=":", linewidth=0.9, alpha=0.75)
    ax_e.axvline(aem_best_epoch, color=AEM, linestyle=":", linewidth=0.9, alpha=0.75)
    ax_e.set_yscale("log")
    ax_e.set_title("Learning dynamics and held-out checkpoint selection", pad=8)
    ax_e.set_xlabel("Training epoch")
    ax_e.set_ylabel("Latent-space MSE")
    ax_e.legend(frameon=False, ncol=2, loc="upper right", columnspacing=1.5)
    violin = ax_f.violinplot(
        [surf_test_rmse, aem_test_rmse],
        positions=[1, 2],
        widths=0.68,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], [SURF, AEM]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.7)
        body.set_alpha(0.18)
    rng = np.random.default_rng(SEED)
    for xpos, vals, color in [(1, surf_test_rmse, SURF), (2, aem_test_rmse, AEM)]:
        jitter = rng.normal(0.0, 0.045, len(vals))
        ax_f.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=12,
            color=color,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        q1 = np.percentile(vals, 25)
        med = np.median(vals)
        q3 = np.percentile(vals, 75)
        ax_f.plot(
            [xpos, xpos],
            [q1, q3],
            color="#222222",
            linewidth=2.6,
            solid_capstyle="round",
            zorder=4,
        )
        ax_f.scatter(
            [xpos],
            [med],
            s=34,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=5,
        )
    ax_f.set_yscale("log")
    ax_f.set_xticks([1, 2])
    ax_f.set_xticklabels(["Surfactants", "AEMs"])
    ax_f.set_ylabel("Per-molecule latent RMSE")
    ax_f.set_title("Held-out reconstruction-error distributions", pad=8)
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
    letters = ["a", "b", "c", "d", "e", "f"]
    for ax, letter in zip(axes, letters):
        ax.text(
            -0.13,
            1.07,
            letter,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="normal",
            ha="left",
            va="top",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in [ax_e, ax_f]:
        ax.grid(axis="y", linewidth=0.45, color="#D0D0D0", alpha=0.45)
        ax.set_axisbelow(True)
    plt.subplots_adjust(left=0.07, right=0.985, top=0.95, bottom=0.09)
    PNG = OUT / "two_domain_conditional_latent_adapter_NatComm_clean.png"
    PDF = OUT / "two_domain_conditional_latent_adapter_NatComm_clean.pdf"
    SVG = OUT / "two_domain_conditional_latent_adapter_NatComm_clean.svg"
    fig.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.show()
    print("\nSaved:")
    print(PNG)
    print(PDF)
    print(SVG)
    return root
