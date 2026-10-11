"""Final gallery plausibility filter and six representative NN/BO structures for the extended-data gallery.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the extended data aem experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    import os

    NNBO_FIG_DIR = os.path.join(DATA_DIR, "nn_bo_comparison")
    os.makedirs(NNBO_FIG_DIR, exist_ok=True)
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image
    from io import BytesIO
    from rdkit import Chem, DataStructs
    from rdkit.Chem import (
        AllChem,
        rdFingerprintGenerator,
        rdMolDescriptors,
        Descriptors,
    )
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

    NN_GALLERY_CSV = os.path.join(DATA_DIR, "aem_augmented_ranked_candidates_NN.csv")
    BO_GALLERY_CSV = os.path.join(DATA_DIR, "aem_novel_candidates_clean_BO.csv")
    N_PER_PANEL = 3
    RANDOM_SEED = 42
    MOL_IMG_PX = 900
    FIG_DPI = 600
    GALLERY_OUT_PNG = os.path.join(
        NNBO_FIG_DIR, "AEM_Figure_representative_NN_vs_BO_candidates.png"
    )
    GALLERY_OUT_PDF = os.path.join(
        NNBO_FIG_DIR, "AEM_Figure_representative_NN_vs_BO_candidates.pdf"
    )
    FIG_TITLE = "Representative novel AEM candidates spanning diverse chemical space"
    PANEL_A_TITLE = "NN-generated candidates"
    PANEL_B_TITLE = "BO-generated candidates"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "savefig.dpi": FIG_DPI,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    def load_valid_smiles(path):
        gdf = pd.read_csv(path)
        smi_col = next(
            (c for c in ["SMILES", "smiles", "Canonical_SMILES"] if c in gdf.columns),
            None,
        )
        if smi_col is None:
            raise ValueError(
                f"No SMILES-like column found in {path}: {list(gdf.columns)}"
            )
        out = []
        for s in gdf[smi_col].dropna().astype(str).tolist():
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                out.append(Chem.MolToSmiles(mol, canonical=True))
        return list(dict.fromkeys(out))

    def structure_is_plausible(smi, max_ring_size=9, max_mw=1200.0):
        """Gallery-only plausibility filter (see module docstring above)."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        if any((a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms())):
            return False
        if rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0:
            return False
        if rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0:
            return False
        ri = mol.GetRingInfo()
        if any((len(r) > max_ring_size for r in ri.AtomRings())):
            return False
        if Descriptors.MolWt(mol) > max_mw:
            return False
        return True

    gallery_nn_smiles_all = load_valid_smiles(NN_GALLERY_CSV)
    gallery_bo_smiles_all = load_valid_smiles(BO_GALLERY_CSV)
    gallery_nn_smiles = [s for s in gallery_nn_smiles_all if structure_is_plausible(s)]
    gallery_bo_smiles = [s for s in gallery_bo_smiles_all if structure_is_plausible(s)]
    print(
        f"NN pool: {len(gallery_nn_smiles_all)} -> {len(gallery_nn_smiles)} after plausibility filter ({len(gallery_nn_smiles_all) - len(gallery_nn_smiles)} excluded)"
    )
    print(
        f"BO pool: {len(gallery_bo_smiles_all)} -> {len(gallery_bo_smiles)} after plausibility filter ({len(gallery_bo_smiles_all) - len(gallery_bo_smiles)} excluded)"
    )

    def diverse_pick(smiles_list, n, seed=RANDOM_SEED):
        if len(smiles_list) <= n:
            return smiles_list
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        mfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [mfp.GetFingerprint(m) for m in mols]

        def dist(i, j):
            return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

        idxs = list(MaxMinPicker().LazyPick(dist, len(fps), n, seed=seed))
        return [smiles_list[i] for i in idxs]

    gallery_nn_pick = diverse_pick(gallery_nn_smiles, N_PER_PANEL)
    gallery_bo_pick = diverse_pick(gallery_bo_smiles, N_PER_PANEL)
    print("NN picked:", gallery_nn_pick)
    print("BO picked:", gallery_bo_pick)

    def render_mol(smiles, size=MOL_IMG_PX):
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.4
        opts.baseFontSize = 0.7
        opts.padding = 0.02
        opts.clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(BytesIO(drawer.GetDrawingText()))

    gallery_nn_imgs = [render_mol(s) for s in gallery_nn_pick]
    gallery_bo_imgs = [render_mol(s) for s in gallery_bo_pick]
    fig = plt.figure(figsize=(14, 9))
    outer = gridspec.GridSpec(
        nrows=2,
        ncols=1,
        figure=fig,
        hspace=0.1,
        left=0.025,
        right=0.99,
        top=0.91,
        bottom=0.02,
    )
    fig.suptitle(FIG_TITLE, fontsize=17, y=0.97)

    def add_panel(panel_gs, imgs, title, panel_letter):
        inner = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=N_PER_PANEL,
            subplot_spec=panel_gs,
            height_ratios=[1, 18],
            hspace=0.04,
            wspace=0.0,
        )
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis("off")
        title_ax.text(
            0.0,
            0.5,
            f"{panel_letter}   {title}",
            transform=title_ax.transAxes,
            fontsize=14,
            ha="left",
            va="center",
            color="black",
        )
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(inner[1, i])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.0)

    add_panel(outer[0], gallery_nn_imgs, PANEL_A_TITLE, "a")
    add_panel(outer[1], gallery_bo_imgs, PANEL_B_TITLE, "b")
    fig.savefig(GALLERY_OUT_PNG, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(GALLERY_OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[SAVED] {GALLERY_OUT_PNG}")
    print(f"[SAVED] {GALLERY_OUT_PDF}")
    pd.DataFrame(
        {
            "branch": ["NN"] * len(gallery_nn_pick) + ["BO"] * len(gallery_bo_pick),
            "SMILES": gallery_nn_pick + gallery_bo_pick,
        }
    ).to_csv(
        os.path.join(NNBO_FIG_DIR, "Extended_Data_Figure_2_selected_molecules.csv"),
        index=False,
    )
    return root
