"""Template-based screening after production ranking; not experimental synthesis validation.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import display, require_dependencies


def run(workspace):
    """Execute the aem synthesizability experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    "\nTemplate-based / retrosynthesis-style synthesizability screening for AEM candidates.\n\nInput:\n    /content/drive/MyDrive/surfactants/IEMS_GEN_files/postprocessing_AEM_corrected/AEM_TOP_100_candidates.csv\n\nOutputs:\n    /content/drive/MyDrive/surfactants/IEMS_GEN_files/postprocessing_AEM_corrected/synthesizability_screening_v2/\n\nMain features:\n    - Published-style structural complexity logic using RDKit descriptors\n    - AEM-specific cation classification\n    - Rule-based unstable/reactive motif filtering\n    - Retrosynthesis-style disconnection labels:\n        benzyl quaternization candidate\n        alkyl quaternization candidate\n        imidazolium alkylation candidate\n        aromatic N-alkylation candidate\n        aryl ether coupling candidate\n        Williamson ether candidate\n        no obvious disconnection\n    - Route hints\n    - Synthesizability class: easy / moderate / difficult / reject\n    - Final pre-MD ranking score\n"
    import os
    import json
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rdkit import Chem
    from rdkit.Chem import Draw

    warnings.filterwarnings("ignore")
    print("Dependencies loaded successfully.")
    BASE_DIR = str(root / "IEMS_GEN_files/postprocessing_AEM_corrected")
    INPUT_CSV = f"{BASE_DIR}/AEM_TOP_100_candidates.csv"
    OUT_DIR = f"{BASE_DIR}/synthesizability_screening_v2"
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Input:", INPUT_CSV)
    print("Output directory:", OUT_DIR)
    df = pd.read_csv(INPUT_CSV)
    if "canonical_SMILES" in df.columns:
        SMILES_COL = "canonical_SMILES"
    elif "SMILES" in df.columns:
        SMILES_COL = "SMILES"
    else:
        raise ValueError("Input must contain either 'canonical_SMILES' or 'SMILES'.")
    print(f"Loaded {len(df)} candidates")
    print("Using SMILES column:", SMILES_COL)
    print("Columns:", df.columns.tolist())

    def mol_from_smiles(smi):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def has_match(mol, patt):
        if mol is None or patt is None:
            return False
        try:
            return bool(mol.HasSubstructMatch(patt))
        except Exception:
            return False

    def count_match(mol, patt):
        if mol is None or patt is None:
            return 0
        try:
            return len(mol.GetSubstructMatches(patt))
        except Exception:
            return 0

    def formal_charge(mol):
        if mol is None:
            return np.nan
        try:
            return Chem.GetFormalCharge(mol)
        except Exception:
            return np.nan

    def fragment_count(mol):
        if mol is None:
            return np.nan
        try:
            return len(Chem.GetMolFrags(mol))
        except Exception:
            return np.nan

    def positive_n_count(mol):
        if mol is None:
            return 0
        return sum(
            (
                1
                for a in mol.GetAtoms()
                if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0
            )
        )

    def aromatic_positive_n_count(mol):
        if mol is None:
            return 0
        return sum(
            (
                1
                for a in mol.GetAtoms()
                if a.GetAtomicNum() == 7
                and a.GetFormalCharge() > 0
                and a.GetIsAromatic()
            )
        )

    def aliphatic_positive_n_count(mol):
        if mol is None:
            return 0
        return sum(
            (
                1
                for a in mol.GetAtoms()
                if a.GetAtomicNum() == 7
                and a.GetFormalCharge() > 0
                and (not a.GetIsAromatic())
            )
        )

    def positive_p_count(mol):
        if mol is None:
            return 0
        return sum(
            (
                1
                for a in mol.GetAtoms()
                if a.GetAtomicNum() == 15 and a.GetFormalCharge() > 0
            )
        )

    def safe_get(row, key, default=np.nan):
        return row[key] if key in row.index else default

    def normalize_series(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        lo = s.quantile(0.05)
        hi = s.quantile(0.95)
        if hi == lo:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return ((s - lo) / (hi - lo)).clip(0, 1).fillna(0)

    SMARTS_PEROXIDE = Chem.MolFromSmarts("O-O")
    SMARTS_AZIDE = Chem.MolFromSmarts("[N-]=[N+]=N")
    SMARTS_NITROSO = Chem.MolFromSmarts("N=O")
    SMARTS_ACID_CHLORIDE = Chem.MolFromSmarts("C(=O)Cl")
    SMARTS_ISOCYANATE = Chem.MolFromSmarts("N=C=O")
    SMARTS_EPOXIDE = Chem.MolFromSmarts("C1OC1")
    SMARTS_BENZYL_QUAT = Chem.MolFromSmarts("c[CH2][N+]")
    SMARTS_ALKYL_QUAT = Chem.MolFromSmarts("[CX4][N+]")
    SMARTS_AROMATIC_N_CATION = Chem.MolFromSmarts("[n+]")
    SMARTS_PHOSPHONIUM = Chem.MolFromSmarts("[P+]")
    SMARTS_ARYL_ETHER = Chem.MolFromSmarts("a-O-a")
    SMARTS_ALKYL_ETHER = Chem.MolFromSmarts("COC")
    SMARTS_METHOXY_ARYL = Chem.MolFromSmarts("COa")
    SMARTS_TERTIARY_AMMONIUM = Chem.MolFromSmarts("[N+]([#6])([#6])([#6])[#6]")
    SMARTS_CYCLIC_AMMONIUM = Chem.MolFromSmarts("[N+]1")
    SMARTS_HALIDE_LIKE = Chem.MolFromSmarts("[Cl,Br,I]")
    SMARTS_BENZYL_QUAT_DISC = Chem.MolFromSmarts("c[CH2][N+]")
    SMARTS_ALKYL_QUAT_DISC = Chem.MolFromSmarts("[CX4][N+]")
    SMARTS_IMIDAZOLIUM_DISC = Chem.MolFromSmarts("c1[n+]([#6])cc[n,c]1")
    SMARTS_AROMATIC_N_DISC = Chem.MolFromSmarts("[n+]")
    SMARTS_ARYL_ETHER_DISC = Chem.MolFromSmarts("a-O-a")
    SMARTS_WILLIAMSON_ETHER_DISC = Chem.MolFromSmarts("[CX4]O[CX4,c]")

    def cation_type_from_mol(mol):
        if mol is None:
            return "invalid"
        n_arom = aromatic_positive_n_count(mol)
        n_alip = aliphatic_positive_n_count(mol)
        n_p = positive_p_count(mol)
        if n_p > 0:
            return "phosphonium"
        if n_arom > 0 and n_alip > 0:
            return "mixed_N_cation"
        if n_arom > 0:
            return "aromatic_N_cation"
        if n_alip > 0:
            return "quat_ammonium"
        if formal_charge(mol) > 0:
            return "other_cationic"
        return "no_cationic_head"

    def assign_disconnection_labels(mol):
        """
        Assign retrosynthesis-style disconnection labels.

        These are not true retrosynthetic routes.
        They are route-hypothesis labels useful for triage and reporting.
        """
        if mol is None:
            return ["invalid"]
        labels = []
        if has_match(mol, SMARTS_BENZYL_QUAT_DISC):
            labels.append("benzyl quaternization candidate")
        if has_match(mol, SMARTS_ALKYL_QUAT_DISC):
            labels.append("alkyl quaternization candidate")
        if has_match(mol, SMARTS_IMIDAZOLIUM_DISC):
            labels.append("imidazolium alkylation candidate")
        elif has_match(mol, SMARTS_AROMATIC_N_DISC):
            labels.append("aromatic N-alkylation candidate")
        if has_match(mol, SMARTS_ARYL_ETHER_DISC):
            labels.append("aryl ether coupling candidate")
        if has_match(mol, SMARTS_WILLIAMSON_ETHER_DISC):
            labels.append("Williamson ether candidate")
        if len(labels) == 0:
            labels.append("no obvious disconnection")
        return labels

    def primary_disconnection_label(labels):
        """
        Choose one primary disconnection label for sorting/reporting.

        Priority favors AEM cation-installation logic first.
        """
        priority = [
            "benzyl quaternization candidate",
            "alkyl quaternization candidate",
            "imidazolium alkylation candidate",
            "aromatic N-alkylation candidate",
            "aryl ether coupling candidate",
            "Williamson ether candidate",
            "no obvious disconnection",
            "invalid",
        ]
        for p in priority:
            if p in labels:
                return p
        return labels[0] if labels else "no obvious disconnection"

    def get_template_flags(mol):
        flags = {}
        flags["valid_rdkit"] = mol is not None
        flags["formal_charge"] = formal_charge(mol) if mol is not None else np.nan
        flags["fragment_count"] = fragment_count(mol) if mol is not None else np.nan
        flags["has_peroxide"] = has_match(mol, SMARTS_PEROXIDE)
        flags["has_azide"] = has_match(mol, SMARTS_AZIDE)
        flags["has_nitroso"] = has_match(mol, SMARTS_NITROSO)
        flags["has_acid_chloride"] = has_match(mol, SMARTS_ACID_CHLORIDE)
        flags["has_isocyanate"] = has_match(mol, SMARTS_ISOCYANATE)
        flags["has_epoxide"] = has_match(mol, SMARTS_EPOXIDE)
        flags["cation_type_template"] = cation_type_from_mol(mol)
        flags["positive_N_count"] = positive_n_count(mol)
        flags["aromatic_positive_N_count"] = aromatic_positive_n_count(mol)
        flags["aliphatic_positive_N_count"] = aliphatic_positive_n_count(mol)
        flags["positive_P_count"] = positive_p_count(mol)
        flags["has_benzyl_quat_template"] = has_match(mol, SMARTS_BENZYL_QUAT)
        flags["has_alkyl_quat_template"] = has_match(mol, SMARTS_ALKYL_QUAT)
        flags["has_aromatic_N_cation_template"] = has_match(
            mol, SMARTS_AROMATIC_N_CATION
        )
        flags["has_phosphonium_template"] = has_match(mol, SMARTS_PHOSPHONIUM)
        flags["has_cyclic_ammonium_template"] = has_match(mol, SMARTS_CYCLIC_AMMONIUM)
        flags["aryl_ring_count_proxy"] = count_match(mol, Chem.MolFromSmarts("a"))
        flags["ether_count_template"] = count_match(mol, SMARTS_ALKYL_ETHER)
        flags["has_aryl_ether_template"] = has_match(mol, SMARTS_ARYL_ETHER)
        flags["has_alkyl_ether_template"] = has_match(mol, SMARTS_ALKYL_ETHER)
        flags["has_methoxy_aryl_template"] = has_match(mol, SMARTS_METHOXY_ARYL)
        flags["has_halogen"] = has_match(mol, SMARTS_HALIDE_LIKE)
        return flags

    def route_hint_from_flags_and_disconnections(flags, disconnection_labels):
        hints = []
        ctype = flags.get("cation_type_template", "unknown")
        if "benzyl quaternization candidate" in disconnection_labels:
            hints.append(
                "Benzyl quaternization route plausible: benzyl-halide precursor plus trimethylamine/tertiary amine."
            )
        if "alkyl quaternization candidate" in disconnection_labels:
            hints.append(
                "Alkyl quaternization route plausible: alkyl-halide or sulfonate ester precursor plus tertiary amine."
            )
        if "imidazolium alkylation candidate" in disconnection_labels:
            hints.append(
                "Imidazolium route plausible: imidazole precursor N-alkylation/quaternization."
            )
        if "aromatic N-alkylation candidate" in disconnection_labels:
            hints.append(
                "Aromatic N-cation route plausible by N-alkylation of pyridine/imidazole/triazole-like precursor."
            )
        if "aryl ether coupling candidate" in disconnection_labels:
            hints.append(
                "Aryl ether disconnection plausible: phenol plus aryl halide/coupling partner."
            )
        if "Williamson ether candidate" in disconnection_labels:
            hints.append(
                "Williamson ether disconnection plausible: alcohol/phenoxide plus alkyl electrophile."
            )
        if "no obvious disconnection" in disconnection_labels:
            hints.append(
                "No obvious simple disconnection identified; manual retrosynthetic review required."
            )
        if ctype == "quat_ammonium":
            hints.append(
                "Quaternary ammonium motif is AEM-relevant and often accessible by amine quaternization."
            )
        elif ctype == "aromatic_N_cation":
            hints.append(
                "Aromatic N-cation is AEM-relevant but should be flagged for alkaline stability risk."
            )
        elif ctype == "mixed_N_cation":
            hints.append(
                "Mixed N-cation motif may require orthogonal alkylation or multi-step route."
            )
        elif ctype == "phosphonium":
            hints.append(
                "Phosphonium motif may be accessible by phosphine alkylation, but precursor handling should be checked."
            )
        elif ctype == "other_cationic":
            hints.append("Unusual cationic motif; manual route review recommended.")
        elif ctype == "no_cationic_head":
            hints.append("No AEM-relevant cationic headgroup detected.")
        if flags.get("has_halogen", False):
            hints.append(
                "Halogen present; could be a precursor handle, but final structure should be checked carefully."
            )
        if flags.get("fragment_count", 1) > 1:
            hints.append(
                "Disconnected fragments detected; unsuitable as a single generated molecule without salt/fragment handling."
            )
        return " ".join(hints)

    def compute_base_synthesizability_score(row, flags):
        """
        Rule-based synthesizability score in [0, 1].

        This is not a true retrosynthesis score.
        It is a practical pre-MD triage score.
        """
        score = 0.5
        if not flags.get("valid_rdkit", False):
            return 0.0
        reject_flags = [
            "has_peroxide",
            "has_azide",
            "has_nitroso",
            "has_acid_chloride",
            "has_isocyanate",
        ]
        if any((flags.get(f, False) for f in reject_flags)):
            return 0.0
        if flags.get("has_epoxide", False):
            score -= 0.25
        ctype = flags.get("cation_type_template", "")
        if ctype == "quat_ammonium":
            score += 0.18
        elif ctype == "aromatic_N_cation":
            score += 0.1
            score -= 0.08
        elif ctype == "mixed_N_cation":
            score += 0.08
            score -= 0.1
        elif ctype == "phosphonium":
            score += 0.08
            score -= 0.05
        elif ctype == "other_cationic":
            score += 0.02
            score -= 0.1
        else:
            score -= 0.4
        if flags.get("has_benzyl_quat_template", False):
            score += 0.12
        if flags.get("has_alkyl_quat_template", False):
            score += 0.08
        if flags.get("has_aromatic_N_cation_template", False):
            score += 0.04
        if flags.get("has_aryl_ether_template", False):
            score += 0.05
        if flags.get("has_alkyl_ether_template", False):
            score += 0.04
        if flags.get("has_methoxy_aryl_template", False):
            score += 0.03
        frag = flags.get("fragment_count", 1)
        if pd.notna(frag) and frag > 1:
            score -= 0.35
        ncat = safe_get(row, "cationic_site_count", flags.get("positive_N_count", 0))
        if pd.notna(ncat):
            if ncat == 1:
                score += 0.08
            elif ncat == 2:
                score += 0.04
            elif ncat > 2:
                score -= 0.1
            elif ncat < 1:
                score -= 0.4
        sa = safe_get(row, "SA_score", np.nan)
        if pd.notna(sa):
            if sa <= 3.5:
                score += 0.12
            elif sa <= 5.0:
                score += 0.07
            elif sa <= 7.0:
                score += 0.02
            else:
                score -= 0.12
        mw = safe_get(row, "MW", np.nan)
        if pd.notna(mw):
            if 250 <= mw <= 600:
                score += 0.08
            elif 600 < mw <= 800:
                score -= 0.05
            elif mw > 800:
                score -= 0.2
            elif mw < 200:
                score -= 0.1
        rot = safe_get(row, "RotBonds", np.nan)
        if pd.notna(rot):
            if rot <= 12:
                score += 0.04
            elif 12 < rot <= 20:
                score -= 0.02
            elif rot > 20:
                score -= 0.12
        logp = safe_get(row, "LogP", np.nan)
        if pd.notna(logp):
            if logp <= 6:
                score += 0.05
            elif 6 < logp <= 9:
                score -= 0.03
            elif logp > 9:
                score -= 0.12
        iec = safe_get(row, "IEC_proxy_mmol_g", np.nan)
        if pd.notna(iec):
            if 1.0 <= iec <= 4.0:
                score += 0.06
            elif 0.5 <= iec < 1.0 or 4.0 < iec <= 6.0:
                score += 0.02
            else:
                score -= 0.06
        return float(np.clip(score, 0.0, 1.0))

    def apply_disconnection_score_adjustment(score, disconnection_labels):
        """
        Adjust score based on clarity of retrosynthesis-style disconnections.
        """
        if "benzyl quaternization candidate" in disconnection_labels:
            score += 0.08
        if "alkyl quaternization candidate" in disconnection_labels:
            score += 0.05
        if "imidazolium alkylation candidate" in disconnection_labels:
            score += 0.04
        if "aromatic N-alkylation candidate" in disconnection_labels:
            score += 0.03
        if "aryl ether coupling candidate" in disconnection_labels:
            score += 0.04
        if "Williamson ether candidate" in disconnection_labels:
            score += 0.04
        if "no obvious disconnection" in disconnection_labels:
            score -= 0.12
        return float(np.clip(score, 0.0, 1.0))

    def classify_synthesizability(score, flags):
        if score <= 0.05:
            return "reject"
        if not flags.get("valid_rdkit", False):
            return "reject"
        if flags.get("fragment_count", 1) > 1:
            return "reject"
        if flags.get("cation_type_template", "") in ["no_cationic_head", "invalid"]:
            return "reject"
        reject_flags = [
            "has_peroxide",
            "has_azide",
            "has_nitroso",
            "has_acid_chloride",
            "has_isocyanate",
        ]
        if any((flags.get(f, False) for f in reject_flags)):
            return "reject"
        if score >= 0.78:
            return "easy"
        elif score >= 0.55:
            return "moderate"
        elif score >= 0.3:
            return "difficult"
        else:
            return "reject"

    def synth_notes_from_class(score, synth_class, flags, row, disconnection_labels):
        notes = []
        if synth_class == "easy":
            notes.append(
                "Recognizable AEM route motifs and manageable structural complexity."
            )
        elif synth_class == "moderate":
            notes.append(
                "Plausible synthesis, but may require multi-step route or careful precursor choice."
            )
        elif synth_class == "difficult":
            notes.append(
                "Synthetically possible but likely challenging or structurally complex."
            )
        else:
            notes.append(
                "Rejected or low-priority due to instability, missing AEM motif, disconnected fragments, or high complexity."
            )
        if "benzyl quaternization candidate" in disconnection_labels:
            notes.append("Benzyl quaternization is a strong route hypothesis.")
        if "alkyl quaternization candidate" in disconnection_labels:
            notes.append(
                "Alkyl quaternization is a plausible cation-installation route."
            )
        if "imidazolium alkylation candidate" in disconnection_labels:
            notes.append(
                "Imidazolium N-alkylation route is plausible but alkaline stability should be checked."
            )
        if "aryl ether coupling candidate" in disconnection_labels:
            notes.append(
                "Aryl ether linkage suggests phenol/aryl-halide coupling or related ether-forming chemistry."
            )
        if "Williamson ether candidate" in disconnection_labels:
            notes.append(
                "Williamson ether disconnection suggests alcohol/phenoxide plus alkyl electrophile route."
            )
        if "no obvious disconnection" in disconnection_labels:
            notes.append(
                "No simple route template detected; manual retrosynthesis needed."
            )
        if flags.get("cation_type_template") == "quat_ammonium":
            notes.append(
                "Quaternary ammonium is AEM-relevant and commonly accessible by quaternization."
            )
        elif flags.get("cation_type_template") == "aromatic_N_cation":
            notes.append(
                "Aromatic N-cation is AEM-relevant but has possible alkaline stability risk."
            )
        if safe_get(row, "LogP", np.nan) > 8:
            notes.append(
                "High LogP may reduce hydration and anion mobility; validate by hydrated MD."
            )
        if safe_get(row, "RotBonds", np.nan) > 20:
            notes.append(
                "High flexibility may complicate conformational stability and synthesis."
            )
        if safe_get(row, "MW", np.nan) > 700:
            notes.append(
                "High molecular weight may indicate a more difficult monomer/repeat-unit candidate."
            )
        return " ".join(notes)

    records = []
    for idx, row in df.iterrows():
        smi = row[SMILES_COL]
        mol = mol_from_smiles(smi)
        flags = get_template_flags(mol)
        disconnection_labels = assign_disconnection_labels(mol)
        primary_disconnection = primary_disconnection_label(disconnection_labels)
        route_hint = route_hint_from_flags_and_disconnections(
            flags, disconnection_labels
        )
        base_synth_score = compute_base_synthesizability_score(row, flags)
        synth_score = apply_disconnection_score_adjustment(
            base_synth_score, disconnection_labels
        )
        synth_class = classify_synthesizability(synth_score, flags)
        synth_notes = synth_notes_from_class(
            synth_score, synth_class, flags, row, disconnection_labels
        )
        rec = {}
        rec["synthesis_input_index"] = idx
        rec["template_flags"] = "; ".join(
            [k for k, v in flags.items() if isinstance(v, bool) and v is True]
        )
        rec["cation_type_template"] = flags.get("cation_type_template")
        rec["disconnection_labels"] = "; ".join(disconnection_labels)
        rec["primary_disconnection"] = primary_disconnection
        rec["synthesis_route_hint"] = route_hint
        rec["base_synthesizability_score"] = base_synth_score
        rec["synthesizability_score"] = synth_score
        rec["synthesizability_class"] = synth_class
        rec["synthesizability_notes"] = synth_notes
        for key, val in flags.items():
            if key not in ["formal_charge", "fragment_count"]:
                rec[f"flag_{key}"] = val
        records.append(rec)
    screen_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    if "aem_composite_score" in screen_df.columns:
        screen_df["norm_aem_structural_score"] = normalize_series(
            screen_df["aem_composite_score"]
        )
    else:
        screen_df["norm_aem_structural_score"] = 0.0
    possible_cond_cols = [
        "predicted_conductivity_oldGNN",
        "predicted_conductivity",
        "Conductivity",
        "Anion_Conductivity_Scm",
        "Anion_Conductivity_pred_Scm",
    ]
    cond_col = None
    for c in possible_cond_cols:
        if c in screen_df.columns:
            cond_col = c
            break
    if cond_col is not None:
        print("Conductivity-like column found:", cond_col)
        screen_df["norm_predicted_conductivity"] = normalize_series(screen_df[cond_col])
        screen_df["final_preMD_score"] = (
            0.35 * screen_df["norm_aem_structural_score"]
            + 0.3 * screen_df["norm_predicted_conductivity"]
            + 0.25 * screen_df["synthesizability_score"]
            + 0.1
            * normalize_series(
                screen_df.get("IEC_proxy_mmol_g", pd.Series(np.zeros(len(screen_df))))
            )
        )
    else:
        print(
            "No conductivity prediction column found. Using structural + synthesizability final score."
        )
        screen_df["norm_predicted_conductivity"] = 0.0
        screen_df["final_preMD_score"] = (
            0.55 * screen_df["norm_aem_structural_score"]
            + 0.35 * screen_df["synthesizability_score"]
            + 0.1
            * normalize_series(
                screen_df.get("IEC_proxy_mmol_g", pd.Series(np.zeros(len(screen_df))))
            )
        )
    screen_df.loc[
        screen_df["synthesizability_class"] == "reject", "final_preMD_score"
    ] *= 0.1
    class_priority = {"easy": 3, "moderate": 2, "difficult": 1, "reject": 0}
    disc_priority = {
        "benzyl quaternization candidate": 6,
        "alkyl quaternization candidate": 5,
        "imidazolium alkylation candidate": 4,
        "aromatic N-alkylation candidate": 3,
        "aryl ether coupling candidate": 2,
        "Williamson ether candidate": 1,
        "no obvious disconnection": 0,
        "invalid": -1,
    }
    screen_df["synthesizability_class_priority"] = (
        screen_df["synthesizability_class"].map(class_priority).fillna(0)
    )
    screen_df["disconnection_priority"] = (
        screen_df["primary_disconnection"].map(disc_priority).fillna(0)
    )
    screen_df = screen_df.sort_values(
        [
            "synthesizability_class_priority",
            "final_preMD_score",
            "disconnection_priority",
            "synthesizability_score",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    screened_path = f"{OUT_DIR}/AEM_TOP_100_synthesizability_screened_v2.csv"
    top20_path = f"{OUT_DIR}/AEM_TOP_20_synthesizable_candidates_v2.csv"
    top10_path = f"{OUT_DIR}/AEM_TOP_10_synthesizable_for_MD_v2.csv"
    screen_df.to_csv(screened_path, index=False)
    usable = screen_df[
        screen_df["synthesizability_class"].isin(["easy", "moderate", "difficult"])
    ].copy()
    top20 = usable.head(20).copy()
    top10 = usable.head(10).copy()
    top20.to_csv(top20_path, index=False)
    top10.to_csv(top10_path, index=False)
    print("\nSaved:")
    print("  Full screened:", screened_path)
    print("  Top 20:", top20_path)
    print("  Top 10 for MD:", top10_path)
    plt.figure(figsize=(7, 5))
    class_order = ["easy", "moderate", "difficult", "reject"]
    class_counts = (
        screen_df["synthesizability_class"]
        .value_counts()
        .reindex(class_order)
        .fillna(0)
    )
    plt.bar(class_counts.index, class_counts.values, edgecolor="black", alpha=0.8)
    plt.xlabel("Synthesizability class")
    plt.ylabel("Count")
    plt.title("AEM Synthesizability Class Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    class_plot_path = f"{OUT_DIR}/AEM_synthesizability_class_distribution_v2.png"
    plt.savefig(class_plot_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.figure(figsize=(11, 5))
    disc_counts = screen_df["primary_disconnection"].value_counts()
    plt.bar(range(len(disc_counts)), disc_counts.values, edgecolor="black", alpha=0.8)
    plt.xticks(range(len(disc_counts)), disc_counts.index, rotation=45, ha="right")
    plt.xlabel("Primary retrosynthesis-style disconnection")
    plt.ylabel("Count")
    plt.title("AEM Primary Disconnection Label Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    disc_plot_path = f"{OUT_DIR}/AEM_primary_disconnection_distribution_v2.png"
    plt.savefig(disc_plot_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.figure(figsize=(7, 5))
    plt.hist(screen_df["final_preMD_score"], bins=25, edgecolor="black", alpha=0.8)
    plt.xlabel("Final pre-MD score")
    plt.ylabel("Count")
    plt.title("AEM Final Pre-MD Score Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    score_plot_path = f"{OUT_DIR}/AEM_final_preMD_score_distribution_v2.png"
    plt.savefig(score_plot_path, dpi=200, bbox_inches="tight")
    plt.show()
    if "aem_composite_score" in screen_df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=screen_df,
            x="aem_composite_score",
            y="synthesizability_score",
            hue="synthesizability_class",
            hue_order=class_order,
            s=70,
        )
        plt.xlabel("AEM structural composite score")
        plt.ylabel("Synthesizability score")
        plt.title("Synthesizability vs AEM Structural Score")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        scatter_path = f"{OUT_DIR}/AEM_synthesizability_vs_AEM_score_v2.png"
        plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.show()
    else:
        scatter_path = None
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=screen_df, x="primary_disconnection", y="final_preMD_score")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Primary disconnection")
    plt.ylabel("Final pre-MD score")
    plt.title("Final Pre-MD Score by Disconnection Label")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    disc_score_path = f"{OUT_DIR}/AEM_final_score_by_disconnection_v2.png"
    plt.savefig(disc_score_path, dpi=200, bbox_inches="tight")
    plt.show()
    top_mols = []
    legends = []
    for i, row in top20.iterrows():
        smi = row[SMILES_COL]
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        top_mols.append(mol)
        legend = f"#{len(top_mols)}\n"
        legend += f"{row['synthesizability_class']} | Synth {row['synthesizability_score']:.2f}\n"
        legend += f"PreMD {row['final_preMD_score']:.2f}\n"
        legend += f"{row['primary_disconnection'][:28]}"
        legends.append(legend)
    if len(top_mols) > 0:
        img = Draw.MolsToGridImage(
            top_mols[:20],
            molsPerRow=4,
            subImgSize=(430, 430),
            legends=legends[:20],
            returnPNG=False,
        )
        grid_path = f"{OUT_DIR}/AEM_TOP_20_synthesizable_molecules_v2.png"
        img.save(grid_path)
        display(img)
        print("Saved molecule grid:", grid_path)
    else:
        grid_path = None
        print("No valid molecules for grid image.")
    summary = {
        "input_file": INPUT_CSV,
        "total_candidates_screened": int(len(screen_df)),
        "conductivity_column_used": cond_col,
        "class_counts": screen_df["synthesizability_class"].value_counts().to_dict(),
        "primary_disconnection_counts": screen_df["primary_disconnection"]
        .value_counts()
        .to_dict(),
        "mean_synthesizability_score": float(
            screen_df["synthesizability_score"].mean()
        ),
        "mean_final_preMD_score": float(screen_df["final_preMD_score"].mean()),
        "outputs": {
            "screened_csv": screened_path,
            "top20_csv": top20_path,
            "top10_csv": top10_path,
            "class_distribution_plot": class_plot_path,
            "primary_disconnection_plot": disc_plot_path,
            "final_score_plot": score_plot_path,
            "synth_vs_aem_score_plot": scatter_path,
            "final_score_by_disconnection_plot": disc_score_path,
            "top20_molecule_grid": grid_path,
        },
    }
    summary_path = f"{OUT_DIR}/AEM_synthesizability_summary_v2.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 100)
    print(" AEM TEMPLATE-BASED SYNTHESIZABILITY SCREENING V2 SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"\nInput: {INPUT_CSV}")
    print(f"Candidates screened: {len(screen_df)}")
    print(f"Conductivity column used: {cond_col}")
    print("\n--- Synthesizability Class Counts ---")
    for cls in class_order:
        print(f"{cls:10s}: {int(class_counts.get(cls, 0))}")
    print("\n--- Primary Disconnection Counts ---")
    for disc, count in screen_df["primary_disconnection"].value_counts().items():
        print(f"{disc:40s}: {count}")
    print("\n--- Top 10 Synthesizable Candidates for MD ---")
    for i, row in top10.iterrows():
        print(f"\n#{i + 1}")
        print(f"  SMILES: {row[SMILES_COL]}")
        print(f"  Class: {row['synthesizability_class']}")
        print(f"  Synth score: {row['synthesizability_score']:.3f}")
        print(f"  Final pre-MD score: {row['final_preMD_score']:.3f}")
        print(f"  Primary disconnection: {row['primary_disconnection']}")
        print(f"  All disconnections: {row['disconnection_labels']}")
        if "aem_head_type" in row.index:
            print(f"  AEM head type: {row['aem_head_type']}")
        if "MW" in row.index:
            print(f"  MW: {row['MW']:.1f}")
        if "LogP" in row.index:
            print(f"  LogP: {row['LogP']:.2f}")
        if "IEC_proxy_mmol_g" in row.index:
            print(f"  IEC proxy: {row['IEC_proxy_mmol_g']:.3f}")
        print(f"  Route hint: {row['synthesis_route_hint']}")
        print(f"  Notes: {row['synthesizability_notes']}")
    print("\n--- Saved Files ---")
    print(f"1. Full screened CSV:       {screened_path}")
    print(f"2. Top 20 CSV:              {top20_path}")
    print(f"3. Top 10 for MD CSV:       {top10_path}")
    print(f"4. Summary JSON:            {summary_path}")
    print(f"5. Plots / molecule grid:   {OUT_DIR}")
    print("\n" + "=" * 100)
    print(" SYNTHESIZABILITY SCREENING V2 COMPLETE ".center(100, "="))
    print("=" * 100)
    return root
