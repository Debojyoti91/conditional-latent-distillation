"""AEM production descriptors and candidate-ranking formulas.
The production ranking recorded in the notebook had no conductivity term.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

OO_smarts = Chem.MolFromSmarts("O-O")


def _bool01(value):
    """Interpret CSV flags consistently with the recorded sensitivity analysis."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def valid_mol(smiles):
    """Return sanitized RDKit mol or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonical_smiles(smiles):
    mol = valid_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def formal_charge(mol):
    try:
        return Chem.GetFormalCharge(mol)
    except Exception:
        return np.nan


def fragment_count(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return len(Chem.GetMolFrags(mol))
    except Exception:
        return np.nan


def has_peroxy(mol):
    try:
        return int(mol.HasSubstructMatch(OO_smarts))
    except Exception:
        return 0


def detect_aem_headgroup(mol, smiles):
    """
    Corrected AEM headgroup classifier.

    Separates:
      - quat_ammonium: aliphatic positively charged N
      - aromatic_N_cation: aromatic positively charged N, e.g. imidazolium/pyridinium/triazolium-like
      - phosphonium: positively charged P
      - mixed_N_cation: both aromatic and aliphatic N+ sites
      - other_cationic
      - no_cationic_head
    """
    if mol is None:
        return "invalid"

    pos_N_atoms = [
        a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0
    ]

    pos_P_atoms = [
        a for a in mol.GetAtoms() if a.GetAtomicNum() == 15 and a.GetFormalCharge() > 0
    ]

    if len(pos_P_atoms) > 0:
        return "phosphonium"

    if len(pos_N_atoms) > 0:
        aromatic_pos_N = [a for a in pos_N_atoms if a.GetIsAromatic()]
        aliphatic_pos_N = [a for a in pos_N_atoms if not a.GetIsAromatic()]

        if len(aromatic_pos_N) > 0 and len(aliphatic_pos_N) > 0:
            return "mixed_N_cation"

        if len(aromatic_pos_N) > 0:
            return "aromatic_N_cation"

        if len(aliphatic_pos_N) > 0:
            return "quat_ammonium"

    if Chem.GetFormalCharge(mol) > 0:
        return "other_cationic"

    return "no_cationic_head"


def longest_carbon_chain_graph(mol):
    """
    Approximate longest C-C path using graph distance over carbon atoms.
    More robust than regex C+ for branched/aromatic molecules.
    """
    try:
        carbon_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        if not carbon_idxs:
            return 0

        adj = {i: [] for i in carbon_idxs}
        carbon_set = set(carbon_idxs)

        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in carbon_set and j in carbon_set:
                adj[i].append(j)
                adj[j].append(i)

        def bfs(start):
            visited = {start}
            queue = [(start, 0)]
            farthest, max_dist = start, 0

            while queue:
                node, dist = queue.pop(0)
                if dist > max_dist:
                    farthest, max_dist = node, dist

                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, dist + 1))

            return farthest, max_dist

        best = 0
        for c in carbon_idxs:
            u, _ = bfs(c)
            _, d = bfs(u)
            best = max(best, d + 1)

        return best

    except Exception:
        return np.nan


def aromatic_fraction(mol):
    try:
        atoms = mol.GetAtoms()
        if len(atoms) == 0:
            return 0.0
        aromatic = sum(1 for a in atoms if a.GetIsAromatic())
        return aromatic / len(atoms)
    except Exception:
        return np.nan


def aliphatic_carbon_fraction(mol):
    try:
        carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        if not carbons:
            return 0.0
        alip = sum(1 for a in carbons if not a.GetIsAromatic())
        return alip / len(carbons)
    except Exception:
        return np.nan


def heteroatom_count(mol):
    try:
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6])
    except Exception:
        return np.nan


def cationic_site_count(mol):
    """
    Approximate number of cationic sites from atom formal charges.
    """
    try:
        return sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0)
    except Exception:
        return np.nan


def n_cationic_nitrogens(mol):
    try:
        return sum(
            1
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0
        )
    except Exception:
        return np.nan


def n_cationic_phosphorus(mol):
    try:
        return sum(
            1
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 15 and a.GetFormalCharge() > 0
        )
    except Exception:
        return np.nan


def ion_exchange_capacity_proxy(mol):
    """
    Rough IEC proxy:
        IEC_proxy = 1000 * cationic_sites / MW

    Interpretable approximately as mmol/g for monomer/repeat-unit level candidates
    if each fixed cationic site contributes one exchange site.
    """
    try:
        mw = Descriptors.MolWt(mol)
        n_cat = cationic_site_count(mol)
        if mw <= 0:
            return np.nan
        return 1000.0 * n_cat / mw
    except Exception:
        return np.nan


def branching_index(mol):
    try:
        carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        if not carbons:
            return 0.0
        return np.mean([a.GetTotalDegree() for a in carbons])
    except Exception:
        return np.nan


def sa_score_simple(mol):
    """
    Simple synthetic accessibility heuristic.
    Lower = easier.

    This is not the official Ertl SA score.
    """
    try:
        rings = mol.GetRingInfo().NumRings()
        chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        mw = Descriptors.MolWt(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        hetero = heteroatom_count(mol)
        formal = abs(Chem.GetFormalCharge(mol))

        score = (
            rings * 0.5
            + chiral * 1.0
            + (mw / 100.0) * 0.3
            + (rot_bonds / 10.0) * 0.2
            + hetero * 0.05
            + formal * 0.5
        )
        return score
    except Exception:
        return np.nan


def normalize_clip(x, lo, hi):
    if pd.isna(x):
        return 0.0
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def aem_composite_score(
    row, conductivity_column=None, conductivity_low=0.0, conductivity_high=1.0
):
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

    # Conductivity, if available
    if conductivity_column is not None and conductivity_column in row:
        cond_val = row[conductivity_column]
        cond_score = normalize_clip(cond_val, conductivity_low, conductivity_high)
        score += 0.35 * cond_score

    # Cationic site count
    ncat = row.get("cationic_site_count", 0)

    if ncat >= 1:
        score += 0.15

    if ncat == 1:
        score += 0.05
    elif ncat == 2:
        score += 0.04

    # IEC proxy
    iec = row.get("IEC_proxy_mmol_g", np.nan)
    if not pd.isna(iec):
        if 1.0 <= iec <= 4.0:
            score += 0.20
        elif 0.5 <= iec < 1.0 or 4.0 < iec <= 6.0:
            score += 0.10

    # Synthetic accessibility
    sa = row.get("SA_score", np.nan)
    if not pd.isna(sa):
        sa_quality = 1.0 - min(sa / 10.0, 1.0)
        score += 0.15 * sa_quality

    # Molecular weight balance
    mw = row.get("MW", np.nan)
    if not pd.isna(mw):
        mw_quality = 1.0 - abs(mw - 450.0) / 400.0
        score += 0.10 * max(0.0, mw_quality)

    # LogP balance
    logp = row.get("LogP", np.nan)
    if not pd.isna(logp):
        logp_quality = 1.0 - abs(logp - 2.5) / 4.0
        score += 0.05 * max(0.0, logp_quality)

    # Penalties
    if _bool01(row.get("has_peroxy", 0)):
        score -= 0.5

    if not _bool01(row.get("has_cationic_head", False)):
        score -= 0.5

    if row.get("fragment_count", 1) > 1:
        score -= 0.1

    return score
