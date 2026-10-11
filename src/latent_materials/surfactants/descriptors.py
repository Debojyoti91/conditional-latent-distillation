"""Final surfactant comparison helpers for the reported surfactant comparison.
Historical descriptor definitions are preserved; regex counts depend on SMILES spelling.
Heuristic accessibility is the in-house complexity score, not Ertl SA.
"""

import re
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

FP_BITS = 2048
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def canon_smiles(smi):

    mol = Chem.MolFromSmiles(str(smi))

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
    )


def longest_chain(smiles):

    runs = re.findall(
        r"C+",
        str(smiles),
    )

    return max(
        (len(run) for run in runs),
        default=0,
    )


def count_EO(smiles):

    return len(
        re.findall(
            r"CCO",
            str(smiles),
        )
    )


def count_PO(smiles):

    return len(
        re.findall(
            r"CC\(C\)O",
            str(smiles),
        )
    )


def ring_count(mol):

    try:

        return mol.GetRingInfo().NumRings()

    except Exception:

        return np.nan


def branching_index(mol):

    try:

        carbons = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]

        if not carbons:

            return 0.0

        return float(np.mean([atom.GetTotalDegree() for atom in carbons]))

    except Exception:

        return np.nan


def aliphatic_fraction(mol):

    try:

        carbons = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]

        if not carbons:

            return 0.0

        n_aliphatic = sum(1 for atom in carbons if not atom.GetIsAromatic())

        return n_aliphatic / len(carbons)

    except Exception:

        return np.nan


def detect_head(smiles):

    s = str(smiles).lower()

    if "s(=o)(=o)o" in s or "so3" in s:

        return "sulfate"

    if "s(=o)(=o)" in s and "o" in s:

        return "sulfonate"

    if "coo" in s:

        return "carboxylate"

    if "n+" in s or "[n+" in s:

        return "quat_ammonium"

    if re.search(
        r"O(?:CCO|CC\(C\)O)+S\(",
        s,
    ):

        return "ethoxylated"

    return "other"


def heuristic_accessibility_score(mol):

    rings = mol.GetRingInfo().NumRings()

    chiral = len(
        Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=True,
        )
    )

    mw = Descriptors.MolWt(mol)

    rot_bonds = Descriptors.NumRotatableBonds(mol)

    score = rings * 0.5 + chiral * 1.0 + (mw / 100.0) * 0.3 + (rot_bonds / 10.0) * 0.2

    return float(score)


def composite_score(row):

    score = 0.0

    chain_score = 1.0 - abs(row["longest_chain"] - 13) / 10.0

    score += (
        max(
            0.0,
            chain_score,
        )
        * 0.30
    )

    eo_score = 1.0 - abs(row["EO_count"] - 6) / 5.0

    score += (
        max(
            0.0,
            eo_score,
        )
        * 0.20
    )

    accessibility_term = 1.0 - row["SA_score"] / 10.0

    score += (
        max(
            0.0,
            accessibility_term,
        )
        * 0.20
    )

    logp_score = 1.0 - abs(row["LogP"] - 3.5) / 3.0

    score += (
        max(
            0.0,
            logp_score,
        )
        * 0.15
    )

    mw_score = 1.0 - abs(row["MW"] - 450) / 200.0

    score += (
        max(
            0.0,
            mw_score,
        )
        * 0.15
    )

    return float(score)


def smiles_to_fp(smi):

    mol = Chem.MolFromSmiles(str(smi))

    if mol is None:
        return None

    return morgan_generator.GetFingerprint(mol)


def fp_to_numpy(fp):

    arr = np.zeros(
        FP_BITS,
        dtype=np.int8,
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr,
    )

    return arr


def fps_for_smiles(smiles_list):

    arrays = []

    for smi in smiles_list:

        fp = smiles_to_fp(smi)

        if fp is not None:

            arrays.append(fp_to_numpy(fp))

    if len(arrays) == 0:

        return np.empty(
            (0, FP_BITS),
            dtype=np.int8,
        )

    return np.asarray(arrays)


def max_tanimoto_to_reference(
    query_smiles,
    reference_fps,
):

    output = []

    for smi in query_smiles:

        fp = smiles_to_fp(smi)

        if fp is None or len(reference_fps) == 0:

            output.append(np.nan)

            continue

        similarities = DataStructs.BulkTanimotoSimilarity(
            fp,
            reference_fps,
        )

        output.append(float(max(similarities)) if similarities else np.nan)

    return np.asarray(
        output,
        dtype=float,
    )


def novelty_category(similarity):

    if pd.isna(similarity):

        return "Invalid"

    if similarity >= 0.85:

        return "Very similar"

    if similarity >= 0.70:

        return "Moderately similar"

    if similarity >= 0.50:

        return "Moderately novel"

    return "Highly novel"


def diversity_filter(
    df,
    threshold=0.80,
    max_keep=500,
):

    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie in (0, 1]")
    if (
        isinstance(max_keep, bool)
        or not isinstance(max_keep, (int, np.integer))
        or max_keep < 0
    ):
        raise ValueError("max_keep must be a nonnegative integer")
    if max_keep == 0:
        return df.iloc[:0].copy()

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

    if max_keep == 1:
        return df.iloc[keep_indices].copy().reset_index(drop=True)

    for i in range(
        1,
        len(fps),
    ):

        max_similarity = max(
            DataStructs.TanimotoSimilarity(
                fps[i],
                selected_fp,
            )
            for selected_fp in keep_fps
        )

        if max_similarity < threshold:

            keep_indices.append(valid_indices[i])

            keep_fps.append(fps[i])

            if len(keep_indices) >= max_keep:

                break

    return df.iloc[keep_indices].copy().reset_index(drop=True)
