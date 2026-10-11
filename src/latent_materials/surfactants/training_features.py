"""Recorded surrogate-training descriptors.

EO substring counts here differ from the comparison descriptors; retain the
original feature contract when fitting or loading the surrogate.
"""

import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def longest_chain(smiles):
    try:
        return max((len(c) for c in re.findall("C+", smiles)))
    except:
        return np.nan


def count_EO(smiles):
    return smiles.count("OCC") + smiles.count("OCO") + smiles.count("COC")


def aliph_frac(mol):
    try:
        carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        alip = sum((1 for a in carbons if not a.GetIsAromatic()))
        return alip / max(len(carbons), 1)
    except:
        return np.nan


def detect_head(smiles):
    s = smiles.lower()
    if "so3" in s or "s(=o)(=o)o" in s:
        return "sulfate"
    elif "coo" in s:
        return "carboxylate"
    elif "n+" in s or "[n+" in s:
        return "quat_ammonium"
    elif "oc" in s and "o(" in s:
        return "ethoxylated"
    else:
        return "other"


def enrich(df):
    feats = []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append([np.nan] * 6)
            continue
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        lc = longest_chain(smi)
        eo = count_EO(smi)
        af = aliph_frac(mol)
        head = detect_head(smi)
        feats.append([logp, tpsa, lc, af, eo, head])
    feat_df = pd.DataFrame(
        feats,
        columns=[
            "logP",
            "TPSA",
            "longest_chain",
            "aliphatic_fraction",
            "EO_count",
            "head_type",
        ],
    )
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out = pd.get_dummies(out, columns=["head_type"], prefix="head")
    return out
