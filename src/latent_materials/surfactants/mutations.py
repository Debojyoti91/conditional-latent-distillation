"""Local surfactant mutation/gate functions used by the surfactant generation workflow.
No training or generation loop runs on import.
"""

import re
import numpy as np
from rdkit import Chem

OO_smarts = Chem.MolFromSmarts("O-O")
carbox_smarts = Chem.MolFromSmarts("C(=O)[O-]")
sulfate_smarts = Chem.MolFromSmarts("OS(=O)(=O)[O-]")
sulfonate_smarts = Chem.MolFromSmarts("S(=O)(=O)(O)")
quatN_smarts = Chem.MolFromSmarts("[N+](C)(C)(C)C")
ether_smarts = Chem.MolFromSmarts("OCCO")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def _normalize(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n


def valid_canon(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        try:
            Chem.SanitizeMol(
                m,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                    | Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                ),
            )
        except Exception:
            return None
    return Chem.MolToSmiles(m, canonical=True)


def mutate_chain(smi, delta=1):
    m = re.search(r"(C{4,})", smi)
    if not m:
        return None
    n = len(m.group(1))
    nn = max(2, n + delta)
    return smi[: m.start()] + ("C" * nn) + smi[m.end() :]


def add_EO(smi, k=1):
    pos = smi.find("O")
    if pos == -1:
        return None
    return smi[: pos + 1] + ("OCC" * k) + smi[pos + 1 :]


def remove_EO(smi, k=1):
    s = smi
    for _ in range(k):
        s2 = s.replace("OCC", "O", 1)
        if s2 == s:
            return None
        s = s2
    return s


def safe_add_EO_before_sulfate(smi, k=1):
    tag = "OS(=O)(=O)[O-]"
    pos = smi.rfind(tag)
    if pos == -1:
        return None
    pre = smi[:pos]  # ...O
    post = smi[pos + 1 :]  # S(=O)(=O)[O-]
    return pre + ("CCO" * k) + post


def safe_chain_delta_before_first_O(smi, delta=1):
    m = re.search(r"C{2,}O", smi)
    if not m:
        return None
    start, end = m.start(), m.end()
    run = re.search(r"C{2,}", smi[start:end])
    if not run:
        return None
    r0, r1 = start + run.start(), start + run.end()
    n = r1 - r0
    nn = max(2, n + delta)
    return smi[:r0] + ("C" * nn) + smi[r1:]


def has_peroxy(smi):
    m = Chem.MolFromSmiles(smi)
    return (m is not None) and (m.HasSubstructMatch(OO_smarts))


def detect_head(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if m.HasSubstructMatch(sulfate_smarts):
        return "sulfate"
    if m.HasSubstructMatch(carbox_smarts):
        return "carboxylate"
    if m.HasSubstructMatch(sulfonate_smarts):
        return "sulfonate"
    if m.HasSubstructMatch(quatN_smarts):
        return "quat_ammonium"
    if m.HasSubstructMatch(ether_smarts):
        return "ethoxylated"
    return None


def longest_chain_len(m):
    smi = Chem.MolToSmiles(m)
    runs = re.findall(r"C{1,}", smi)
    return max((len(r) for r in runs), default=0)


def ring_count(m):
    return m.GetRingInfo().NumRings()


def branching_index(m):
    degs = [a.GetTotalDegree() for a in m.GetAtoms()]
    return float(np.mean(degs)) if degs else 0.0


def structure_pass(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    if has_peroxy(smi):
        return False
    head = detect_head(smi)
    if head not in {
        "sulfate",
        "carboxylate",
        "sulfonate",
        "quat_ammonium",
        "ethoxylated",
    }:
        return False
    if longest_chain_len(m) < 6:
        return False
    if ring_count(m) > 2:
        return False
    if branching_index(m) > 3.5:
        return False
    return True


def to_canon(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        try:
            Chem.SanitizeMol(
                m,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                    | Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ),
            )
        except Exception:
            return None
    return Chem.MolToSmiles(m, canonical=True)


def try_repairs(smi):
    if has_peroxy(smi):
        return None  # hard reject peroxy; do not try to “fix” O–O

    c = to_canon(smi)
    if c and structure_pass(c):
        return c

    # EO fragment harmonization: mild replace sequences that often break
    # attempt: collapse accidental duplicated "O" neighbors like "OOCC" -> "OCC"
    s = smi.replace("OOCC", "OCC")
    c = to_canon(s)
    if c and structure_pass(c):
        return c

    # small tail adjustments near first O if valence/rings fail
    for dlt in (1, -1, 2, -2):
        r = safe_chain_delta_before_first_O(smi, dlt)
        if r:
            c = to_canon(r)
            if c and structure_pass(c):
                return c

    return None
