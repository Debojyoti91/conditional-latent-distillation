"""AEM historical mutation vocabulary and gate; charged headgroups use chemically valid valences."""

import random
from rdkit import Chem

OO_smarts = Chem.MolFromSmarts("O-O")
quatN_smarts = Chem.MolFromSmarts("[N+](C)(C)(C)C")
imidaz_smarts = Chem.MolFromSmarts("c1[n+](C)cnc1")
phos_smarts = Chem.MolFromSmarts("[P+](C)(C)(C)C")


def valid_canon(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    try:
        Chem.SanitizeMol(m)
    except:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def has_peroxy(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None and m.HasSubstructMatch(OO_smarts)


def has_cationic_head(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False

    return (
        m.HasSubstructMatch(quatN_smarts)
        or m.HasSubstructMatch(imidaz_smarts)
        or m.HasSubstructMatch(phos_smarts)
    )


def structure_pass(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False

    if has_peroxy(smi):
        return False

    # MUST have a cationic head group
    if not has_cationic_head(smi):
        return False

    return True


def add_quatN(smi):
    return smi + "[N+](C)(C)C"


def add_imidazolium(smi):
    return smi + "c1[n+](C)cn(C)c1"


def add_phosphonium(smi):
    return smi + "[P+](C)(C)C"


def extend_chain(smi, k=2):
    return smi + ("C" * k)


def insert_ether(smi):
    return smi.replace("CC", "COCC", 1)


def mutate_aem(smi):
    ops = [add_quatN, add_imidazolium, add_phosphonium, extend_chain, insert_ether]
    op = random.choice(ops)
    return op(smi)
