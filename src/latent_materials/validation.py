"""Validation-only selection and molecular-identity safeguards for new experiments."""

import numpy as np
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit


def canonical_identities(smiles):
    identities = []
    for s in smiles:
        molecule = Chem.MolFromSmiles(str(s))
        if molecule is None:
            raise ValueError("Invalid SMILES: " + str(s))
        identities.append(Chem.MolToSmiles(molecule, canonical=True))
    return np.array(identities, dtype=object)


def assert_disjoint_molecules(left, right):
    shared = set(canonical_identities(left)) & set(canonical_identities(right))
    if shared:
        raise ValueError(
            "Shared canonical molecular identities: " + str(sorted(shared))
        )


def grouped_development_split(smiles, validation_fraction=0.1, seed=42):
    groups = canonical_identities(smiles)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    return next(
        GroupShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=seed
        ).split(np.arange(len(groups)), groups=groups)
    )


def select_by_validation_rmse(candidates, validation_graphs, config, evaluator):
    if not candidates:
        raise ValueError("No candidate ensembles")
    evaluations = [
        evaluator(models, validation_graphs, config, label="selection_validation")
        for models in candidates
    ]
    errors = np.array([float(e["rmse"]) for e in evaluations])
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Invalid validation RMSE")
    selected = int(np.argmin(errors))
    return candidates[selected], selected, evaluations
