import numpy as np
import pytest
from sklearn.preprocessing import PowerTransformer
from latent_materials.validation import (
    assert_disjoint_molecules,
    grouped_development_split,
    select_by_validation_rmse,
)
from latent_materials.inference import inverse_target_transform
from latent_materials.retrieval import anchor_indices


def test_identity_check_detects_different_smiles_for_same_molecule():
    with pytest.raises(ValueError, match="canonical"):
        assert_disjoint_molecules(["CCO"], ["OCC"])
    assert_disjoint_molecules(["CCO"], ["CCN"])


def test_grouped_split_keeps_all_duplicates_together():
    smiles = ["CCO", "OCC", "CCC", "CCN", "NCC", "CCCC", "CCCl", "CCBr"]
    train, val = grouped_development_split(smiles, validation_fraction=0.3)
    assert_disjoint_molecules(np.array(smiles)[train], np.array(smiles)[val])
    assert set(train) | set(val) == set(range(len(smiles)))


def test_selection_uses_only_validation_and_handles_ties():
    calls = []
    validation = object()

    def evaluate(models, graphs, config, *, label):
        calls.append((graphs, label))
        return {"rmse": models[0]}

    selected, index, scores = select_by_validation_rmse(
        [[3], [1], [1]], validation, {}, evaluate
    )
    assert selected == [1] and index == 1
    assert all(
        g is validation and label == "selection_validation" for g, label in calls
    )


def test_target_inverse_uses_fitted_training_transformer():
    original = np.array([0.01, 0.03, 0.08, 0.1, 0.2])[:, None]
    transformer = PowerTransformer().fit(original)
    np.testing.assert_allclose(
        inverse_target_transform(transformer.transform(original), transformer),
        original.ravel(),
    )
    with pytest.raises(ValueError):
        inverse_target_transform([1], PowerTransformer())


def test_anchor_cosine_similarity_and_first_tie():
    fingerprints = np.zeros((3, 2048))
    fingerprints[0, 0] = 1
    fingerprints[1, 1] = 1
    fingerprints[2, 0] = 1
    query = np.zeros((2, 2050))
    query[0, 0] = 5
    query[1, 1] = 8
    indices, similarity = anchor_indices(query, fingerprints)
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(similarity, 1, atol=1e-7)
