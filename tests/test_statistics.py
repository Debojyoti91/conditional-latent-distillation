import numpy as np
import pytest
from latent_materials.statistics import (
    latent_metrics,
    bh_fdr,
    corrected_empirical_p,
    paired_bootstrap_improvement,
    cliffs_delta,
    permutation_p,
    shuffle_properties_within_partitions,
)


def test_latent_metrics_against_hand_calculation():
    result = latent_metrics([[0, 2], [4, 6]], [[1, 3], [2, 4]])
    assert result["test_mse"] == 2.5
    assert result["test_rmse"] == pytest.approx(np.sqrt(2.5))
    np.testing.assert_allclose(result["molecule_rmse"], [1, 2])


@pytest.mark.parametrize(
    "truth,prediction", [([], []), ([[1]], [[1, 2]]), ([[np.nan]], [[1]])]
)
def test_metrics_reject_invalid_input(truth, prediction):
    with pytest.raises(ValueError):
        latent_metrics(truth, prediction)


def test_bh_fdr_known_values_and_original_order():
    np.testing.assert_allclose(
        bh_fdr([0.04, 0.001, 0.03, 0.2]), [0.053333333333, 0.004, 0.053333333333, 0.2]
    )


@pytest.mark.parametrize("bad", [[-1], [1.1], [np.nan], [[0.1]]])
def test_bh_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        bh_fdr(bad)


def test_corrected_probability_includes_finite_sample_offset():
    assert corrected_empirical_p([2, 3, 4], 1) == 0.25
    assert corrected_empirical_p([2, 3, 4], 3) == 0.75


def test_bootstrap_paired_constant_improvement():
    r = paired_bootstrap_improvement([1, 2, 3], [3, 4, 5], n_bootstrap=200)
    assert r["mean_RMSE_improvement"] == 2
    assert r["CI95_low"] == r["CI95_high"] == 2
    assert r["bootstrap_fraction_improvement_le_0"] == 0


def test_effect_size_direction_and_ties():
    assert cliffs_delta([3, 4], [1, 2])[0] == 1
    assert cliffs_delta([1, 1], [1, 1])[0] == 0
    assert permutation_p([1, 2], [1, 2], n=100) == 1


def test_shuffling_preserves_pairs_and_partition_membership():
    p = np.column_stack([np.arange(20), np.arange(20) + 100])
    parts = [np.arange(10), np.arange(10, 20)]
    shuffled = shuffle_properties_within_partitions(p, parts, 7)
    np.testing.assert_array_equal(shuffled[:, 1] - shuffled[:, 0], 100)
    for ids in parts:
        assert set(shuffled[ids, 0]) == set(p[ids, 0])
    assert not np.array_equal(p, shuffled)
    np.testing.assert_array_equal(
        shuffled, shuffle_properties_within_partitions(p, parts, 7)
    )


def test_overlapping_shuffle_partitions_rejected():
    with pytest.raises(ValueError):
        shuffle_properties_within_partitions([[1, 2], [3, 4]], [[0, 1], [1]], 0)
