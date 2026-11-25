import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import AveragedModel
from latent_materials.exports import export_ensemble_snapshot


def arguments():
    model = torch.nn.Linear(2, 1)
    scaler = StandardScaler().fit([[0, 1], [2, 3]])
    return dict(
        models=[AveragedModel(model)],
        config={"width": np.int64(2)},
        aux_columns=["a", "b"],
        aux_scaler=scaler,
        target_transformer=scaler,
        evaluations={"test": {"y_true": [1.0, 3.0], "y_pred": [2.0, 2.0], "rmse": 1.0}},
    )


def test_snapshot_matches_selected_model_predictions_and_hashes(tmp_path):
    args = arguments()
    destination = tmp_path / "selected"
    export_ensemble_snapshot(destination, **args)
    artifact = torch.load(destination / "ensemble.pt", weights_only=True)
    for name, tensor in args["models"][0].module.state_dict().items():
        torch.testing.assert_close(artifact["joint_states"][0][name], tensor)
    predictions = pd.read_csv(destination / "predictions.csv")
    np.testing.assert_array_equal(predictions["y_pred"], [2.0, 2.0])
    assert json.loads((destination / "metrics.json").read_text())["test"]["rmse"] == 1.0
    for name, expected in json.loads((destination / "snapshot.json").read_text())[
        "files"
    ].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected


def test_inconsistent_metrics_rejected_before_any_export(tmp_path):
    args = arguments()
    args["evaluations"]["test"]["rmse"] = 0.1
    destination = tmp_path / "bad"
    with pytest.raises(ValueError, match="disagrees"):
        export_ensemble_snapshot(destination, **args)
    assert not destination.exists()


def test_existing_snapshot_is_never_overwritten(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        export_ensemble_snapshot(destination, **arguments())
