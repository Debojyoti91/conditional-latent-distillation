"""Export a selected ensemble and its actual predictions as one coherent version."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.swa_utils import AveragedModel


def _plain_metadata(value):
    """Keep checkpoint configuration compatible with weights-only loading."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        # RDKit enums subclass int; do not retain their pickle constructors.
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain_metadata(value.tolist())
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Unsupported checkpoint metadata type: " + type(value).__name__)


def _selected_model(model):
    while isinstance(model, AveragedModel):
        model = model.module
    return model


def scalar_metrics(truth, prediction):
    t = np.asarray(truth, dtype=float).reshape(-1)
    p = np.asarray(prediction, dtype=float).reshape(-1)
    if (
        len(t) < 2
        or t.shape != p.shape
        or not np.isfinite(t).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError(
            "Evaluation requires aligned finite arrays with at least two rows."
        )
    mse = float(mean_squared_error(t, p))
    return {
        "mae": float(mean_absolute_error(t, p)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(t, p)),
    }


def export_ensemble_snapshot(
    destination,
    *,
    models,
    config,
    aux_columns,
    aux_scaler,
    target_transformer,
    evaluations,
    metadata=None,
):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not models:
        raise ValueError("No selected models supplied.")
    models = [_selected_model(model) for model in models]
    config = _plain_metadata(config)
    metadata = _plain_metadata(metadata or {})
    metrics = {}
    tables = []
    for split, evaluation in evaluations.items():
        truth = np.asarray(evaluation["y_true"], float).reshape(-1)
        prediction = np.asarray(evaluation["y_pred"], float).reshape(-1)
        calculated = scalar_metrics(truth, prediction)
        for key in calculated:
            if key in evaluation and not np.isclose(
                calculated[key], evaluation[key], rtol=1e-5, atol=1e-9
            ):
                raise ValueError(
                    f"{split} {key} disagrees with the supplied prediction arrays."
                )
        metrics[split] = calculated
        tables.append(
            pd.DataFrame(
                {
                    "split": split,
                    "row_in_split": np.arange(len(truth)),
                    "y_true": truth,
                    "y_pred": prediction,
                }
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination.parent))
    try:
        artifact = {
            "config": config,
            "aux_cols": list(aux_columns),
            "joint_states": [
                {
                    key: value.detach().cpu().clone()
                    for key, value in m.state_dict().items()
                }
                for m in models
            ],
            "model_classes": [type(model).__name__ for model in models],
            "metadata": metadata or {},
        }
        torch.save(artifact, temporary / "ensemble.pt")
        joblib.dump(aux_scaler, temporary / "aux_scaler.pkl")
        joblib.dump(target_transformer, temporary / "target_transformer.pkl")
        pd.concat(tables, ignore_index=True).to_csv(
            temporary / "predictions.csv", index=False
        )
        (temporary / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(temporary.iterdir())
        }
        (temporary / "snapshot.json").write_text(
            json.dumps(
                {"schema_version": 1, "files": files, "metadata": metadata or {}},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
