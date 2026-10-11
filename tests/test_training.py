import numpy as np
import pytest
import torch
from sklearn.preprocessing import PowerTransformer
from latent_materials.surfactants.models import smiles_to_graph, SearchGINRegressor
from latent_materials.surfactants.training import train_one_config
from latent_materials.aems.config import default_config
from latent_materials.aems.data import smiles_to_pyg, get_atom_feature_dims
from latent_materials.aems.training import (
    pretrain_encoder,
    train_single_joint,
    weighted_supervised_loss,
)


def test_surfactant_training_and_validation_checkpoint_smoke(tmp_path):
    torch.manual_seed(1)
    graphs = []
    for i, smiles in enumerate(["CCO", "CCN", "CCC", "CCCC", "CCCl", "CCBr"]):
        graph = smiles_to_graph(smiles, [float(i) / 6])
        graph.y = torch.tensor([float(i) / 6, float(i + 1) / 6])
        graphs.append(graph)
    cfg = dict(
        layer_sizes=[8, 8],
        dropout=0.0,
        lr=0.001,
        weight_decay=0.0,
        epochs=2,
        patience=2,
        batch_size=2,
    )
    model, metrics = train_one_config(
        cfg,
        graphs[:2],
        graphs[2:4],
        graphs[4:],
        "cpu",
        model_class=SearchGINRegressor,
        output_dir=tmp_path,
    )
    assert metrics["epochs_ran"] == 2
    assert np.isfinite(metrics["val_rmse_mean"])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_weighted_aem_loss_against_hand_calculation():
    loss = weighted_supervised_loss(
        torch.tensor([1.0, 3.0]), torch.zeros(2), torch.tensor([1.0, 3.0])
    )
    assert loss.item() == pytest.approx(3.75)


def test_aem_pretraining_joint_training_and_validation_smoke():
    torch.manual_seed(2)
    cfg = default_config()
    cfg.update(
        enc_hidden=8,
        enc_layers=1,
        growth=8,
        n_blocks=1,
        head_hidden=8,
        ae_epochs=1,
        epochs=2,
        batch_size=2,
        ae_batch_size=2,
    )
    values = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12])
    transformer = PowerTransformer().fit(values[:4, None])
    transformed = transformer.transform(values[:, None]).ravel()
    smiles = [
        "CC[N+](C)(C)C",
        "CCC[N+](C)(C)C",
        "CCCC[N+](C)(C)C",
        "CC[P+](C)(C)C",
        "CCC[P+](C)(C)C",
        "CCCC[P+](C)(C)C",
    ]
    graphs = [
        smiles_to_pyg(s, y, [float(i) / 6], cfg)
        for i, (s, y) in enumerate(zip(smiles, transformed))
    ]
    bin_dim, cont_dim = get_atom_feature_dims(cfg)
    initialization = pretrain_encoder(
        graphs[:4], cfg, bin_dim, cont_dim, seed=2, device="cpu"
    )
    model, history, evaluation = train_single_joint(
        graphs[:4],
        graphs[4:],
        initialization,
        cfg,
        bin_dim,
        cont_dim,
        device="cpu",
        target_transformer=transformer,
        aux_columns=["a"],
        base_seed=2,
    )
    assert len(history["epoch"]) == 2
    assert np.isfinite(evaluation["rmse"]) and len(evaluation["y_pred"]) == 2
    assert all(torch.isfinite(p).all() for p in model.parameters())
