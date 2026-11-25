import pytest
import torch
from torch_geometric.data import Data, Batch
from latent_materials.surfactants.models import smiles_to_graph, GINHetero, gaussian_nll
from latent_materials.aems.models import JointGraphAER
from latent_materials.adapter_model import ConditionalAdapter


def test_surfactant_batch_shapes_and_gradients():
    torch.manual_seed(2)
    graphs = [smiles_to_graph(s, [0.0, 1.0]) for s in ["CCO", "CCN"]]
    batch = Batch.from_data_list(graphs)
    model = GINHetero(
        graphs[0].x.shape[1], 2, layer_sizes=(8, 8), dropout=0.0, head_dim=8
    )
    mu, logvar = model(batch)
    assert mu.shape == logvar.shape == (2, 2)
    loss = gaussian_nll(mu, logvar, torch.zeros_like(mu))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_gaussian_loss_known_value():
    assert (
        gaussian_nll(torch.ones(2, 2), torch.zeros(2, 2), torch.zeros(2, 2)).item() == 1
    )


def test_aem_batch_shapes_and_gradients():
    torch.manual_seed(3)
    cfg = dict(
        enc_hidden=8,
        enc_layers=2,
        enc_dropout=0.0,
        growth=4,
        n_blocks=2,
        dropout=0.0,
        head_hidden=8,
    )
    graphs = [
        Data(
            x=torch.randn(3, 5),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_attr=torch.ones(4, 2),
            num_feats=torch.ones(1, 3),
        )
        for _ in range(2)
    ]
    model = JointGraphAER(5, 2, cfg, num_extra_feats=3)
    prediction, reconstruction, nodes, embedding = model(Batch.from_data_list(graphs))
    assert prediction.numel() == 2
    assert reconstruction.shape == (6, 5)
    assert embedding.shape == (2, 19)
    loss = prediction.square().mean() + reconstruction.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_adapter_learning_reduces_tiny_problem_loss():
    torch.manual_seed(4)
    model = ConditionalAdapter(3, latent_dim=2, hidden=8)
    x = torch.randn(12, 3)
    y = x[:, :2]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = (model(x) - y).square().mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
    assert (model(x) - y).square().mean().item() < initial / 3


def test_real_aem_graph_features_and_model_are_compatible():
    from latent_materials.aems.config import default_config
    from latent_materials.aems.data import (
        smiles_to_pyg,
        get_atom_feature_dims,
        bond_feature_dim,
    )

    cfg = default_config()
    cfg.update(enc_hidden=8, enc_layers=1, growth=8, n_blocks=1, head_hidden=8)
    graphs = [
        smiles_to_pyg(s, 0.1, [0.0, 1.0], cfg)
        for s in ["CCC[N+](C)(C)C", "CC[P+](C)(C)C"]
    ]
    bin_dim, cont_dim = get_atom_feature_dims(cfg)
    assert all(g.x.shape[1] == bin_dim + cont_dim for g in graphs)
    model = JointGraphAER(
        bin_dim + cont_dim, bond_feature_dim(cfg), cfg, num_extra_feats=2
    )
    result = model(Batch.from_data_list(graphs))
    assert result[0].numel() == 2 and torch.isfinite(result[0]).all()
    assert smiles_to_pyg("invalid", 0.1, [0.0, 1.0], cfg) is None


def test_surfactant_dataset_preserves_alignment_when_invalid_graph_skipped():
    import pandas as pd
    from latent_materials.surfactants.data import SurfactantDataset

    x = pd.DataFrame({"SMILES": ["CCO", "invalid", "CCN"], "a": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"s": [10.0, 20.0, 30.0], "c": [2.0, 4.0, 6.0]})
    dataset = SurfactantDataset(x, y, aux_columns=["a"], target_columns=["s", "c"])
    assert len(dataset) == 2 and dataset.skipped_invalid == 1
    torch.testing.assert_close(dataset[1].y, torch.tensor([30.0, 6.0]))
    torch.testing.assert_close(dataset[1].aux, torch.tensor([[3.0]]))
    assert smiles_to_graph("", [1.0]) is None


def test_structure_only_generation_model_predicts_two_properties():
    from latent_materials.surfactants.models import StructureOnlyGIN

    graphs = [smiles_to_graph(s, []) for s in ["CCO", "CCN"]]
    model = StructureOnlyGIN(
        graphs[0].x.shape[1], 0, layer_sizes=(8, 8), head_dim=8
    ).eval()
    mean, log_variance = model(Batch.from_data_list(graphs))
    assert mean.shape == log_variance.shape == (2, 2)
    assert torch.isfinite(mean).all() and torch.isfinite(log_variance).all()
