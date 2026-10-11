"""Graph construction and explicitly configured surfactant datasets."""

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from .models import smiles_to_graph

atom_feats = [
    lambda a: a.GetAtomicNum(),
    lambda a: a.GetDegree(),
    lambda a: a.GetTotalNumHs(),
    lambda a: int(a.GetIsAromatic()),
]


def pc20_graph(smiles, aux_vec, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = []
    for a in mol.GetAtoms():
        x.append([f(a) for f in atom_feats])
    x = torch.tensor(x, dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        edges += [[i, j], [j, i]]
    if not edges:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    aux = torch.tensor(aux_vec, dtype=torch.float).view(1, -1)
    y = torch.tensor([float(y)], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y, aux=aux)


class SurfactantDataset(InMemoryDataset):
    """Build aligned graphs/targets using the supplied fitted feature columns."""

    def __init__(self, features, targets, *, aux_columns, target_columns):
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same row count")
        self.X = features.reset_index(drop=True)
        self.Y = targets.reset_index(drop=True)
        super().__init__(".")
        graphs = []
        skipped = 0
        for index, (smiles, values) in enumerate(
            zip(self.X["SMILES"], self.X[list(aux_columns)].values)
        ):
            graph = smiles_to_graph(smiles, values.astype(np.float32))
            if graph is None:
                skipped += 1
                continue
            graph.y = torch.tensor(
                self.Y.iloc[index][list(target_columns)].values, dtype=torch.float
            )
            graph.smiles = smiles
            graphs.append(graph)
        if not graphs:
            raise ValueError("No valid molecular graphs")
        self.data, self.slices = self.collate(graphs)
        self.skipped_invalid = skipped
