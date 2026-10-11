"""Structure-only heteroscedastic GNN components for the surfactant surrogate.
These tests verify graph shapes, batching, loss and gradients, not fitted performance.
"""

import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType as Hyb
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, MLP

ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 11, 3]
HYB_LIST = [Hyb.SP, Hyb.SP2, Hyb.SP3]


def atom_feat(a):
    Z = a.GetAtomicNum()
    z_one = [1 if Z == z else 0 for z in ATOM_LIST] + [int(Z not in ATOM_LIST)]
    hyb = a.GetHybridization()
    hyb_one = [1 if hyb == h else 0 for h in HYB_LIST] + [int(hyb not in HYB_LIST)]
    x = [
        a.GetTotalDegree(),
        a.GetFormalCharge(),
        int(a.GetIsAromatic()),
        int(a.IsInRing()),
        a.GetTotalNumHs(includeNeighbors=True),
    ]
    return torch.tensor(z_one + hyb_one + x, dtype=torch.float)


def smiles_to_graph(smiles, aux_vec):
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = torch.stack([atom_feat(a) for a in m.GetAtoms()], dim=0)
    src, dst = ([], [])
    for b in m.GetBonds():
        i, j = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        src += [i, j]
        dst += [j, i]
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    aux = torch.tensor(aux_vec, dtype=torch.float).unsqueeze(0)
    return Data(x=x, edge_index=edge_index, aux=aux)


def gaussian_nll(mu, logvar, y):
    if y.shape != mu.shape:
        y = y.view_as(mu)
    if logvar.shape != mu.shape:
        logvar = logvar.view_as(mu)
    logvar = torch.clamp(logvar, -10, 10)
    inv_var = torch.exp(-logvar)
    return 0.5 * ((y - mu) ** 2 * inv_var + logvar).sum(dim=1).mean()


import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PC20GraphRegressor(nn.Module):

    def __init__(self, h, aux_dim):
        super().__init__()
        self.g1 = GCNConv(4, h)
        self.g2 = GCNConv(h, h)
        self.g3 = GCNConv(h, h)
        self.fc1 = nn.Linear(h + aux_dim, h)
        self.fc2 = nn.Linear(h, h // 2)
        self.fc3 = nn.Linear(h // 2, 1)

    def forward(self, data):
        x = F.relu(self.g1(data.x, data.edge_index))
        x = F.relu(self.g2(x, data.edge_index))
        x = F.relu(self.g3(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        x = torch.cat([x, data.aux], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).view(-1)


class SearchGINRegressor(nn.Module):

    def __init__(
        self, atom_dim, aux_dim, layer_sizes=(128, 128, 128), dropout=0.1, head_dim=64
    ):
        super().__init__()
        layers = []
        in_dim = atom_dim
        for h in layer_sizes:
            layers.append(GINConv(MLP([in_dim, h, h], act="relu")))
            in_dim = h
        self.g_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        last_h = layer_sizes[-1]
        mid = max(last_h // 2, head_dim)
        self.proj = nn.Sequential(
            nn.Linear(last_h + aux_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(head_dim, 4)

    def forward(self, data):
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        h = x
        for g in self.g_layers:
            h = g(h, edge_index)
            h = self.dropout(h)
        g = global_mean_pool(h, batch)
        aux = data.aux
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        if aux.size(0) != g.size(0):
            aux = aux.view(g.size(0), -1)
        z = torch.cat([g, aux], dim=1)
        z = self.proj(z)
        out = self.head(z)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        mu, logvar = (out[:, :2], out[:, 2:])
        return (mu, logvar)


class RefinedGINRegressor(nn.Module):

    def __init__(
        self, atom_dim, aux_dim, layer_sizes=(128, 128, 128), dropout=0.1, head_dim=64
    ):
        super().__init__()
        layers = []
        in_dim = atom_dim
        for h in layer_sizes:
            layers.append(GINConv(MLP([in_dim, h, h], act="relu")))
            in_dim = h
        self.g_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        last_h = layer_sizes[-1]
        mid = max(last_h // 2, head_dim)
        self.proj = nn.Sequential(
            nn.Linear(last_h + aux_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(head_dim, 4)

    def forward(self, data):
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        h = x
        for g in self.g_layers:
            h = g(h, edge_index)
            h = self.dropout(h)
        g = global_mean_pool(h, batch)
        aux = data.aux
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        if aux.size(0) != g.size(0):
            aux = aux.view(g.size(0), -1)
        z = torch.cat([g, aux], dim=1)
        z = self.proj(z)
        out = self.head(z)
        mu, logvar = (out[:, :2], out[:, 2:])
        return (mu, logvar)


class GINHetero(torch.nn.Module):

    def __init__(self, atom_dim, aux_dim, layer_sizes, dropout, head_dim=64):
        super().__init__()
        from torch_geometric.nn import GINConv, MLP
        import torch.nn as nn

        layers = []
        in_dim = atom_dim
        for h in layer_sizes:
            layers.append(GINConv(MLP([in_dim, h, h], act="relu")))
            in_dim = h
        self.g_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        last_h = layer_sizes[-1]
        mid = max(last_h // 2, head_dim)
        self.proj = nn.Sequential(
            nn.Linear(last_h + aux_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(head_dim, 4)

    def forward(self, data):
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        h = x
        for g in self.g_layers:
            h = g(h, edge_index)
            h = self.dropout(h)
        from torch_geometric.nn import global_mean_pool

        g = global_mean_pool(h, batch)
        aux = data.aux
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        if aux.size(0) != g.size(0):
            aux = aux.view(g.size(0), -1)
        z = torch.cat([g, aux], dim=1)
        z = self.proj(z)
        out = self.head(z)
        mu, logvar = (out[:, :2], out[:, 2:])
        return (mu, logvar)


class StructureOnlyGIN(nn.Module):

    def __init__(
        self,
        atom_dim,
        aux_dim,
        layer_sizes=(64, 128, 256, 128, 64),
        dropout=0.0,
        head_dim=64,
    ):
        super().__init__()
        layers = []
        in_dim = atom_dim
        for h in layer_sizes:
            layers.append(GINConv(MLP([in_dim, h, h], act="relu")))
            in_dim = h
        self.g_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        last_h = layer_sizes[-1]
        mid = max(last_h // 2, head_dim)
        self.proj = nn.Sequential(
            nn.Linear(last_h + aux_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(head_dim, 4)

    def forward(self, data):
        x, ei, batch = (data.x, data.edge_index, data.batch)
        h = x
        for g in self.g_layers:
            h = g(h, ei)
            h = self.dropout(h)
        g = global_mean_pool(h, batch)
        aux = data.aux
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        if aux.size(0) != g.size(0):
            aux = aux.view(g.size(0), -1)
        z = torch.cat([g, aux], dim=1)
        z = self.proj(z)
        out = self.head(z)
        mu, logvar = (out[:, :2], out[:, 2:])
        return (mu, logvar)
