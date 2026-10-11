"""Graph encoder, reconstruction decoder and conductivity regressor."""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool, global_max_pool


class GNNEncoder(nn.Module):

    def __init__(self, in_dim, hidden, layers, dropout, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(last, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.bns.append(BatchNorm(hidden))
            last = hidden
        self.dropout = dropout
        self.out_dim = hidden

    def forward(self, x, edge_index, edge_attr):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class NodeFeatDecoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, h_nodes):
        return self.net(h_nodes)


class SEBlock(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(4, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return x * w


class DenseSERegressor(nn.Module):

    def __init__(self, in_dim, growth, n_blocks, dropout, head_hidden):
        super().__init__()
        layers = [
            nn.Sequential(
                nn.Linear(in_dim, growth),
                nn.BatchNorm1d(growth),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        ]
        for _ in range(n_blocks - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(growth, growth),
                    nn.BatchNorm1d(growth),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.Sequential(*layers)
        self.se = SEBlock(growth, reduction=8)
        self.head = nn.Sequential(
            nn.Linear(growth, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, z):
        h = self.blocks(z)
        h = self.se(h)
        return self.head(h)


class JointGraphAER(nn.Module):

    def __init__(self, node_in, edge_in, cfg, num_extra_feats=None):
        super().__init__()
        if num_extra_feats is None:
            raise ValueError(
                "Specify num_extra_feats explicitly for the fitted feature contract"
            )
        self.encoder = GNNEncoder(
            node_in, cfg["enc_hidden"], cfg["enc_layers"], cfg["enc_dropout"], edge_in
        )
        self.decoder = NodeFeatDecoder(cfg["enc_hidden"], node_in)
        self.num_extra_feats = int(num_extra_feats)
        self.use_mean_max_pool = bool(cfg.get("use_mean_max_pool", True))
        pool_multiplier = 2 if self.use_mean_max_pool else 1
        reg_input_dim = pool_multiplier * cfg["enc_hidden"] + self.num_extra_feats
        self.regressor = DenseSERegressor(
            reg_input_dim,
            cfg["growth"],
            cfg["n_blocks"],
            cfg["dropout"],
            cfg["head_hidden"],
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        h_nodes = self.encoder(x, edge_index, edge_attr)
        if self.use_mean_max_pool:
            z_graph = torch.cat(
                [global_mean_pool(h_nodes, batch), global_max_pool(h_nodes, batch)],
                dim=1,
            )
        else:
            z_graph = global_mean_pool(h_nodes, batch)
        if self.num_extra_feats > 0:
            num_feats = getattr(data, "num_feats", None)
            if num_feats is None or num_feats.numel() == 0:
                num_feats = torch.zeros(
                    z_graph.size(0),
                    self.num_extra_feats,
                    device=z_graph.device,
                    dtype=z_graph.dtype,
                )
            else:
                num_feats = num_feats.to(z_graph.device)
                if num_feats.dim() == 1:
                    num_feats = num_feats.unsqueeze(0)
                if num_feats.size(0) != z_graph.size(0):
                    num_feats = num_feats.view(z_graph.size(0), -1)
                if num_feats.size(1) < self.num_extra_feats:
                    pad = torch.zeros(
                        z_graph.size(0),
                        self.num_extra_feats - num_feats.size(1),
                        device=z_graph.device,
                        dtype=z_graph.dtype,
                    )
                    num_feats = torch.cat([num_feats, pad], dim=1)
                elif num_feats.size(1) > self.num_extra_feats:
                    num_feats = num_feats[:, : self.num_extra_feats]
            z_graph = torch.cat([z_graph, num_feats], dim=1)
        y_hat = self.regressor(z_graph)
        x_logits = self.decoder(h_nodes)
        return (y_hat, x_logits, h_nodes, z_graph)
