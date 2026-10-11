"""AEM atom/bond features and fitted-transformer graph assembly."""

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


def bond_feature_dim(cfg):
    return len(cfg["bond_types"]) + 1 + 2


def get_atom_feature_dims(cfg):
    bin_dim = (
        len(cfg["atom_nums"])
        + 1
        + (cfg["max_degree"] + 1 + 1)
        + (len(cfg["charges"]) + 1)
        + (len(cfg["hybs"]) + 1)
        + 2
    )
    cont_dim = 2
    return (bin_dim, cont_dim)


def atom_features(atom, cfg):
    nums, hybs, charges, max_deg = (
        cfg["atom_nums"],
        cfg["hybs"],
        cfg["charges"],
        cfg["max_degree"],
    )
    f_num = [1.0 if atom.GetAtomicNum() == z else 0.0 for z in nums] + [0.0]
    if sum(f_num[:-1]) == 0:
        f_num[-1] = 1.0
    d = atom.GetTotalDegree()
    f_deg = [1.0 if d == k else 0.0 for k in range(max_deg + 1)] + [
        1.0 if d > max_deg else 0.0
    ]
    ch = atom.GetFormalCharge()
    f_ch = [1.0 if ch == c else 0.0 for c in charges] + [
        1.0 if ch not in charges else 0.0
    ]
    hyb = atom.GetHybridization()
    f_hy = [1.0 if hyb == h else 0.0 for h in hybs] + [1.0 if hyb not in hybs else 0.0]
    f_misc = [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    f_cont = [
        float(atom.GetTotalNumHs(includeNeighbors=True)),
        float(atom.GetImplicitValence()),
    ]
    return np.array(f_num + f_deg + f_ch + f_hy + f_misc + f_cont, dtype=np.float32)


def bond_features(bond, cfg):
    btypes = cfg["bond_types"]
    f_bt = [1.0 if bond.GetBondType() == t else 0.0 for t in btypes] + [0.0]
    if sum(f_bt[:-1]) == 0:
        f_bt[-1] = 1.0
    f_misc = [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    return np.array(f_bt + f_misc, dtype=np.float32)


def conductivity_sample_weight(y_original, cfg):
    if not cfg.get("use_sample_weight", True):
        return 1.0
    return float(
        cfg.get("high_cond_weight", 3.0)
        if float(y_original) >= cfg.get("high_cond_threshold", 0.09)
        else 1.0
    )


def smiles_to_pyg(smiles, y_value, aux_vec_scaled, cfg):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=False)
    except Exception:
        pass
    x = torch.tensor(
        np.vstack([atom_features(a, cfg) for a in mol.GetAtoms()]), dtype=torch.float32
    )
    ei_src, ei_dst, eattr = ([], [], [])
    for b in mol.GetBonds():
        i, j = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        bf = bond_features(b, cfg)
        ei_src += [i, j]
        ei_dst += [j, i]
        eattr.append(bf)
        eattr.append(bf)
    if len(ei_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feature_dim(cfg)), dtype=torch.float32)
    else:
        edge_index = torch.tensor([ei_src, ei_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.vstack(eattr), dtype=torch.float32)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y_value)], dtype=torch.float32),
    )
    data.num_feats = torch.tensor(aux_vec_scaled, dtype=torch.float32)
    data.sample_weight = torch.tensor([1.0], dtype=torch.float32)
    data.y_original = torch.tensor([float(y_value)], dtype=torch.float32)
    data.smiles = str(smiles)
    return data


def build_graph_list(
    input_df,
    cfg,
    aux_scaler,
    target_transformer,
    transform_y=True,
    *,
    aux_columns,
    target_column,
):
    graphs, skipped = ([], 0)
    aux_scaled = aux_scaler.transform(input_df[aux_columns].values.astype(np.float32))
    y_original_vals = input_df[target_column].values.astype(np.float32)
    if transform_y:
        y_vals = target_transformer.transform(
            input_df[[target_column]].values.astype(np.float32)
        ).reshape(-1)
    else:
        y_vals = y_original_vals
    for smi, y, y_original, aux in zip(
        input_df["SMILES"].values, y_vals, y_original_vals, aux_scaled
    ):
        g = smiles_to_pyg(smi, y, aux.astype(np.float32), cfg)
        if g is None:
            skipped += 1
        else:
            g.y_original = torch.tensor([float(y_original)], dtype=torch.float32)
            g.sample_weight = torch.tensor(
                [conductivity_sample_weight(y_original, cfg)], dtype=torch.float32
            )
            graphs.append(g)
    print(f"Built {len(graphs)} graphs; skipped {skipped} invalid molecules.")
    return graphs
