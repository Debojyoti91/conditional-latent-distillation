"""Surfactant constrained-library generation, structure-only guidance and conditional-adapter mutation expansion."""

from pathlib import Path


def run(workspace):
    """Run generation in a prepared writable workspace with the required teacher and split inputs."""
    root = Path(workspace).resolve()
    import os, re, pandas as pd
    from rdkit import Chem

    BASE_DIR = str(root)
    DATA_DIR = f"{BASE_DIR}/data"
    OUT_DIR = f"{BASE_DIR}/outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    train_csv = f"{DATA_DIR}/surfpro_train80.csv"
    test_csv = f"{DATA_DIR}/surfpro_test20.csv"
    assert os.path.exists(train_csv) and os.path.exists(
        test_csv
    ), "Run your split first."
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    PEROXY_SMARTS = Chem.MolFromSmarts("[O;X2]-[O;X2]")

    def canon(smi: str):
        m = Chem.MolFromSmiles(smi)
        if not m:
            return None
        try:
            Chem.SanitizeMol(m)
        except Exception:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    def has_peroxy(smi: str) -> bool:
        m = Chem.MolFromSmiles(smi)
        if not m:
            return True
        return m.HasSubstructMatch(PEROXY_SMARTS)

    BLOCK_REGEX = re.compile("O(?:(?:CCO)|(?:CC\\(C\\)O))+(?=S\\()")

    def eopo_block_info(smi: str):
        """Return eo_count, po_count, contiguous_block_bool, has_sulfate_bool."""
        has_sulf = "S(" in smi
        m = BLOCK_REGEX.search(smi)
        if not m:
            return (0, 0, False, has_sulf)
        block = smi[m.start() : m.end()]
        tail_left = smi[: m.start()]
        po = len(re.findall("CC\\(C\\)O", block))
        tmp = re.sub("CC\\(C\\)O", "", block)
        eo = len(re.findall("CCO", tmp))
        contiguous = "O" not in tail_left
        return (eo, po, contiguous, has_sulf)

    def valid_feasible(smi: str):
        c = canon(smi)
        if not c:
            return (None, (0, 0, False, False))
        if has_peroxy(c):
            return (None, (0, 0, False, False))
        eo, po, contig, has_s = eopo_block_info(c)
        ok = has_s and contig and (eo >= 1) and (po >= 1)
        return (c if ok else None, (eo, po, contig, has_s))

    def build_candidate(
        chain_len: int,
        eo: int,
        po: int,
        order: str = "EO-PO",
        counterion: str = "[Na+]",
    ):
        """
        Linear tail: C{chain_len}, contiguous block: O + EO* + PO* (order controlled), head: S(=O)(=O)[O-]
        Salt paired with counterion (default Na+).
        """
        assert chain_len >= 4 and eo >= 1 and (po >= 1)
        if order.upper() == "EO-PO":
            block = "O" + "CCO" * eo + "CC(C)O" * po
        else:
            block = "O" + "CC(C)O" * po + "CCO" * eo
        smi = "C" * chain_len + block + "S(=O)(=O)[O-]." + counterion
        return smi

    seen = set()
    for df in (train, test):
        if "SMILES" in df.columns:
            for s in df["SMILES"].dropna().astype(str):
                c = canon(s)
                if c:
                    seen.add(c)
    CHAIN_LENS = list(range(10, 19))
    EO_RANGE = list(range(1, 8))
    PO_RANGE = list(range(1, 8))
    ORDERS = ["EO-PO", "PO-EO"]
    COUNTERION = "[Na+]"
    rows = []
    for nC in CHAIN_LENS:
        for eo in EO_RANGE:
            for po in PO_RANGE:
                for ord_ in ORDERS:
                    raw = build_candidate(nC, eo, po, ord_, COUNTERION)
                    c, (ceo, cpo, contig, has_s) = valid_feasible(raw)
                    if not c:
                        continue
                    if c in seen:
                        continue
                    rows.append(
                        {
                            "SMILES": c,
                            "chain_len": nC,
                            "EO_block_count": ceo,
                            "PO_block_count": cpo,
                            "order": ord_,
                            "counterion": COUNTERION,
                        }
                    )
    lib = pd.DataFrame(rows).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    out_csv = f"{OUT_DIR}/genlib_candidates_v1.csv"
    lib.to_csv(out_csv, index=False)
    print(
        f"Generated combos (pre-filter): {len(CHAIN_LENS) * len(EO_RANGE) * len(PO_RANGE) * len(ORDERS)}"
    )
    print(f"Feasible, valid, novel candidates: {len(lib)}")
    print(f"Saved → {out_csv}")
    print(lib.head(10).to_string(index=False))
    import os, re, json, numpy as np, pandas as pd
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType as Hyb
    import torch, torch.nn as nn
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import joblib
    import matplotlib.pyplot as plt

    BASE_DIR = str(root)
    DATA_DIR = f"{BASE_DIR}/data"
    OUT_DIR = f"{BASE_DIR}/outputs"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    RAW = f"{DATA_DIR}/surfpro_imputed.csv"
    CANDS = f"{OUT_DIR}/genlib_candidates_v1.csv"
    assert os.path.exists(RAW), f"Missing {RAW}"
    assert os.path.exists(CANDS), f"Missing {CANDS} (run Step 6A first)"

    def longest_chain(smiles):
        try:
            return max((len(c) for c in re.findall("C+", smiles)))
        except:
            return np.nan

    def count_EO(smiles):
        return len(re.findall("CCO", smiles))

    def aliph_frac(mol):
        try:
            carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            alip = sum((1 for a in carbons if not a.GetIsAromatic()))
            return float(alip) / len(carbons)
        except:
            return 0.0

    def ring_count(mol):
        try:
            return float(mol.GetRingInfo().NumRings())
        except:
            return 0.0

    def branching_index(mol):
        try:
            carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            return float(np.mean([a.GetTotalDegree() for a in carbons]))
        except:
            return 0.0

    def detect_head(smiles):
        s = smiles.lower()
        if "s(=o)(=o)o" in s or "so3" in s:
            return "sulfate"
        if "s(=o)(=o)" in s and "o" in s:
            return "sulfonate"
        if "coo" in s:
            return "carboxylate"
        if "n+" in s or "[n+" in s:
            return "quat_ammonium"
        if re.search("O(?:CCO|CC\\(C\\)O)+S\\(", s):
            return "ethoxylated"
        return "other"

    STRUCT_NUMERIC = [
        "longest_chain",
        "EO_count",
        "aliphatic_fraction",
        "ring_count",
        "branching_index",
    ]
    HEAD_COL = "head_type"
    df = pd.read_csv(RAW).rename(columns={"types": "type"})
    assert "SMILES" in df.columns, "No SMILES column found"
    rows = []
    for smi in df["SMILES"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(
                {
                    "SMILES": smi,
                    "type": "unk",
                    **{k: np.nan for k in STRUCT_NUMERIC},
                    HEAD_COL: "other",
                }
            )
            continue
        feat = {
            "SMILES": smi,
            "type": "orig",
            "longest_chain": longest_chain(smi),
            "EO_count": count_EO(smi),
            "aliphatic_fraction": aliph_frac(mol),
            "ring_count": ring_count(mol),
            "branching_index": branching_index(mol),
            HEAD_COL: detect_head(smi),
        }
        rows.append(feat)
    X_struct = pd.DataFrame(rows)
    Y = df[["AW_ST_CMC", "pCMC"]].copy()
    X_struct = pd.get_dummies(X_struct, columns=[HEAD_COL], prefix="head")
    train, test, y_train, y_test = train_test_split(
        X_struct,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=X_struct["type"] if "type" in X_struct.columns else None,
    )
    numeric_cols = [c for c in STRUCT_NUMERIC if c in X_struct.columns]
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[numeric_cols] = scaler.fit_transform(
        train[numeric_cols].fillna(train[numeric_cols].median())
    )
    test_scaled[numeric_cols] = scaler.transform(
        test[numeric_cols].fillna(train[numeric_cols].median())
    )
    train_scaled.to_csv(f"{DATA_DIR}/X_train_struct_scaled.csv", index=False)
    test_scaled.to_csv(f"{DATA_DIR}/X_test_struct_scaled.csv", index=False)
    y_train.to_csv(f"{DATA_DIR}/y_train_struct.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/y_test_struct.csv", index=False)
    joblib.dump(scaler, f"{DATA_DIR}/scaler_struct.pkl")
    print("[OK] Rebuilt structure-only features.")
    print("Train X:", train_scaled.shape, "Test X:", test_scaled.shape)
    print("Numeric columns scaled:", numeric_cols)
    print(
        "Head one-hots (train):", [c for c in X_struct.columns if c.startswith("head_")]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        if m is None:
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

    AUX_COLS = [c for c in train_scaled.columns if c not in ["SMILES", "type"]]
    print("[INFO] AUX_COLS (structure-only):", AUX_COLS)

    class SurfDataset(InMemoryDataset):

        def __init__(self, X, Y, aux_cols):
            self.X = X.reset_index(drop=True).copy()
            self.Y = Y.reset_index(drop=True).copy()
            self.aux_cols = list(aux_cols)
            for c in self.aux_cols:
                if self.X[c].dtype == object:
                    self.X[c] = pd.to_numeric(self.X[c], errors="coerce").fillna(0.0)
            super().__init__(".")
            data_list = []
            bad = 0
            for i, (smi, aux_row) in enumerate(
                zip(self.X["SMILES"], self.X[self.aux_cols].values)
            ):
                g = smiles_to_graph(smi, aux_row.astype(np.float32))
                if g is None:
                    bad += 1
                    continue
                y = torch.tensor(
                    self.Y.iloc[i][["AW_ST_CMC", "pCMC"]].values, dtype=torch.float
                )
                g.y = y
                g.smiles = smi
                data_list.append(g)
            self.data, self.slices = self.collate(data_list)
            if bad:
                print(f"[WARN] Skipped {bad} invalid molecules")

    tr_ds = SurfDataset(train_scaled, y_train, AUX_COLS)
    te_ds = SurfDataset(test_scaled, y_test, AUX_COLS)
    from torch.utils.data import random_split

    n_val = max(1, int(len(tr_ds) * 0.1))
    n_tr = len(tr_ds) - n_val
    tr_sub, va_sub = random_split(
        tr_ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42)
    )
    tr_ld = DataLoader(tr_sub, batch_size=64, shuffle=True)
    va_ld = DataLoader(va_sub, batch_size=128, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=128, shuffle=False)
    from latent_materials.surfactants.models import StructureOnlyGIN as GINHetero

    def gaussian_nll(mu, logvar, y):
        if y.shape != mu.shape:
            y = y.view_as(mu)
        if logvar.shape != mu.shape:
            logvar = logvar.view_as(mu)
        logvar = torch.clamp(logvar, -10, 10)
        inv_var = torch.exp(-logvar)
        return 0.5 * ((y - mu) ** 2 * inv_var + logvar).sum(dim=1).mean()

    @torch.no_grad()
    def eval_loader(model, ld):
        model.eval()
        preds, trues = ([], [])
        for d in ld:
            d = d.to(device)
            mu, _ = model(d)
            y = d.y
            if y.shape != mu.shape:
                y = y.view_as(mu)
            preds.append(mu.cpu().numpy())
            trues.append(y.cpu().numpy())
        P, T = (np.vstack(preds), np.vstack(trues))
        rmse = np.sqrt(np.mean((P - T) ** 2, axis=0))
        mae = np.mean(np.abs(P - T), axis=0)
        r2 = [r2_score(T[:, i], P[:, i]) for i in range(2)]
        return (rmse, mae, r2)

    sample = next(iter(tr_ld))
    atom_dim, aux_dim = (sample.x.size(1), sample.aux.size(1))
    model = GINHetero(atom_dim, aux_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, wait, PATIENCE = (1000000000.0, 0, 25)
    for epoch in range(1, 201):
        model.train()
        total = 0.0
        steps = 0
        for d in tr_ld:
            d = d.to(device)
            mu, logvar = model(d)
            loss = gaussian_nll(mu, logvar, d.y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        tr_loss = total / max(1, steps)
        va_rmse, _, _ = eval_loader(model, va_ld)
        val = float(np.mean(va_rmse))
        if val < best:
            best = val
            wait = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    tr_rmse, tr_mae, tr_r2 = eval_loader(model, tr_ld)
    va_rmse, va_mae, va_r2 = eval_loader(model, va_ld)
    te_rmse, te_mae, te_r2 = eval_loader(model, te_ld)
    print("[DONE] Train/Val/Test RMSE:", tr_rmse, va_rmse, te_rmse)
    torch.save(model.state_dict(), f"{DATA_DIR}/gnn_struct_only.pt")
    cand = pd.read_csv(CANDS)
    assert "SMILES" in cand.columns
    rows = []
    for smi in cand["SMILES"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(
                {
                    "SMILES": smi,
                    **{k: np.nan for k in STRUCT_NUMERIC},
                    HEAD_COL: "other",
                }
            )
            continue
        rows.append(
            {
                "SMILES": smi,
                "longest_chain": longest_chain(smi),
                "EO_count": count_EO(smi),
                "aliphatic_fraction": aliph_frac(mol),
                "ring_count": ring_count(mol),
                "branching_index": branching_index(mol),
                HEAD_COL: detect_head(smi),
            }
        )
    augX = pd.DataFrame(rows)
    augX = pd.get_dummies(augX, columns=[HEAD_COL], prefix="head")
    for col in [c for c in train_scaled.columns if c.startswith("head_")]:
        if col not in augX.columns:
            augX[col] = 0.0
    extra_heads = [
        c
        for c in augX.columns
        if c.startswith("head_") and c not in train_scaled.columns
    ]
    if extra_heads:
        augX.drop(columns=extra_heads, inplace=True)
    augX[numeric_cols] = augX[numeric_cols].fillna(train[numeric_cols].median())
    scaler = joblib.load(f"{DATA_DIR}/scaler_struct.pkl")
    augX[numeric_cols] = scaler.transform(augX[numeric_cols])
    AUX_COLS = [c for c in train_scaled.columns if c not in ["SMILES", "type"]]

    def smiles_to_graph2(smi, aux_row):
        m = Chem.MolFromSmiles(smi)
        if m is None:
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
        aux = torch.tensor(aux_row, dtype=torch.float).unsqueeze(0)
        return Data(x=x, edge_index=edge_index, aux=aux)

    data_list = []
    for smi, aux_row in zip(augX["SMILES"], augX[AUX_COLS].values.astype(np.float32)):
        g = smiles_to_graph2(smi, aux_row)
        if g is not None:
            g.smiles = smi
            data_list.append(g)
    ld = DataLoader(data_list, batch_size=128, shuffle=False)
    model.eval()
    preds, sigmas = ([], [])
    with torch.no_grad():
        for d in ld:
            d = d.to(device)
            mu, logvar = model(d)
            sigma = torch.sqrt(torch.exp(torch.clamp(logvar, -10, 10)))
            preds.append(mu.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())
    P = np.vstack(preds)
    S = np.vstack(sigmas)
    cand_pred = pd.DataFrame(
        {
            "SMILES": [g.smiles for g in data_list],
            "AW_ST_CMC_pred": P[:, 0],
            "pCMC_pred": P[:, 1],
            "AW_ST_CMC_sigma": S[:, 0],
            "pCMC_sigma": S[:, 1],
        }
    )
    cand_pred = cand.merge(cand_pred, on="SMILES", how="inner")
    p1 = f"{OUT_DIR}/candidates_with_predictions.csv"
    cand_pred.to_csv(p1, index=False)
    print(f"[OK] Saved predictions:  {p1}")
    ytr = pd.read_csv(f"{DATA_DIR}/y_train_struct.csv")
    pCMC_min, pCMC_max = np.quantile(ytr["pCMC"], [0.1, 0.95])
    aw_min, aw_max = (25.0, 45.0)
    mask = (
        (cand_pred["pCMC_pred"] >= pCMC_min)
        & (cand_pred["pCMC_pred"] <= pCMC_max)
        & (cand_pred["AW_ST_CMC_pred"] >= aw_min)
        & (cand_pred["AW_ST_CMC_pred"] <= aw_max)
    )
    filtered = cand_pred[mask].copy().reset_index(drop=True)
    p2 = f"{OUT_DIR}/candidates_filtered_by_properties.csv"
    filtered.to_csv(p2, index=False)
    print("\n PROPERTY SCREEN COMPLETE: ")
    print(f"Saved predictions: {p1}")
    print(f"Saved filtered set: {p2}")
    print(
        f"Filters applied: {{'pCMC_range': ({pCMC_min:.3f}, {pCMC_max:.3f}), 'AW_ST_CMC_range': ({aw_min}, {aw_max})}}"
    )
    print(
        f"Kept {len(filtered)} / {len(cand_pred)} ({100.0 * len(filtered) / max(1, len(cand_pred)):.1f}%)"
    )
    TRAIN_STRUCT_SCALED = f"{DATA_DIR}/X_train_struct_scaled.csv"
    SCALER_STRUCT = f"{DATA_DIR}/scaler_struct.pkl"
    GNN_CKPT = f"{DATA_DIR}/gnn_struct_only.pt"
    CANDS_CSV = f"{OUT_DIR}/genlib_candidates_v1.csv"
    assert os.path.exists(TRAIN_STRUCT_SCALED), "Missing X_train_struct_scaled.csv"
    assert os.path.exists(SCALER_STRUCT), "Missing scaler_struct.pkl"
    assert os.path.exists(GNN_CKPT), "Missing gnn_struct_only.pt "
    assert os.path.exists(CANDS_CSV), "Missing genlib_candidates_v1.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def longest_chain(smiles):
        try:
            return max((len(c) for c in re.findall("C+", smiles)))
        except:
            return np.nan

    def count_EO(smiles):
        return len(re.findall("CCO", smiles))

    def aliph_frac(mol):
        try:
            carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            alip = sum((1 for a in carbons if not a.GetIsAromatic()))
            return float(alip) / len(carbons)
        except:
            return 0.0

    def ring_count(mol):
        try:
            return float(mol.GetRingInfo().NumRings())
        except:
            return 0.0

    def branching_index(mol):
        try:
            carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
            if not carbons:
                return 0.0
            return float(np.mean([a.GetTotalDegree() for a in carbons]))
        except:
            return 0.0

    def detect_head(smiles):
        s = smiles.lower()
        if "s(=o)(=o)o" in s or "so3" in s:
            return "sulfate"
        if "s(=o)(=o)" in s and "o" in s:
            return "sulfonate"
        if "coo" in s:
            return "carboxylate"
        if "n+" in s or "[n+" in s:
            return "quat_ammonium"
        if re.search("O(?:CCO|CC\\(C\\)O)+S\\(", s):
            return "ethoxylated"
        return "other"

    STRUCT_NUMERIC = [
        "longest_chain",
        "EO_count",
        "aliphatic_fraction",
        "ring_count",
        "branching_index",
    ]
    HEAD_COL = "head_type"

    def has_peroxy(smiles):
        return "OO" in re.sub("=O|O=|S\\(=O\\)\\(=O\\)", "", smiles)

    def structural_flags(smiles):
        """
        Hard structural constraints for surfactant-likeness.
        These are light heuristics — tune thresholds to your chemistry.
        """
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return {
                "flag_valid_smiles": False,
                "flag_peroxy_banned": False,
                "flag_has_head": False,
                "flag_hydrophobe_len": False,
                "flag_ring_ok": False,
                "flag_branch_ok": False,
                "structure_pass": False,
            }
        head = detect_head(smiles)
        has_head = head in {
            "sulfate",
            "sulfonate",
            "carboxylate",
            "quat_ammonium",
            "ethoxylated",
        }
        lc = longest_chain(smiles) or 0
        ring = ring_count(m)
        br = branching_index(m)
        hydrophobe_ok = lc >= 6
        ring_ok = ring <= 2
        branch_ok = br <= 3.5
        peroxy_bad = has_peroxy(smiles)
        out = {
            "flag_valid_smiles": True,
            "flag_peroxy_banned": not peroxy_bad,
            "flag_has_head": has_head,
            "flag_hydrophobe_len": hydrophobe_ok,
            "flag_ring_ok": ring_ok,
            "flag_branch_ok": branch_ok,
        }
        out["structure_pass"] = all(out.values())
        return out

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
        if m is None:
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

    X_train_struct_scaled = pd.read_csv(TRAIN_STRUCT_SCALED)
    SCALER = joblib.load(SCALER_STRUCT)
    AUX_COLS = [c for c in X_train_struct_scaled.columns if c not in ["SMILES", "type"]]
    NUMERIC_COLS = [c for c in STRUCT_NUMERIC if c in X_train_struct_scaled.columns]
    HEAD_ONE_HOTS = [c for c in X_train_struct_scaled.columns if c.startswith("head_")]
    print("AUX_COLS loaded:", AUX_COLS)
    from latent_materials.surfactants.models import StructureOnlyGIN as GINHetero

    cand = pd.read_csv(CANDS_CSV)
    assert "SMILES" in cand.columns, "Candidates file must have a SMILES column"
    rows = []
    for smi in cand["SMILES"].astype(str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rows.append(
                {
                    "SMILES": smi,
                    **{k: np.nan for k in STRUCT_NUMERIC},
                    HEAD_COL: "other",
                }
            )
            continue
        rows.append(
            {
                "SMILES": smi,
                "longest_chain": longest_chain(smi),
                "EO_count": count_EO(smi),
                "aliphatic_fraction": aliph_frac(m),
                "ring_count": ring_count(m),
                "branching_index": branching_index(m),
                HEAD_COL: detect_head(smi),
            }
        )
    Xc = pd.DataFrame(rows)
    Xc = pd.get_dummies(Xc, columns=[HEAD_COL], prefix="head")
    for col in HEAD_ONE_HOTS:
        if col not in Xc.columns:
            Xc[col] = 0.0
    extra = [c for c in Xc.columns if c.startswith("head_") and c not in HEAD_ONE_HOTS]
    if extra:
        Xc.drop(columns=extra, inplace=True)
    Xc[NUMERIC_COLS] = Xc[NUMERIC_COLS].fillna(Xc[NUMERIC_COLS].median())
    Xc[NUMERIC_COLS] = SCALER.transform(Xc[NUMERIC_COLS])
    for col in AUX_COLS:
        if col not in Xc.columns:
            Xc[col] = 0.0
    Xc = Xc[["SMILES"] + AUX_COLS].copy()
    probe = Chem.MolFromSmiles("CCO")
    atom_dim = len(atom_feat(probe.GetAtomWithIdx(0)))
    aux_dim = len(AUX_COLS)

    def build_graph_list(df):
        data_list = []
        for smi, aux_row in zip(df["SMILES"], df[AUX_COLS].values.astype(np.float32)):
            g = smiles_to_graph(smi, aux_row)
            if g is not None:
                g.smiles = smi
                data_list.append(g)
        return data_list

    data_list = build_graph_list(Xc)
    ld = DataLoader(data_list, batch_size=128, shuffle=False)
    model = GINHetero(atom_dim, aux_dim).to(device)
    state = torch.load(GNN_CKPT, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    preds, sigmas = ([], [])
    with torch.no_grad():
        for d in ld:
            d = d.to(device)
            mu, logvar = model(d)
            sigma = torch.sqrt(torch.exp(torch.clamp(logvar, -10, 10)))
            preds.append(mu.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())
    P = np.vstack(preds)
    S = np.vstack(sigmas)
    flags = [structural_flags(s) for s in Xc["SMILES"].tolist()]
    flags_df = pd.DataFrame(flags)
    res = pd.DataFrame(
        {
            "SMILES": Xc["SMILES"].tolist(),
            "AW_ST_CMC_pred": P[:, 0],
            "pCMC_pred": P[:, 1],
            "AW_ST_CMC_sigma": S[:, 0],
            "pCMC_sigma": S[:, 1],
        }
    )
    res = pd.concat([res, flags_df], axis=1)
    struct_pass = res[res["structure_pass"] == True].copy().reset_index(drop=True)
    p_struct = f"{OUT_DIR}/decoded_candidates_structure_pass.csv"
    struct_pass.to_csv(p_struct, index=False)
    TARGETS = {"AW_ST_CMC": 32.0, "pCMC": 3.0}
    WEIGHTS = {"AW_ST_CMC": 1.0, "pCMC": 1.0}
    TEMP = 1.0
    BASE_ACCEPT = 0.95

    def guidance_score(df, targets, weights):
        eps = 1e-06
        dz_aw = (df["AW_ST_CMC_pred"] - targets["AW_ST_CMC"]) / (
            df["AW_ST_CMC_sigma"] + eps
        )
        dz_pc = (df["pCMC_pred"] - targets["pCMC"]) / (df["pCMC_sigma"] + eps)
        return weights["AW_ST_CMC"] * dz_aw**2 + weights["pCMC"] * dz_pc**2

    def acceptance_prob(score, temp=1.0, base=0.95):
        p = base * np.exp(-np.clip(score, 0, 1000000.0) / max(temp, 1e-08))
        return np.clip(p, 0.0, base)

    scored_all = res.copy()
    if TARGETS is not None:
        scored_all["guidance_score"] = guidance_score(scored_all, TARGETS, WEIGHTS)
        scored_all["accept_prob"] = acceptance_prob(
            scored_all["guidance_score"].values, TEMP, BASE_ACCEPT
        )
        scored_all.sort_values(
            "guidance_score", ascending=True, inplace=True, ignore_index=True
        )
        rng = np.random.default_rng(42)
        mask_struct = scored_all["structure_pass"].values.astype(bool)
        randu = rng.random(len(scored_all))
        soft_accept = mask_struct & (randu < scored_all["accept_prob"].values)
        accepted = scored_all[soft_accept].copy().reset_index(drop=True)
    else:
        accepted = struct_pass.copy()
        scored_all["guidance_score"] = np.nan
        scored_all["accept_prob"] = np.nan
    p_all = f"{OUT_DIR}/decoded_candidates_scored.csv"
    p_soft = f"{OUT_DIR}/decoded_candidates_soft_accept.csv"
    scored_all.to_csv(p_all, index=False)
    accepted.to_csv(p_soft, index=False)
    print("\n")
    print(f"[OK] Structure-pass saved : {p_struct}  (n={len(struct_pass)})")
    print(f"[OK] Scored (all) saved:  {p_all}      (n={len(scored_all)})")
    print(f"[OK] Soft-accepted saved:  {p_soft}    (n={len(accepted)})")
    if TARGETS is not None:
        print(
            f"Guidance targets = {TARGETS} | weights = {WEIGHTS} | temp = {TEMP} | base_accept = {BASE_ACCEPT}"
        )
        print(
            "Soft acceptance uses uncertainty-aware squared-z distance. No hard property clipping."
        )
    else:
        print("Guidance disabled (TARGETS=None). Returned structure-pass only.")
    import re
    from rdkit import Chem

    HEAD_PATTERNS = [
        "s\\(=o\\)\\(=o\\)o",
        "s\\(=o\\)\\(=o\\)",
        "\\[n\\+\\]",
        "n\\+",
        "coo",
        "po3",
        "p\\(=o\\)",
        "so3",
    ]
    HEAD_REGEX = re.compile("|".join(HEAD_PATTERNS), re.IGNORECASE)

    def has_polar_head(smiles: str) -> bool:
        if smiles is None or not isinstance(smiles, str):
            return False
        s = smiles.lower()
        return bool(HEAD_REGEX.search(s))

    def has_peroxy(smiles: str) -> bool:
        """Detect –O–O– peroxy linkage."""
        return "OO" in smiles.replace("=", "").replace("#", "")

    def longest_aliphatic_chain(smiles: str) -> int:
        """Count longest run of 'C' (coarse hydrophobe proxy, consistent with your features)."""
        try:
            return max((len(c) for c in re.findall("C+", smiles)))
        except:
            return 0

    HYDROPHOBE_MIN = 5
    BLOCK_PEROXY = True

    def structural_flags_minimal(smiles: str):
        """
        Minimal ‘not-garbage’ constraints:
          - valid SMILES
          - has at least one polar head motif
          - hydrophobe length >= HYDROPHOBE_MIN
          - (optional) no peroxy linkage
        Everything else is *observational* only (not used to filter).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "flag_valid_smiles": False,
                "flag_has_polar_head": False,
                "flag_hydrophobe_len": False,
                "flag_no_peroxy": not BLOCK_PEROXY,
                "structure_pass": False,
            }
        head_ok = has_polar_head(smiles)
        hydrophobe_ok = longest_aliphatic_chain(smiles) >= HYDROPHOBE_MIN
        peroxy_ok = not has_peroxy(smiles) if BLOCK_PEROXY else True
        flags = {
            "flag_valid_smiles": True,
            "flag_has_polar_head": head_ok,
            "flag_hydrophobe_len": hydrophobe_ok,
            "flag_no_peroxy": peroxy_ok,
        }
        flags["structure_pass"] = (
            flags["flag_valid_smiles"] and head_ok and hydrophobe_ok and peroxy_ok
        )
        return flags

    flags_min = [structural_flags_minimal(s) for s in augX["SMILES"].tolist()]
    flags_min_df = pd.DataFrame(flags_min)
    cand_pred = pd.DataFrame(
        {
            "SMILES": [g.smiles for g in data_list],
            "AW_ST_CMC_pred": P[:, 0],
            "pCMC_pred": P[:, 1],
            "AW_ST_CMC_sigma": S[:, 0],
            "pCMC_sigma": S[:, 1],
        }
    )
    cand_pred = cand.merge(cand_pred, on="SMILES", how="inner")
    cand_pred = pd.concat(
        [cand_pred.reset_index(drop=True), flags_min_df.reset_index(drop=True)], axis=1
    )
    p_all = f"{OUT_DIR}/candidates_with_predictions.csv"
    cand_pred.to_csv(p_all, index=False)
    print(f"Saved (no property filter) → {p_all}")
    struct_pass = cand_pred[cand_pred["structure_pass"]].copy().reset_index(drop=True)
    p_struct = f"{OUT_DIR}/candidates_structure_pass.csv"
    struct_pass.to_csv(p_struct, index=False)
    print(
        f"Minimal structural-pass → {p_struct}  ({len(struct_pass)} / {len(cand_pred)})"
    )
    print(
        f"Rules::  valid SMILES, has polar head, longest C-run ≥ {HYDROPHOBE_MIN}, {('no peroxy' if BLOCK_PEROXY else 'peroxy allowed')}"
    )
    import torch, os
    from latent_materials.teacher import VAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_ckpt = str(root / "models/pretrained_vae.pt")
    vae = VAE(input_dim=4920, latent_dim=32).to(device)
    if os.path.exists(vae_ckpt):
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        vae.eval()
        print(f"Loaded pretrained VAE successfully: {vae_ckpt}")
    else:
        print("pretrained_vae.pt not found in Drive — please verify.")
    import os, re, json, numpy as np, pandas as pd
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType as Hyb
    import torch, torch.nn as nn
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import joblib
    import matplotlib.pyplot as plt

    BASE_DIR = str(root)
    DATA_DIR = f"{BASE_DIR}/data"
    OUT_DIR = f"{BASE_DIR}/outputs"
    MODEL_DIR = f"{BASE_DIR}/models"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    vae_ckpt = f"{MODEL_DIR}/pretrained_vae.pt"
    cond_csv = f"{OUT_DIR}/candidates_with_predictions.csv"
    latent_out = f"{DATA_DIR}/latent_conditional_adapter.pt"
    scaler_out = f"{DATA_DIR}/cond_scaler.pkl"
    cols_out = f"{DATA_DIR}/cond_columns.json"
    df = pd.read_csv(cond_csv)
    print(f"[INFO] Loaded conditioning CSV: {df.shape}")
    expected_cols = [
        "longest_chain",
        "EO_count",
        "aliphatic_fraction",
        "ring_count",
        "branching_index",
        "head_carboxylate",
        "head_other",
        "head_quat_ammonium",
        "head_sulfonate",
        "AW_ST_CMC_pred",
        "pCMC_pred",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")
        for c in missing:
            df[c] = 0.0
    else:
        print("[OK] All conditioning columns present.")
    fixed_csv = f"{OUT_DIR}/candidates_with_predictions_fixed.csv"
    df.to_csv(fixed_csv, index=False)
    print(f"[OK] Saved fixed CSV:  {fixed_csv}")
    X_cond = df[expected_cols].copy().fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cond.values.astype(np.float32))
    joblib.dump(scaler, scaler_out)
    json.dump(expected_cols, open(cols_out, "w"))
    print(f"Saved cond scaler: {scaler_out}")
    print(f"Saved cond columns: {cols_out}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(input_dim=4920, latent_dim=32).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()
    print(f"Using pretrained VAE (latent_dim = 32) : {vae_ckpt}")

    def smiles_to_input_vector(smi: str):
        """
        Convert SMILES to the same 4920-dim input vector the VAE expects.
        Replace this with your actual featurization (e.g. ECFP, one-hot, tokenizer output).
        """
        v = np.zeros(4920, dtype=np.float32)
        for i, ch in enumerate(smi[: min(len(smi), 4920)]):
            v[i] = ord(ch) % 128 / 127.0
        return v

    smiles_list = df["SMILES"].astype(str).tolist()
    z_mu_list = []
    with torch.no_grad():
        for smi in smiles_list:
            x = torch.tensor(smiles_to_input_vector(smi)).unsqueeze(0).to(device)
            _, mu, log_var = vae(x)
            z_mu_list.append(mu.cpu())
    teacher_latents = torch.cat(z_mu_list, dim=0).to(device)
    print(f"Teacher latents extracted: {teacher_latents.shape}")
    from latent_materials.adapter_model import ConditionalAdapter as ConditionalAdapter

    cond_dim, latent_dim = (X_scaled.shape[1], teacher_latents.shape[1])
    adapter = ConditionalAdapter(cond_dim, latent_dim).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    Y_tensor = teacher_latents
    epochs, bs = (1000, 128)
    loss_trace = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X_tensor))
        total_loss = 0.0
        for i in range(0, len(X_tensor), bs):
            idx = perm[i : i + bs]
            xb, yb = (X_tensor[idx], Y_tensor[idx])
            pred = adapter(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        loss_trace.append(total_loss / max(1, len(X_tensor) / bs))
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={loss_trace[-1]:.5f}")
    torch.save(adapter.state_dict(), latent_out)
    print(f"\n Saved conditional adapter : {latent_out}")
    print(f"   Saved cond scaler : {scaler_out}")
    print(f"   Saved cond columns : {cols_out}")
    adapter.eval()
    with torch.no_grad():
        z_pred = adapter(X_tensor)
        mse = torch.mean((z_pred - Y_tensor) ** 2).item()
        corr = np.corrcoef(
            z_pred.cpu().numpy().ravel(), Y_tensor.cpu().numpy().ravel()
        )[0, 1]
    print(f"\n Adapter MSE: {mse:.4f} | Pearson corr: {corr:.4f}")
    z_t = Y_tensor.cpu().numpy()
    z_p = z_pred.cpu().numpy()
    pca = PCA(n_components=2)
    proj_t = pca.fit_transform(z_t)
    proj_p = pca.transform(z_p)
    plt.figure(figsize=(6, 6))
    plt.scatter(
        proj_t[:, 0],
        proj_t[:, 1],
        s=12,
        alpha=0.45,
        label="VAE latent μ (teacher)",
        edgecolors="none",
    )
    plt.scatter(
        proj_p[:, 0],
        proj_p[:, 1],
        s=12,
        alpha=0.45,
        label="Conditional adapter prediction",
        edgecolors="none",
    )
    plt.xlabel("PC1 (latent space)")
    plt.ylabel("PC2 (latent space)")
    plt.title("Alignment of Conditional Adapter with VAE Latent Manifold")
    plt.legend(frameon=False)
    plt.tight_layout()
    pca_fig = f"{OUT_DIR}/latent_adapter_alignment_pca.png"
    plt.savefig(pca_fig, dpi=300)
    plt.show()
    print(f"[OK] Saved PCA alignment figure: {pca_fig}")
    plt.figure(figsize=(6, 4))
    plt.plot(loss_trace, lw=2)
    plt.xlabel("Training Epoch")
    plt.ylabel("Mean Squared Error (Latent Space)")
    plt.title("Convergence of Conditional Latent Adapter")
    plt.tight_layout()
    loss_fig = f"{OUT_DIR}/latent_adapter_training_curve.png"
    plt.savefig(loss_fig, dpi=300)
    plt.show()
    print(f"[OK] Saved training curve: {loss_fig}")
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    BASE_DIR = str(root)
    OUT_DIR = f"{BASE_DIR}/outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "latent_adapter_combined_figure.png")
    out_pdf = os.path.join(OUT_DIR, "latent_adapter_combined_figure.pdf")
    required_vars = ["proj_t", "proj_p", "loss_trace"]
    missing_vars = [v for v in required_vars if v not in globals()]
    if missing_vars:
        raise ValueError(
            f"Missing variables in the current session: {missing_vars}\nPlease make sure the adapter training/PCA code has already been run."
        )
    proj_t = np.asarray(proj_t)
    proj_p = np.asarray(proj_p)
    loss_trace = np.asarray(loss_trace)
    if "pca" in globals() and hasattr(pca, "explained_variance_ratio_"):
        evr = pca.explained_variance_ratio_
        pc1_label = f"Principal component 1"
        pc2_label = f"Principal component 2"
    else:
        pc1_label = "Principal component 1"
        pc2_label = "Principal component 2"
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "legend.fontsize": 10.5,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    teacher_color = "#2C7FB8"
    adapter_color = "#1B9E77"
    loss_color = "#6A3D9A"
    grid_color = "#C7CDD4"
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    ax0 = axes[0]
    ax0.scatter(
        proj_t[:, 0],
        proj_t[:, 1],
        s=24,
        alpha=0.55,
        color=teacher_color,
        edgecolors="none",
        label="Teacher VAE latent means",
    )
    ax0.scatter(
        proj_p[:, 0],
        proj_p[:, 1],
        s=24,
        alpha=0.55,
        color=adapter_color,
        edgecolors="none",
        label="Conditional adapter predictions",
    )
    ax0.set_xlabel(pc1_label)
    ax0.set_ylabel(pc2_label)
    ax0.set_title("Latent-space alignment", pad=10)
    ax0.grid(True, linestyle="--", linewidth=0.7, alpha=0.35, color=grid_color)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["left"].set_linewidth(1.2)
    ax0.spines["bottom"].set_linewidth(1.2)
    ax0.legend(loc="best")
    ax0.text(
        -0.12, 1.04, "A", transform=ax0.transAxes, fontsize=14, va="bottom", ha="left"
    )
    ax1 = axes[1]
    epochs = np.arange(1, len(loss_trace) + 1)
    ax1.plot(epochs, loss_trace, color=loss_color, linewidth=2.4, label="Training loss")
    ax1.scatter(epochs[-1], loss_trace[-1], s=42, color=loss_color, zorder=5)
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("Mean squared error in latent space")
    ax1.set_title("Training convergence", pad=10)
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.35, color=grid_color)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.2)
    ax1.spines["bottom"].set_linewidth(1.2)
    ax1.text(
        -0.12, 1.04, "B", transform=ax1.transAxes, fontsize=14, va="bottom", ha="left"
    )
    fig.suptitle(
        "Performance of the conditional latent adapter: latent-space alignment and training behavior",
        fontsize=15.5,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved combined PNG: {out_png}")
    print(f"Saved combined PDF: {out_pdf}")
    import os, re, json, joblib, random, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import HybridizationType as Hyb

    BASE_DIR = str(root)
    DATA_DIR = f"{BASE_DIR}/data"
    OUT_DIR = f"{BASE_DIR}/outputs"
    MODEL_DIR = f"{BASE_DIR}/models"
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = joblib.load(f"{DATA_DIR}/cond_scaler.pkl")
    with open(f"{DATA_DIR}/cond_columns.json") as f:
        cond_cols = json.load(f)
    in_dim, out_dim = (len(cond_cols), 32)
    from latent_materials.adapter_model import ConditionalAdapter as AdapterMLP

    adapter = AdapterMLP(in_dim, out_dim).to(device)
    adapter.load_state_dict(
        torch.load(f"{DATA_DIR}/latent_conditional_adapter.pt", map_location=device)
    )
    adapter.eval()
    vae = VAE(input_dim=4920, latent_dim=out_dim).to(device)
    vae.load_state_dict(
        torch.load(f"{MODEL_DIR}/pretrained_vae.pt", map_location=device)
    )
    vae.eval()
    cond_df = pd.read_csv(f"{OUT_DIR}/candidates_with_predictions_fixed.csv")
    Xc = scaler.transform(cond_df[cond_cols].fillna(0).astype(np.float32))
    Xc_t = torch.tensor(Xc, dtype=torch.float32, device=device)
    with torch.no_grad():
        MU = adapter(Xc_t)
        z_list = [MU]
        for s in (0.1, 0.15, 0.2, 0.25, 0.3):
            z_list.append(MU + torch.randn_like(MU) * s)
    Z = torch.cat(z_list, dim=0)
    decoded = []
    with torch.no_grad():
        for i in range(0, Z.size(0), 256):
            xb = Z[i : i + 256]
            decoded.append(vae.decoder(xb).cpu().numpy())
    decoded_vecs = np.vstack(decoded).astype(np.float32)
    train_df = pd.read_csv(f"{DATA_DIR}/surfpro_train80.csv").dropna(subset=["SMILES"])
    test_df = pd.read_csv(f"{DATA_DIR}/surfpro_test20.csv").dropna(subset=["SMILES"])
    from latent_materials.surfactants.mutations import canon

    train_smiles = []
    for s in train_df["SMILES"]:
        c = canon(s)
        if c:
            train_smiles.append(c)
    train_smiles = list(dict.fromkeys(train_smiles))
    fps_train = []
    for s in train_smiles:
        m = Chem.MolFromSmiles(s)
        fps_train.append(
            np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        )
    fps_train = np.stack(fps_train).astype(np.float32)
    decoded_fp = decoded_vecs[:, :2048].astype(np.float32)
    from latent_materials.surfactants.mutations import _normalize

    sims = _normalize(decoded_fp) @ _normalize(fps_train).T
    idx = sims.argmax(axis=1)
    base_smi = [train_smiles[i] for i in idx]
    anchors = (
        pd.DataFrame({"anchor": base_smi})
        .drop_duplicates("anchor")
        .reset_index(drop=True)
    )
    anchors_path = f"{OUT_DIR}/anchors_unique.csv"
    anchors.to_csv(anchors_path, index=False)
    ref_smiles = set(
        train_smiles + [c for c in (canon(s) for s in test_df["SMILES"]) if c]
    )
    from latent_materials.surfactants.mutations import valid_canon
    from latent_materials.surfactants.mutations import mutate_chain
    from latent_materials.surfactants.mutations import add_EO
    from latent_materials.surfactants.mutations import remove_EO
    from latent_materials.surfactants.mutations import safe_add_EO_before_sulfate
    from latent_materials.surfactants.mutations import safe_chain_delta_before_first_O

    OO_smarts = Chem.MolFromSmarts("O-O")
    carbox_smarts = Chem.MolFromSmarts("C(=O)[O-]")
    sulfate_smarts = Chem.MolFromSmarts("OS(=O)(=O)[O-]")
    sulfonate_smarts = Chem.MolFromSmarts("S(=O)(=O)(O)")
    quatN_smarts = Chem.MolFromSmarts("[N+](C)(C)(C)C")
    ether_smarts = Chem.MolFromSmarts("OCCO")
    from latent_materials.surfactants.mutations import has_peroxy
    from latent_materials.surfactants.mutations import detect_head
    from latent_materials.surfactants.mutations import ring_count
    from latent_materials.surfactants.mutations import branching_index
    from latent_materials.surfactants.mutations import structure_pass

    TargetNovel = 250
    MaxAttempts = 20000
    generated = set()
    ops_chain = [-3, -2, -1, 1, 2, 3]
    ops_eo = [1, 2, 3, 4]
    anchors_list = anchors["anchor"].tolist()
    random.shuffle(anchors_list)
    attempts = 0
    while len(generated) < TargetNovel and attempts < MaxAttempts:
        attempts += 1
        smi0 = random.choice(anchors_list)
        k_ops = random.choice([1, 1, 2, 2, 3])
        smi = smi0
        for _ in range(k_ops):
            op = random.choice(
                ["chain", "eo_add", "eo_remove", "eo_headaware", "chain_before_O"]
            )
            if op == "chain":
                smi = mutate_chain(smi, random.choice(ops_chain)) or smi
            elif op == "eo_add":
                smi = add_EO(smi, random.choice(ops_eo)) or smi
            elif op == "eo_remove":
                smi = remove_EO(smi, random.choice([1, 2, 3])) or smi
            elif op == "eo_headaware":
                smi = safe_add_EO_before_sulfate(smi, random.choice([1, 2, 3])) or smi
            elif op == "chain_before_O":
                smi = (
                    safe_chain_delta_before_first_O(smi, random.choice([-2, -1, 1, 2]))
                    or smi
                )
        c = valid_canon(smi)
        if not c:
            continue
        if c in ref_smiles:
            continue
        if not structure_pass(c):
            continue
        generated.add(c)
    novel_raw = sorted(generated)
    novel_raw_df = pd.DataFrame({"SMILES": novel_raw})
    novel_raw_path = f"{OUT_DIR}/novel_candidates_property_guided_structural_raw.csv"
    novel_raw_df.to_csv(novel_raw_path, index=False)
    from latent_materials.surfactants.mutations import try_repairs

    valid_set, invalids = (set(), [])
    for smi in novel_raw:
        fixed = try_repairs(smi)
        if fixed:
            valid_set.add(fixed)
        else:
            invalids.append(smi)
    novel_clean = sorted(valid_set)
    novel_clean_df = (
        pd.DataFrame({"SMILES": novel_clean})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    novel_clean_path = (
        f"{OUT_DIR}/novel_candidates_property_guided_structural_clean.csv"
    )
    novel_clean_df.to_csv(novel_clean_path, index=False)
    print(f"Anchors: {len(anchors)}")
    print(f"Generation attempts: {attempts}")
    print(f"Raw novel (pre-clean): {len(novel_raw)} :: saved: {novel_raw_path}")
    print(
        f"Cleaned novel (post-sanitize/repair): {len(novel_clean)}  :: saved: {novel_clean_path}"
    )
    if invalids[:5]:
        print("Examples dropped:", invalids[:5])
    return root
