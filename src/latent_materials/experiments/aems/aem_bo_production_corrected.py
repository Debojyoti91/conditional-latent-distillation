"""New experiment: corrected executable mutations/anchor linkage.

Call run(workspace) with an isolated writable workspace; use the CLI for input preparation.
"""

from pathlib import Path as WorkspacePath
from latent_materials.workspace import require_dependencies


def run(workspace):
    """Execute the aem bo production corrected experiment in a prepared workspace."""
    root = WorkspacePath(workspace).resolve()
    _external_setup = require_dependencies
    project_dir = str(root / "IEMS_GEN_files/AEM/CVAE") if "aem" == "aem" else str(root)
    data_dir = DATA_DIR = str(WorkspacePath(project_dir) / "data")
    OUT_DIR = str(WorkspacePath(project_dir) / "outputs")
    import torch, os

    VAE_CKPT = str(
        root / "Generative_AI_surfactants_molecules/models/pretrained_vae.pt"
    )
    from latent_materials.teacher import VAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=4920, latent_dim=32).to(device)
    if os.path.exists(VAE_CKPT):
        vae.load_state_dict(torch.load(VAE_CKPT, map_location=device))
        vae.eval()
        print(f"Loaded pretrained VAE successfully: {VAE_CKPT}")
    else:
        print("pretrained_vae.pt not found in Drive — please verify.")
    import os, re, random, numpy as np, pandas as pd, torch
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, DataStructs

    BASE_DIR = str(root / "IEMS_GEN_files/AEM/CVAE")
    DATA_DIR = f"{BASE_DIR}/data"
    OUT_DIR = f"{BASE_DIR}/outputs"
    MODEL_DIR = f"{BASE_DIR}/models"
    os.makedirs(OUT_DIR, exist_ok=True)
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device_bo = torch.device("cpu")
    device_vae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.double)

    VAE_CKPT = str(
        root / "Generative_AI_surfactants_molecules/models/pretrained_vae.pt"
    )
    from latent_materials.teacher import VAE

    LATENT_DIM = 32
    vae = VAE(input_dim=4920, latent_dim=LATENT_DIM).to(device_vae)
    state = torch.load(VAE_CKPT, map_location=device_vae)
    vae.load_state_dict(state, strict=False)
    vae = vae.float().eval()
    print("[INIT] VAE loaded on", device_vae)

    def _canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None

    train_df = pd.read_csv(f"{DATA_DIR}/aem_train80.csv").dropna(subset=["SMILES"])
    test_df = pd.read_csv(f"{DATA_DIR}/aem_test20.csv").dropna(subset=["SMILES"])
    train_smiles = sorted({_canon(s) for s in train_df["SMILES"] if _canon(s)})
    test_smiles = sorted({_canon(s) for s in test_df["SMILES"] if _canon(s)})
    prop_df = pd.concat([train_df, test_df], ignore_index=True)
    prop_df = prop_df.dropna(subset=["SMILES"])
    COND_CANDIDATES = [
        "Conductivity",
        "Anion_Conductivity_Scm",
        "Anion_Conductivity_pred_Scm",
        "Membrane_Anion_Conductivity_Scm",
        "Membrane Anion Conductivity [S/cm]",
    ]
    cond_col = None
    for c in COND_CANDIDATES:
        if c in prop_df.columns:
            cond_col = c
            break
    if cond_col is None:
        raise ValueError(
            f"Could not find conductivity column. Tried: {COND_CANDIDATES}. Columns in file: {list(prop_df.columns)}"
        )
    print(f"[INIT] Using conductivity column: {cond_col}")
    prop_df["SMILES"] = prop_df["SMILES"].map(_canon)
    prop_df = prop_df.dropna(subset=["SMILES"]).drop_duplicates("SMILES")
    prop_lookup = {r.SMILES: float(r[cond_col]) for _, r in prop_df.iterrows()}
    print(
        f"[INIT] Train SMILES: {len(train_smiles)} | Test SMILES: {len(test_smiles)} | Props: {len(prop_lookup)}"
    )
    FP_BITS, FP_RADIUS = (2048, 2)
    _mgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

    def smiles_to_fp_np(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return np.zeros((FP_BITS,), dtype=np.float32)
        bv = _mgen.GetFingerprint(m)
        arr = np.zeros((FP_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    print("[INIT] Building AEM train fingerprints …")
    fps_train_np = np.stack([smiles_to_fp_np(s) for s in train_smiles]).astype(
        np.float32
    )
    fps_train_norm_np = fps_train_np / (
        np.linalg.norm(fps_train_np, axis=1, keepdims=True) + 1e-08
    )
    fps_train_norm_t = torch.from_numpy(fps_train_norm_np).to(
        device_vae, dtype=torch.float32
    )
    fps_train_norm_t = torch.nn.functional.normalize(fps_train_norm_t, dim=1)

    @torch.no_grad()
    def latent_to_fp_like(z_double_on_cpu: torch.Tensor, batch=512) -> torch.Tensor:
        """
        Input: z (double, CPU), shape (M, LATENT_DIM)
        Output: normalized fp-like vectors (float32, GPU), shape (M, 2048)
        """
        outs = []
        for i in range(0, z_double_on_cpu.size(0), batch):
            z_chunk = z_double_on_cpu[i : i + batch].to(device_vae, dtype=torch.float32)
            vec = vae.decoder(z_chunk)[:, :FP_BITS]
            vec = torch.nn.functional.normalize(vec, dim=1)
            outs.append(vec)
        return torch.cat(outs, dim=0)

    def anchor_obj_from_idx(i: int) -> np.ndarray:
        """
        Multi-objective proxy for BO:
          - Here we maximize conductivity (twice, so 2D for EHVI).
        """
        smi = train_smiles[i]
        cond = prop_lookup.get(smi, 0.0)
        return np.array(
            [cond if cond >= 0.05 else 0, cond if cond >= 0.05 else 0], dtype=np.float64
        )

    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms import Standardize
    from botorch.utils.transforms import normalize, unnormalize
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import SumMarginalLogLikelihood

    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogExpectedHypervolumeImprovement as ACQ_CLS,
        )

        ACQ_KW = {}
        print("[INIT] Using qLogExpectedHypervolumeImprovement")
    except Exception:
        from botorch.acquisition.multi_objective.monte_carlo import (
            qExpectedHypervolumeImprovement as ACQ_CLS,
        )

        ACQ_KW = {"use_log": True}
        print("[INIT] Using qExpectedHypervolumeImprovement(use_log=True)")
    INIT_N, BATCH_Q, N_ITER = (256, 8, 5)
    RAW_SAMPLES, NUM_RESTARTS, QMC_SAMPLES = (64, 8, 32)
    LATENT_BOUNDS = torch.tensor(
        [[-4.0] * LATENT_DIM, [4.0] * LATENT_DIM], device=device_bo, dtype=torch.double
    )
    torch.manual_seed(SEED)
    init_z = LATENT_BOUNDS[0] + (LATENT_BOUNDS[1] - LATENT_BOUNDS[0]) * torch.rand(
        INIT_N, LATENT_DIM, device=device_bo, dtype=torch.double
    )
    print("[BO] Decoding initial latents → fp_like (GPU) …")
    with torch.no_grad():
        fp_like_0_t = latent_to_fp_like(init_z)
    print("[BO] Similarity (GPU) …")
    sims0_t = fp_like_0_t @ fps_train_norm_t.T
    idx0 = torch.argmax(sims0_t, dim=1).detach().cpu().numpy()
    Y0 = np.stack([anchor_obj_from_idx(i) for i in idx0])
    train_x = normalize(init_z, bounds=LATENT_BOUNDS)
    train_y = torch.tensor(Y0, device=device_bo, dtype=torch.double)

    def fit_model(X, Y):
        models = []
        for j in range(Y.shape[1]):
            gp = SingleTaskGP(X, Y[:, j : j + 1], outcome_transform=Standardize(m=1))
            models.append(gp)
        mdl = ModelListGP(*models).to(device_bo)
        mll = SumMarginalLogLikelihood(mdl.likelihood, mdl)
        fit_gpytorch_mll(mll)
        return mdl

    print("[BO] Fitting initial GP …")
    model = fit_model(train_x, train_y)
    y_np = train_y.detach().cpu().numpy()
    ref_point = torch.tensor(
        [y_np[:, 0].min() - 0.01, y_np[:, 1].min() - 0.01],
        device=device_bo,
        dtype=torch.double,
    )
    sampler = SobolQMCNormalSampler(torch.Size([QMC_SAMPLES]))
    for it in range(N_ITER):
        print(f"[BO] === Iteration {it + 1}/{N_ITER} ===")
        with torch.no_grad():
            partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
        acq = ACQ_CLS(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
            **ACQ_KW,
        )
        bounds01 = torch.stack(
            [
                torch.zeros(LATENT_DIM, device=device_bo, dtype=torch.double),
                torch.ones(LATENT_DIM, device=device_bo, dtype=torch.double),
            ]
        )
        cand_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds01,
            q=BATCH_Q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            sequential=True,
        )
        cand_z = unnormalize(cand_norm, bounds=LATENT_BOUNDS)
        with torch.no_grad():
            fp_like_c_t = latent_to_fp_like(cand_z)
            sims_c_t = fp_like_c_t @ fps_train_norm_t.T
            idx_c = torch.argmax(sims_c_t, dim=1).detach().cpu().numpy()
        Yc = torch.tensor(
            np.stack([anchor_obj_from_idx(i) for i in idx_c]),
            device=device_bo,
            dtype=torch.double,
        )
        train_x = torch.cat([train_x, cand_norm], dim=0)
        train_y = torch.cat([train_y, Yc], dim=0)
        print(f"[BO]   → dataset size now: {train_x.size(0)}")
        model = fit_model(train_x, train_y)
    print(f"[BO] Total latent points evaluated: {train_x.size(0)}")
    TARGET_ANCHORS = 400
    TOPK_PER_LATENT = 25
    CHUNK_DEC = 1024
    CHUNK_SIM = 2048
    print("[ANCHORS] Decoding ALL evaluated latents → fp_like (GPU) …")
    with torch.no_grad():
        pool_z = unnormalize(train_x, LATENT_BOUNDS)
        fp_like_all = []
        for i in range(0, pool_z.size(0), CHUNK_DEC):
            fp_like_all.append(latent_to_fp_like(pool_z[i : i + CHUNK_DEC]))
        fp_like_all_t = torch.cat(fp_like_all, dim=0)
    print("[ANCHORS] GPU top-k nearest per latent …")
    cand_idx_set = set()
    n_train = fps_train_norm_t.size(0)
    for i in range(0, fp_like_all_t.size(0), CHUNK_SIM):
        blk = fp_like_all_t[i : i + CHUNK_SIM]
        sims_blk = blk @ fps_train_norm_t.T
        topk_vals, topk_idx = torch.topk(
            sims_blk, k=min(TOPK_PER_LATENT, n_train), dim=1
        )
        cand_idx_set.update(topk_idx.detach().cpu().numpy().reshape(-1).tolist())
    cand_idx = np.array(sorted(cand_idx_set), dtype=np.int64)
    print(f"[ANCHORS] Candidate train indices gathered: {len(cand_idx)}")
    print("[ANCHORS] Greedy max–min diversity selection …")
    cand_fps = (
        torch.nn.functional.normalize(
            torch.from_numpy(fps_train_norm_np[cand_idx]).to(torch.float32), dim=1
        )
        .cpu()
        .numpy()
    )
    sel = []
    if cand_fps.shape[0] > 0:
        mean_vec = cand_fps.mean(axis=0, keepdims=True)
        d0 = np.linalg.norm(cand_fps - mean_vec, axis=1)
        sel.append(int(np.argmax(d0)))
        while len(sel) < min(TARGET_ANCHORS, cand_fps.shape[0]):
            S = cand_fps[sel]
            d_to_S = np.linalg.norm(cand_fps[:, None, :] - S[None, :, :], axis=2).min(
                axis=1
            )
            d_to_S[sel] = -1.0
            nxt = int(np.argmax(d_to_S))
            if d_to_S[nxt] <= 0:
                break
            sel.append(nxt)
    picked = cand_idx[sel] if len(sel) else cand_idx
    anchors_df = (
        pd.DataFrame({"anchor": [train_smiles[i] for i in picked]})
        .drop_duplicates("anchor")
        .reset_index(drop=True)
    )
    anchors_path = f"{OUT_DIR}/aem_anchors_unique_bo_qlogehvi_fast_2.csv"
    anchors_df.to_csv(anchors_path, index=False)
    print(
        f"[DONE] AEM BO anchors saved → {anchors_path} | count={len(anchors_df)} | candidates_scanned={len(cand_idx)} | latents={pool_z.size(0)}"
    )
    import os, re, random, numpy as np, pandas as pd
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    BASE_DIR = str(root / "IEMS_GEN_files/AEM/CVAE")
    DATA_DIR = f"{BASE_DIR}/data"
    OUT_DIR = f"{BASE_DIR}/outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    anchors_path = f"{OUT_DIR}/aem_anchors_unique_bo_qlogehvi_fast_2.csv"
    anchors_df = (
        pd.read_csv(anchors_path).dropna(subset=["anchor"]).drop_duplicates("anchor")
    )
    anchors_list = anchors_df["anchor"].tolist()
    random.shuffle(anchors_list)
    print(f"[INIT] Loaded AEM BO anchors: {len(anchors_list)}")

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None

    train_df = pd.read_csv(f"{DATA_DIR}/aem_train80.csv").dropna(subset=["SMILES"])
    test_df = pd.read_csv(f"{DATA_DIR}/aem_test20.csv").dropna(subset=["SMILES"])
    train_smiles = sorted({canon(s) for s in train_df["SMILES"] if canon(s)})
    test_smiles = sorted({canon(s) for s in test_df["SMILES"] if canon(s)})
    ref_smiles = set(train_smiles + test_smiles)
    OO_smarts = Chem.MolFromSmarts("O-O")
    quatN_smarts = Chem.MolFromSmarts("[N+](C)(C)(C)C")
    imidaz_smarts = Chem.MolFromSmarts("c1[n+](C)cnc1")
    phos_smarts = Chem.MolFromSmarts("[P+](C)(C)(C)C")
    assert quatN_smarts is not None, "Error in quatN SMARTS"
    assert imidaz_smarts is not None, "Error in imidaz SMARTS"
    assert phos_smarts is not None, "Error in phosphonium SMARTS"

    def valid_canon(smi):
        m = Chem.MolFromSmiles(smi)
        if not m:
            return None
        try:
            Chem.SanitizeMol(m)
        except:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    def has_peroxy(smi):
        m = Chem.MolFromSmiles(smi)
        return m is not None and m.HasSubstructMatch(OO_smarts)

    def has_cationic_head(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        return (
            m.HasSubstructMatch(quatN_smarts)
            or m.HasSubstructMatch(imidaz_smarts)
            or m.HasSubstructMatch(phos_smarts)
        )

    def structure_pass(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        if has_peroxy(smi):
            return False
        if not has_cationic_head(smi):
            return False
        return True

    from latent_materials.aems.mutations import (
        mutate_aem,
    )

    _mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps_rd = [
        _mgen.GetFingerprint(Chem.MolFromSmiles(s))
        for s in train_smiles
        if Chem.MolFromSmiles(s)
    ]

    def novel_enough(smi, thr=0.9):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        fp = _mgen.GetFingerprint(m)
        sims = [DataStructs.TanimotoSimilarity(fp, t) for t in train_fps_rd]
        return (max(sims) if sims else 0.0) < thr

    MaxAttempts = 20000
    NOVEL_THR = 0.9
    attempts, generated = (0, set())
    print(f"[GEN] Rolling AEM mutations until {MaxAttempts} attempts...")
    while attempts < MaxAttempts:
        attempts += 1
        if attempts % 2000 == 0:
            print(f"[GEN] attempts={attempts} | kept={len(generated)}")
        smi0 = random.choice(anchors_list)
        smi = smi0
        for _ in range(random.choice([1, 2, 3])):
            smi = mutate_aem(smi) or smi
        c = valid_canon(smi)
        if not c:
            continue
        if c in ref_smiles:
            continue
        if not structure_pass(c):
            continue
        if not novel_enough(c, NOVEL_THR):
            continue
        generated.add(c)
    print(f"[GEN] Completed {attempts} attempts | raw novel={len(generated)}")
    novel_raw = sorted(generated)
    raw_path = f"{OUT_DIR}/aem_novel_candidates_raw_2.csv"
    pd.DataFrame({"SMILES": novel_raw}).to_csv(raw_path, index=False)
    print(f"[DONE] Raw AEM candidates saved → {raw_path}")
    import pandas as pd
    from rdkit import Chem

    raw_path = f"{OUT_DIR}/aem_novel_candidates_raw_2.csv"
    df_raw = pd.read_csv(raw_path)
    raw_smis = df_raw["SMILES"].dropna().unique().tolist()
    print(f"[CLEAN] Loaded {len(raw_smis)} raw AEM candidates")

    def to_canon(smi):
        m = Chem.MolFromSmiles(smi)
        if not m:
            return None
        try:
            Chem.SanitizeMol(m)
        except Exception:
            try:
                Chem.SanitizeMol(
                    m,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                    | Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
                )
            except Exception:
                return None
        return Chem.MolToSmiles(m, canonical=True)

    def try_repairs(smi):
        c = to_canon(smi)
        if c and structure_pass(c):
            return c
        if "OOCC" in smi:
            s2 = smi.replace("OOCC", "OCC")
            c = to_canon(s2)
            if c and structure_pass(c):
                return c
        for fix in ["N+(C)(C)C", "c1[n+](C)cnc1", "P+(C)(C)C"]:
            if fix in smi:
                c = to_canon(smi)
                if c and structure_pass(c):
                    return c
        for delta in [1, -1, 2, -2]:
            m = Chem.MolFromSmiles(smi)
            if not m:
                continue
            runs = list(re.finditer("C{2,}", smi))
            if not runs:
                continue
            r = max(runs, key=lambda x: len(x.group(0)))
            n = len(r.group(0))
            nn = max(2, n + delta)
            fixed = smi[: r.start()] + "C" * nn + smi[r.end() :]
            c = to_canon(fixed)
            if c and structure_pass(c):
                return c
        return None

    cleaned = []
    invalid = []
    for smi in raw_smis:
        fixed = try_repairs(smi)
        if fixed:
            cleaned.append(fixed)
        else:
            invalid.append(smi)
    cleaned = sorted(set(cleaned))
    clean_path = f"{OUT_DIR}/aem_novel_candidates_clean_2.csv"
    pd.DataFrame({"SMILES": cleaned}).to_csv(clean_path, index=False)
    print(f"[CLEAN] Raw count      : {len(raw_smis)}")
    print(f"[CLEAN] Clean count     : {len(cleaned)}")
    print(f"[CLEAN] Dropped invalid : {len(invalid)}")
    print(f"[CLEAN] Clean file saved → {clean_path}")
    if len(invalid) > 0:
        log_path = f"{OUT_DIR}/aem_failed_cleaning_log_2.csv"
        pd.DataFrame({"SMILES": invalid}).to_csv(log_path, index=False)
        print(f"[CLEAN] Invalid molecules logged → {log_path}")
    return root
