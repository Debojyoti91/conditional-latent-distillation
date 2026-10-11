"""Reported AEM graph and training defaults; returns a fresh configuration."""

from rdkit.Chem import rdchem


def default_config():
    return dict(
        atom_nums=[1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53],
        max_degree=6,
        charges=[-2, -1, 0, 1, 2],
        hybs=[
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,
        ],
        bond_types=[
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
        ],
        enc_hidden=128,
        enc_layers=3,
        enc_dropout=0.1,
        use_mean_max_pool=False,
        ae_lr=0.001,
        ae_weight_decay=1e-05,
        ae_epochs=150,
        ae_batch_size=64,
        growth=256,
        n_blocks=4,
        dropout=0.28,
        head_hidden=384,
        lr=0.00105,
        weight_decay=0.0009,
        batch_size=32,
        epochs=600,
        T0=70,
        Tmult=1,
        swa_ratio=0.33,
        alpha=0.65,
        sup_weight=0.9,
        recon_weight=0.1,
        use_sample_weight=False,
        high_cond_threshold=0.09,
        high_cond_weight=3.0,
        ensemble_size=10,
        use_graph_augmentation=False,
        edge_drop_p=0.1,
        aux_noise_std=0.02,
    )
