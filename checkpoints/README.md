# Recovered model checkpoints

This directory makes the **original author-supplied trained models** available with the code. The files were recovered without retraining or modification from `surfactants(1).zip`; all hashes are recorded in `manifest.json`. Duplicate copies of the same pretrained VAE have been **deduplicated in Git**, with two historical path aliases recreated on the local machine only when needed.

## One-time installation after cloning

```bash
python scripts/restore_data.py
python checkpoints/restore.py
```

The `checkpoints/packed/` files are lossless chunks of a compressed archive containing **14 unique model or fitted-transformer artifacts**. The restoration script verifies every chunk, reconstructs the exact original files, creates the two necessary local compatibility aliases and verifies original SHA-256 checksums. No separate weights download is required. To verify without reconstruction: `python checkpoints/restore.py --verify-packed`. To verify the restored files: `python checkpoints/restore.py --verify-restored`.

For generative workflows that accept a model root, use `--model-root checkpoints`, e.g.:

```bash
latent-materials run surfactant_generation --data-root data --model-root checkpoints --output results/surfactant_generation
latent-materials run aem_surrogate_validation_selected --data-root data --model-root checkpoints --output results/aem_surrogate_validation
```

## Scope and reproducibility

The archive contains: the common pretrained VAE; two-domain and property-conditional surfactant/AEM adapters; combined AEM VAE and graph autoencoder/conductivity model ensemble; surfactant prediction models and the pC20 GNN validation model; and fitted scaling/target-transform objects. The same teacher is used in multiple workflows, **not three separately pretrained teachers**.

Recovered byte identity does not by itself establish that every paper result can be reproduced from these specific historical checkpoints. Matching source-code version, model architecture, featurization, train/test splits and pipeline inputs still requires experiment-level validation. Saved numerical analyses have a separate reproduction scope described in the top-level README. Python pickle and PyTorch model archives may execute code on loading: load only trusted checkpoints in the correct environment.
