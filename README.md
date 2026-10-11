# Conditional latent distillation for surfactants and AEMs

Source code, data and notebooks for the surfactant and anion-exchange membrane (AEM) workflows associated with the manuscript and supplementary information. The package separates reusable scientific components from named experiments. Notebooks explain and call those components; they do not contain duplicate training implementations.

## Repository organization

| Directory | Contents |
|---|---|
| `src/latent_materials/surfactants/` | Graph datasets, training features, comparison descriptors, model architectures, training functions, mutations and ablation configuration |
| `src/latent_materials/aems/` | Atom/bond features, graph assembly, conductivity models, training functions, descriptors, mutations and configuration |
| `src/latent_materials/experiments/surfactants/` | Named surfactant training, ablation and figure functions |
| `src/latent_materials/experiments/aems/` | Named AEM training, generation, screening and analysis functions |
| `src/latent_materials/` | Shared adapter/VAE architectures, statistics, identity safeguards, retrieval, inverse transforms, coherent model export and command-line interface |
| `data/surfactants/`, `data/aems/`, `data/shared/` | Supplied inputs and reference numerical results, organized by domain |
| `notebooks/surfactants/`, `notebooks/aems/` | Four clean workflow notebooks for each domain |
| `tests/` | Unit tests and small numerical/training integration checks |
| `.github/workflows/tests.yml` | Automated tests on pushes and pull requests |

Model weights, serialized fitted models, test logs, notebook outputs and generated figures are excluded. Training can create private model artifacts in a user-selected output directory.

## Installation

Use Python 3.12 and an isolated environment. From the repository root:

```bash
python scripts/restore_data.py
python -m venv .venv
# Activate .venv using the command appropriate for your operating system.
python -m pip install -e ".[dev]"
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[research,notebooks]"
python -m pytest
```

The core installation supports saved-data analyses. The `research` extra adds graph-learning, Bayesian-optimization, SELFIES and clustering dependencies; `notebooks` adds JupyterLab. RDKit is pinned to **2025.9.1** because descriptor changes in later releases alter some acceptor counts. `requirements-reproduction.txt` records versions used for the checks described below; it does not claim to reconstruct the original authors' training environment. Use a suitable PyTorch accelerator build for full GPU training.

## Run an experiment

```bash
latent-materials list
latent-materials run figure_2 --data-root data --output results/figure_2
latent-materials run aem_statistics --data-root data --output results/aem_statistics
```

Equivalent Python API:

```python
from latent_materials.cli import run_experiment
run_experiment("figure_2", data_root="data", output="results/figure_2")
```

Choose a **new output directory** for every run. Inputs are copied into an isolated writable workspace, and the supplied data are preserved. The workspace retains historical relative input names so existing numerical artifacts remain traceable. Machine-specific drive paths and notebook-cell execution are not required. Run the notebooks from this repository after installing the package; expensive analyses and training have explicit opt-in switches.

## Manuscript and SI workflows

| Experiment | Purpose and principal outputs |
|---|---|
| `surfactant_conditioning_ablation` | Five conditioning variants, five optimization seeds and 20 partitionwise property shuffles; Tables S7/S8 summaries, paired bootstrap comparisons and prediction arrays |
| `figure_s11` | Two-panel ablation figure from the saved ablation results |
| `figure_2` | Two-domain latent validation figure from the supplied teacher/student cache |
| `figure_3` | Surfactant NN/BO descriptor, accessibility, novelty and chemical-space comparison |
| `surfactant_generation` | Constrained EO/PO library, structure-only guidance, conditional-adapter fitting and mutation expansion; requires training split inputs and teacher weights |
| `surfactant_surrogates` | Historical data preparation, pC20 imputation, two-target graph-surrogate fitting, configuration search and evaluation |
| `figure_4_aem` | Matched AEM NN/BO comparison, descriptor summaries, novelty and PCA coverage |
| `aem_statistics` | Canonical-SMILES conductivity alignment, effect sizes and comparison statistics |
| `aem_weight_sensitivity` | Equation 14 weight perturbations, rank stability and sensitivity summaries |
| `aem_production_postprocess` | Production candidate annotation, filtering, ranking, diversity selection and shortlists |
| `aem_synthesizability` | Heuristic screening, representative structures and candidate shortlists |
| `extended_data_aem` | Plausibility-filtered gallery of representative NN/BO structures |
| `aem_bo_production_corrected` | Production BO with valid charged mutations and consistent use of the fresh candidate file |
| `aem_surrogate_validation_selected` | Graph-autoencoder/conductivity training, combined-VAE workflow, candidate prediction and validation-only final ensemble selection |

Default ablation settings are available through `surfactants.config.AblationConfig`; AEM defaults through `aems.config.default_config()`. A reduced ablation can be supplied through `run_experiment(..., experiment_options={"config": config})`. Reduced epochs, seeds or shuffles are software checks, not manuscript results.

## Data and method conventions

The distribution contains **158 data files**: CSV/XLSX tables, numerical NPY/NPZ arrays and JSON metadata. Candidate SMILES, properties, supplied split tables, latent caches, ablation results and comparison outputs retain their original filenames. The surfactant and AEM polymer corpora differ and are both retained. `src/latent_materials/resources/data_layout.json` maps each distributed file to its experiment input location; `required_inputs.json` specifies prerequisites per experiment.

Surfactant models predict surface tension at CMC and pCMC, with a separate pC20 imputation workflow. Historical EO/PO and chain descriptors include SMILES substring rules. Surrogate-training features and comparison features have different recorded definitions and are implemented separately. AEM graphs combine atom/bond features with scaled numerical inputs; target inversion uses the fitted transformer. Accessibility scores are in-house heuristics, and IEC is a molecular proxy.

The matched AEM BO benchmark contains 491 candidates; the production population contains 482. They must not be interchanged. Decoder-to-anchor retrieval compares the first 2,048 numerical decoder components with reference fingerprint vectors using normalized dot products; it does not establish fingerprint semantics for decoder outputs. The two-domain frozen-teacher cache and the combined AEM VAE training workflow are distinct experiments. The recovered surfactant generator uses its recorded ASCII-based 4,920-component input encoding; equivalence to the teacher pretraining representation has not been established.

The four large source inputs are stored as lossless gzip parts under `data/packed/` to make authenticated GitHub transfers reliable. Run `python scripts/restore_data.py` once after cloning. It reconstructs the original filenames and verifies every part and restored file against SHA256 checksums; the scientific source code and numerical data match the tested package. The restored files are ignored by Git, while the packed parts remain tracked. No external dataset download or model weights are required for this restoration.

## External checkpoints and new training

Saved-data figures and analyses use the distributed numerical artifacts. Generative workflows additionally require compatible author-owned teacher checkpoints. Inspect the relevant entry in `required_inputs.json`, place those files beneath an external directory using the listed relative paths, and pass `--model-root /path/to/private_models` or `model_root=...` in Python. Missing prerequisites are reported before an output workspace is created.

New AEM ensemble selection uses validation RMSE. The selected ensemble, fitted preprocessing, prediction arrays and recomputed metrics are exported together. Charged mutation operators are shared by the generation routes and tested for sanitization and gate acceptance. Newly trained results remain new realizations; validation-based corrections do not retroactively validate historical test-selected results.

## Verification and scope

**39 tests pass**, covering statistical formulas, partitionwise shuffling, canonical molecular identities, fitted target inversion, anchor retrieval, chemical mutations, graph batching, loss/gradients, both surrogate trainers and model-export consistency. A reduced end-to-end ablation also completes. All nine saved-data analysis workflows run. Fourteen rewritten comparison/statistics/sensitivity/screening CSV tables agree with supplied references within absolute tolerance `1e-9`. Eight production annotation tables differ only in cached HBA values; their other fields, including rankings and selected structures, agree. These HBA annotations require reconciliation before claiming complete numerical identity.

Full expensive surrogate/BO training was not rerun during this refactor. The source materials also leave unresolved cohort-count/canonical-overlap differences in the historical surfactant split, historical held-out checkpoint selection, and the upstream producer of the Figure 2 cache. MD trajectory/topology inputs and the corresponding Figure 5/S9–S10 analysis producer were not identified. This repository makes the available workflows executable and testable; those remaining manuscript-level issues are not resolved by unit tests.
