# Manuscript code release — publication status

The public repository currently contains release metadata only. **It does not yet contain the entire scientific source, datasets, figures, tests, or provenance records.** Do not cite this repository as a complete deposited research-code release until the package is pushed and independently verified on GitHub.

## Prepared offline release

The checkpoint-free archive contains 321 files, including original manuscript figures and selected data, source code, tests, and audit reports. It intentionally excludes five historical checkpoint/scaler files. Its local integrity manifest and ZIP CRC check passed on 2026-10-10. In that environment 129 runnable unit tests and 97 reported artifact checks passed. Seven additional tests were not runnable without the optional torch-geometric dependency. These are local results, **not GitHub CI**.

One 88.2 MB CSV in the archive is configured for Git LFS, and Git LFS upload is necessary to make its pointer usable.

## Before citing this repository

- Push every approved release file, including all Git LFS objects. Do not publish checkpoints or scalers.
- Replace the placeholder entry in `CITATION.cff` with the final confirmed author list.
- Resolve applicable code and third-party dataset redistribution terms before public distribution.
- Run GitHub Actions, check all files and LFS assets, and verify the published commit.
- Disclose missing original historical final model states, complete Figure 2 training provenance, and MD inputs without claiming exact end-to-end reproduction.

The release README and `docs/PRE_SUBMISSION_ITEMS.md` in the prepared archive contain the detailed workflows and limitations.
