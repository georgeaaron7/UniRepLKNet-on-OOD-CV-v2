# UniRepLKNet-P on OOD-CV-v2

A completed Kaggle computer vision project combining an existing model from one
IEEE Transactions paper with an existing dataset from another. Both source
journal publications fall within 2024–2026:

- **Model:** Zhang, Ding & Yue, *Scaling Up Your Kernels: Large Kernel Design in
  ConvNets Toward Universal Representations*, IEEE TPAMI, **2025**.
  [DOI](https://doi.org/10.1109/TPAMI.2025.3600702).
- **Dataset:** Zhao et al., *OOD-CV-v2: An Extended Benchmark for Robustness to
  Out-of-Distribution Shifts of Individual Nuisances in Natural Images*, IEEE
  TPAMI, **2024**. [DOI](https://doi.org/10.1109/TPAMI.2024.3462293).

The contribution is a candidate previously unreported model–dataset application
with documented experiments. Targeted searches found no direct prior evaluation;
this is not proof of a world-first combination. See
[source verification and novelty audit](docs/ieee_source_verification.md).

## Completed experiment

[Open saved Kaggle Version 1](https://www.kaggle.com/code/aarooongeorge/unireplknet-on-ood-cv-v2-tpami-transfer-study?scriptVersionId=367303127).
The notebook is private in the signed-in account. All model computation ran on
Kaggle; the T4 session was stopped after saving its outputs.

| Frozen encoder + linear classifier | Test accuracy | Test macro-F1 |
|---|---:|---:|
| ResNet-18 baseline | 69.84% | 69.74% |
| UniRepLKNet-P | 77.63% | 77.69% |

The macro-F1 gain is **7.95 percentage points** under the declared protocol.
Train: 8,588 images; validation: 4,059; held-out test: 17,135. Classifier
regularization was selected using labeled phase-1 validation, so this is not
source-only domain generalization. Scores are not directly comparable with
end-to-end fine-tuning results in the dataset paper.

The saved Output tab contains **40 files**, including `unireplknet_oodcv.zip`
(114.65 MB), a measured report, confusion matrices, nuisance-group accuracy,
label-efficiency and error plots, validation searches, predictions, integrity
records, source/environment pins, and the fitted linear classifier. Download
that ZIP from Kaggle's Output tab. Executed outputs remain on Kaggle; the local
notebook is reproducible source without copied execution outputs.

## Reproduce on Kaggle

1. Import `notebooks/unireplknet_oodcv.ipynb` into Kaggle.
2. Enable Internet, select an available GPU, and wait for the setting to apply.
3. Run All. Official data, code, and checkpoint are downloaded automatically.
4. Save a version with output files; download `unireplknet_oodcv.zip` from Output.

Both encoders remain frozen. Identical preprocessing and a validation-selected
logistic regression are used, with 10%, 25%, and 100% training-label conditions.
The notebook audits exact image duplicates and logs unused or repeated labels.
Raw images and pretrained weights are not redistributed in the output bundle.

## Project files

- [Notebook](notebooks/unireplknet_oodcv.ipynb) and [readable source](notebooks/unireplknet_oodcv.py).
- [Project scope and protocol](docs/tpami_project_proposal.md).
- [Source verification](docs/ieee_source_verification.md) and [BibTeX](docs/ieee_sources.bib).
- [Measured results summary](docs/results_summary.md) and [execution status](docs/run_status.md).

Rebuild with `python3 tools/build_notebook.py unireplknet_oodcv.py`.
Code cells were syntax-checked locally; the actual experiment ran on Kaggle.

## Superseded preliminary work

The earlier SigLIP 2 + RealWaste experiment does **not** meet the source-paper
requirement and is retained only as preliminary work. Its files and scores must
not be attributed to this TPAMI project.
