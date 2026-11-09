# Execution status

## Current publication requirement

Both source papers must be IEEE Transactions publications from 2024–2026.
The verified replacement is UniRepLKNet (TPAMI 2025) + OOD-CV-v2 (TPAMI 2024).
See `ieee_source_verification.md`. Official OOD-CV-Cls.zip access was verified in
Kaggle (692,430,073 bytes; 30,205 archive entries), including released labels and
nuisance-group annotations for phase-1 and phase-2.

The replacement notebook is `notebooks/unireplknet_oodcv.ipynb`, with 12 cells.
All Python cells pass local syntax validation. Its full Kaggle run completed
on a verified T4 GPU. It uses the official train / phase-1 validation / phase-2
test protocol, frozen encoders, a ResNet-18 baseline, training-label subsets,
and an automatically generated measured report. All 15 evaluation conditions
(14 encoder/subset conditions and one majority baseline) passed the final checks.

Saved Kaggle Quick Version 1 (successful, with outputs):
https://www.kaggle.com/code/aarooongeorge/unireplknet-on-ood-cv-v2-tpami-transfer-study?scriptVersionId=367303127

Version name: `TPAMI pair - completed T4 experiment`. Quick Save persisted the
already executed cells and output files; its displayed 10-second save runtime is
not the experiment runtime. The report records 306.6 seconds after setup. The
saved Output tab was verified: 40 files and `unireplknet_oodcv.zip` (114.65 MB).
Outputs remain on Kaggle; no local result archive download is claimed. The GPU
session was stopped afterward.

Full-label test accuracy / macro-F1: ResNet-18 69.8395% / 69.7372%; UniRepLKNet-P
77.6306% / 77.6905%. Macro-F1 gain: +7.9533 percentage points. See
`results_summary.md` for protocol and limitations. These are measured outcomes.

Replacement Kaggle editor:
https://www.kaggle.com/code/aarooongeorge/unireplknet-on-ood-cv-v2-tpami-transfer-study/edit

The replacement data audit passed: 30,090 uniquely annotated image files;
29,782 after exact-image deduplication. Train: 8,588; validation: 4,059;
test: 17,135. There were no conflicting exact-image labels; 308 validation
duplicates were removed under the declared test-preserving rule. The 39 repeated
CSV rows and 92 unreferenced training files are separately logged.

## Completed preliminary experiment (does not meet that requirement)

- Notebook implementation: prepared for Kaggle.
- Local validation: passed notebook structure, unique cell IDs, all code-cell
  syntax, class totals, fixed conditions, and pinned checkpoint hashes. Confirmed
  HTTP 200 for the official dataset archive and pinned SigLIP 2 weights. No local
  feature extraction or classifier training was performed.
- Kaggle account: signed in successfully.
- Kaggle notebook: https://www.kaggle.com/code/aarooongeorge/siglip-2-on-realwaste-frozen-transfer-study
- Browser local-file permission: enabled. Kaggle's file importer stalled, so all
  20 prepared cells were entered directly in the notebook editor.
- Kaggle execution: all 21 evaluation conditions completed on a verified Tesla
  T4 session with Internet enabled. The experiment used one GPU.
  An initial CPU session stopped at the explicit CUDA check and was restarted
  after applying the GPU setting; no model experiment ran on that CPU session.
- Confirmed data audit: 4,752 readable images, zero exact pixel duplicates;
  expected counts for all nine classes. Each seed has 3,326 training, 713
  validation, and 713 test images.
- Confirmed hardware: Tesla T4, 14.56 GiB device memory; Kaggle Python 3.13.16,
  PyTorch 2.11.0+cu130.
- Both encoder extractions, all evaluations, plots, bootstrap, inference example,
  report generation, and final assertions completed without errors.
- Full-label mean macro-F1 was approximately 92.47% for SigLIP 2 and 81.67% for
  ResNet-18 (these preliminary numbers belong only to the RealWaste experiment).
- Kaggle Quick Version 1 was completed with “Save output for this version” chosen.
  Version ID: 367294970. Generated files total approximately 55.9 MiB.
- The preliminary GPU session was stopped after saving the completed version.

This file is updated as execution progresses. Results from the preliminary
pairing must not be attributed to the replacement model–dataset combination.
