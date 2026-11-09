# Measured Kaggle results

Run date: 9 November 2026. Model: UniRepLKNet-P from IEEE TPAMI 2025
([paper A](https://doi.org/10.1109/TPAMI.2025.3600702)). Dataset: OOD-CV-v2 from
IEEE TPAMI 2024 ([paper B](https://doi.org/10.1109/TPAMI.2024.3462293)).

**Provenance:** Values below were observed in the executed notebook and verified
again in its persisted [Kaggle Version 1](https://www.kaggle.com/code/aarooongeorge/unireplknet-on-ood-cv-v2-tpami-transfer-study?scriptVersionId=367303127).
This file is a local summary, not a downloaded copy of the complete output report.
The full generated `experiment_report.md`, plots, predictions, CSV metrics, and
fitted classifier are saved in that version's Output tab and ZIP (114.65 MB).

## Evaluation

Official train labels fit the class-balanced logistic regression; official
phase-1 labels select C from [0.1, 1, 10, 100] by validation macro-F1. Phase-2
is the held-out test set. Both ImageNet-1K pretrained encoders remain frozen;
resize shorter side 256, center crop 224, ImageNet normalization, and L2 feature
normalization are identical. T4 float16 inference uses only one GPU. No test-time
adaptation or backbone fine-tuning is performed.

After the declared exact-RGB-duplicate audit: 8,588 training, 4,059 validation,
and 17,135 test images across ten classes. The audit removes 308 exact validation
duplicates, counts 39 repeated training annotation rows once per filename, and
logs 92 unreferenced training image files as unused. There are no conflicting
identical-pixel labels. Manifest SHA-256:
`690d0c19c7213237acc1b48b6096e9d0534a2a1d9eeb2069f9e719b3194ad5a4`.

## Full-label test scores

| Encoder | Selected C | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 10 | 69.8395% | 70.7613% | 69.7372% | 70.4450% |
| UniRepLKNet-P | 1 | 77.6306% | 77.8224% | 77.6905% | 77.8877% |

UniRepLKNet-P gains **7.9533 macro-F1 percentage points**. Each deterministic
full-label fit runs once. Secondary 10%/25% nested per-class subsets use seeds
42, 123, and 2026. Their results and sample SD are in the generated Kaggle report.
Full phase-1 validation labels are used in every condition. These subsets are
training-label sensitivity experiments, not strict few-shot learning.

## Compute and artifacts

Kaggle Tesla T4, 14.56 GiB device memory, batch 64. Feature extraction including
image loading and transforms took 83.09 seconds for ResNet-18 and 93.64 seconds
for UniRepLKNet-P; peak PyTorch allocated memory was 0.3110 and 0.5866 GiB,
respectively. These exclude checkpoint loading and warmup. The report recorded
306.6 seconds elapsed after setup. The saved-version 10-second runtime describes
Quick Save, not a new execution of the experiment.

All 15 evaluation rows (14 encoder/subset conditions plus majority baseline)
passed finite-score and range checks. All model computation ran on Kaggle. The
saved version was marked successful and its 40 output files were verified; the
GPU session was then stopped.

The notebook documents per-class and nuisance-group results, confusion matrices,
training-label efficiency, high-confidence errors, source/environment pins,
validation searches, predictions, a data manifest, and a fitted linear classifier.

## Interpretation limits

This is a candidate previously unreported combination of two existing research
contributions. Targeted searches found no direct prior evaluation, but cannot
prove worldwide absence. The protocol uses labeled shifted-domain validation;
it is not source-only domain generalization, and cannot be directly compared
with end-to-end benchmark training scores. ImageNet pretraining can overlap
benchmark source imagery. Exact hashing does not remove all crop/scene dependence.
The comparison does not isolate kernel size as the cause of the measured gain.
