# UniRepLKNet-P + OOD-CV-v2

**Title:** Frozen large-kernel representations for classification under real
distribution shifts: a UniRepLKNet-P evaluation on OOD-CV-v2.

**Paper A (existing model):** Zhang, Ding & Yue, *Scaling Up Your Kernels: Large
Kernel Design in ConvNets Toward Universal Representations*, IEEE TPAMI 47(12),
11692–11707, 2025. DOI: [10.1109/TPAMI.2025.3600702](https://doi.org/10.1109/TPAMI.2025.3600702).

**Paper B (existing dataset):** Zhao et al., *OOD-CV-v2: An Extended Benchmark for
Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural
Images*, IEEE TPAMI 46(12), 11104–11118, 2024.
DOI: [10.1109/TPAMI.2024.3462293](https://doi.org/10.1109/TPAMI.2024.3462293).

Both journal source papers qualify for the 2024–2026 requirement. Publication
metadata, authors' primary pages, code/weight commits, and a qualified novelty
audit are recorded in `ieee_source_verification.md`.

**Combination:** Use the released ImageNet-1K UniRepLKNet-P checkpoint as a frozen
feature encoder for OOD-CV-v2's existing ten-class classification task. Fit a
standard logistic regression on its features. This applies the model from A to
the dataset from B; it does not add an architecture, loss function, or dataset.

**Question:** Does a compact pretrained large-kernel ConvNet transfer effectively
to real shifts of shape, pose, texture, context, weather, and occlusion, relative
to a similarly sized ImageNet-pretrained ResNet-18 baseline?

**Novelty scope:** A candidate previously unreported model–dataset application.
Targeted searches found no direct UniRepLKNet–OOD-CV-v2 evaluation. Neither
available source manuscript named the other component, and the publisher-deposited
reference lists had no cross-component match (166 model-paper and 81 dataset-paper
references). This is not proof of worldwide absence. No state-of-the-art claim.

**Protocol:** Use the official `train` labels to fit classifiers, `phase-1` labels
to select regularization by macro-F1, and `phase-2` as the held-out test set,
following the authors' stated project-page option. This uses labeled shifted-domain
validation and must not be described as source-only domain generalization.

**Data integrity:** The downloaded official archive is 692,430,073 bytes. The
training CSV has 39 repeated identical annotation rows, with no conflicting
filename labels. These are logged and each annotated filename is used once.
The archive contains 92 training image files absent from its label CSV; they are
logged and unused. No labels are inferred for them. Exact decoded-image duplicate
groups are audited before fitting; ambiguous identical pixels and cross-partition
duplicates are handled according to the documented, test-preserving rule.

**Experiments:** Compare frozen UniRepLKNet-P and ResNet-18 using identical data
and 224-pixel deterministic preprocessing. Primary: all training labels.
Secondary: 10% and 25% nested per-class label subsets at three seeds. Select C
using validation only. Report accuracy, balanced accuracy, macro-F1, weighted-F1,
per-class results, released nuisance-group breakdowns, confusion matrices,
training-label sensitivity, observed failures, runtime, and GPU memory.

**Compute:** Kaggle notebooks only. One T4 GPU; frozen feature extraction once
per encoder; CPU linear probes. UniRepLKNet-P is small enough to avoid full-model
training. Use the authors' standard PyTorch implementation without compiling a
custom CUDA kernel. Actual resource measurements come from the notebook.

**Deliverables:** Kaggle notebook and saved output version, measured Markdown
report, plots, metrics/validation/prediction CSVs, data and source manifests,
package lock, and the fitted linear classifier. Negative outcomes are documented.

**Limits:** Pretraining and benchmark source images can overlap. Exact hashes
do not remove all near-duplicate crops. Different representations prevent causal
claims that kernel size alone explains score differences. This frozen-transfer
protocol is distinct from end-to-end fine-tuning results in the source papers.
