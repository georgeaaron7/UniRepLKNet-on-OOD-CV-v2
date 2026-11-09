# IEEE Transactions source verification

Verified on 9 November 2026. The user requires **both** the model source paper
and the separate dataset source paper to be IEEE Transactions publications from
2024–2026. IEEE conferences, IEEE Access, preprints, and papers that merely cite a
model do not satisfy this requirement.

## The preliminary pairing does not qualify

- SigLIP 2's source is Tschannen et al. (2025), arXiv:2502.14786.
- RealWaste's source is Single et al. (2023), *Information* 14(12), 633 (MDPI),
  DOI 10.3390/info14120633.
- The completed Kaggle SigLIP 2–RealWaste experiment is preliminary work and must
  not be represented as satisfying the new publication requirement.

## Verified replacement source papers

| Role | Paper | IEEE journal publication | DOI / publisher record |
|---|---|---|---|
| A: existing model | Yiyuan Zhang, Xiaohan Ding, Xiangyu Yue, **Scaling Up Your Kernels: Large Kernel Design in ConvNets Toward Universal Representations** | *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 47(12), 11692–11707, December **2025** | [10.1109/TPAMI.2025.3600702](https://doi.org/10.1109/TPAMI.2025.3600702), [IEEE record](https://ieeexplore.ieee.org/document/11130640/) |
| B: existing dataset | Bingchen Zhao et al., **OOD-CV-v2: An Extended Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images** | *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(12), 11104–11118, December **2024** | [10.1109/TPAMI.2024.3462293](https://doi.org/10.1109/TPAMI.2024.3462293), [IEEE record](https://ieeexplore.ieee.org/document/10682066/) |

The journal, volume, issue, year, pages, and DOI were checked against the
publisher-deposited Crossref records, not inferred from an IEEE-style PDF header.
IEEE Xplore's automated web page returned a JavaScript/bot check, so publication
metadata was cross-checked with the authors' primary pages:

- [Official UniRepLKNet repository](https://github.com/AILab-CVC/UniRepLKNet)
  explicitly identifies the TPAMI 2025 journal version and links its manuscript.
- [Official dataset project page](https://genintel.de/ood-cv-v2.html) explicitly
  identifies TPAMI 2024 and provides classification, detection, and pose downloads.

Earlier conference/preprint versions exist. The eligibility dates above are the
IEEE Transactions **journal publication dates**, not the first preprint dates.

## Proposed combination and scope

**UniRepLKNet-P + OOD-CV-v2 image classification.** Apply the already released
ImageNet-pretrained model to the existing ten-class robustness dataset. Use a
frozen encoder and linear classifier initially, with a ResNet baseline. Document
classification accuracy, macro-F1, domain-shift results when the release exposes
the corresponding labels, runtime, memory, and failure cases. No new architecture,
loss, or dataset is needed. This is a transfer evaluation, not a claim to reproduce
the dataset paper's full fine-tuning protocol.

The official model repository lists UniRepLKNet-P at 10.7M parameters, 224-pixel
input, and 1.6G FLOPs. This supports a modest Kaggle inference/linear-probe study;
the Kaggle experiment subsequently completed successfully. Frozen feature
extraction over 29,782 audited images took 93.64 seconds for UniRepLKNet-P and
83.09 seconds for ResNet-18 on a T4. See `results_summary.md`.

## Novelty audit: candidate gap, not a guaranteed first

Exact and variant searches checked:

- `"UniRepLKNet" "OOD-CV-v2"`
- `"UniRepLKNet" "OOD-CV"`
- `"UniRepLKNet" "OODCV"`
- `"UniRepLKNet" "ROBIN" classification`

No direct evaluation of this exact combination was identified in the returned
results. In addition, the available [model journal manuscript](https://arxiv.org/html/2410.08049v1)
does not mention OOD-CV, and the available [dataset manuscript](https://arxiv.org/html/2304.10266v2)
does not mention UniRepLKNet. These manuscript checks do not exhaust later papers,
unindexed notebooks, or differences in the final published versions.

The publisher-deposited reference lists were also checked: 166 model-paper
references and 81 dataset-paper references contained no cross-component match.
This strengthens the scoped literature audit but still does not prove absence.

Defensible claim: **“We evaluate a candidate previously unreported model–dataset
combination; our targeted literature search found no direct prior evaluation.”**
Do not claim that the model itself, dataset itself, or transfer learning is novel.

## Reproducibility pointers

- Official model code commit: `b5677d8fb0e7b8af61eac39c1bf02727f5afc1d5`.
- Official checkpoint repository: `DingXiaoH/UniRepLKNet`, revision
  `af190f72f79f02d3212ffacc6b1d221a1cb94632`.
- Candidate checkpoint: `unireplknet_p_in1k_224_acc80.23.pth`.
- Classification data: [authors' OOD-CV-Cls.zip](https://drive.google.com/file/d/1bOLZYmxw0H4oDBP_afdmsbMt5LRS_f17/view).
- The project page recommends `train` for training and `phase-1` / `phase-2` as
  validation/test data. Released labels were verified and used directly; no
  labels were inferred from filenames. See the notebook data-integrity audit.
