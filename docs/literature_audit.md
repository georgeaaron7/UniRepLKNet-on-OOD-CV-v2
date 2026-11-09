# Literature and source audit

Search date: **9 November 2026**. Scope: targeted web search plus inspection of
official model cards, the dataset source, and related primary literature.

## Selected pair

- Model: [SigLIP 2 ViT-B/16 image encoder](https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli),
  checkpoint `4c3661e5ac879a276ddc5ddc6d3f0ecc78fd5d82`.
- Dataset: [UCI RealWaste](https://www.archive.ics.uci.edu/dataset/908/realwaste),
  4,752 images in nine classes; DOI [10.24432/C5SS4G](https://doi.org/10.24432/C5SS4G).
- Method: supervised linear probe on frozen visual embeddings; no text encoder.

The official model card confirms that the `v2_webli` checkpoint is **SigLIP 2**.
It is public and ungated. The SigLIP 2 [paper](https://arxiv.org/abs/2502.14786)
describes the model family. This implementation uses the image tower only.

## Queries and findings

| Query | Finding |
|---|---|
| `"SigLIP2" "RealWaste"` | No direct evaluation found in returned results. |
| `"SigLIP 2" "RealWaste"` | No direct evaluation found in returned results. |
| `"siglip2" "real waste"` | No direct evaluation found in returned results. |
| `"DINOv3" "RealWaste"` | Direct prior evaluation found; this candidate was rejected. |

Search engines may omit notebooks, theses, unindexed code, spelling variants, or
recent work. **Do not write “first ever” or “never done before.”** The defensible
statement is: “Our targeted search found no direct SigLIP 2–RealWaste evaluation;
we evaluate this candidate pairing under a reproducible Kaggle protocol.”
Recheck before submission if strict unpublished-combination novelty is required.

## Closest related work

1. [Single, Iranmanesh & Raad (2023)](https://doi.org/10.3390/info14120633)
   introduced RealWaste and studied existing CNNs. Their results are context;
   our image splits differ, so score differences are not controlled comparisons.
2. [Tschannen et al. (2025)](https://arxiv.org/abs/2502.14786) introduced SigLIP 2.
   The contribution here is application and evaluation, not the encoder.
3. [Funk et al. (2026)](https://doi.org/10.1016/j.clwas.2026.100475),
   “Evaluation of vision–language models for waste classification: Zero- and
   few-shot with a training-free tip-adapter,” evaluates foundation models on
   waste datasets including RealWaste and includes a DINOv3 1-NN baseline.
   This establishes that foundation-model transfer for waste is already studied;
   our scope is the specific SigLIP 2 image-tower linear-probe evaluation.
4. [ResNet-18 model card](https://huggingface.co/timm/resnet18.a1_in1k)
   documents the ImageNet-trained baseline. Its evaluation resolution is overridden
   to 224 in our notebook, and this choice is recorded.

## Source and license details

The UCI page lists RealWaste as CC BY 4.0; the [authors' repository](https://github.com/sam-single/realwaste)
README lists CC BY-NC-SA 4.0. Record this discrepancy and retain attribution.
The notebook downloads the original UCI archive for an academic experiment and
does not bundle raw images in the deliverables. Check terms before redistribution.
The selected SigLIP 2 and ResNet checkpoint cards list Apache-2.0.

## Claims supported only after execution

Performance, model ranking, class-specific failures, label efficiency, memory,
throughput, and runtime must come from notebook outputs. No expected accuracy is
presented as a result. The automatically generated report follows this rule.
