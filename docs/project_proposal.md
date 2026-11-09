# Project proposal

**Preliminary and superseded:** this SigLIP 2–RealWaste proposal does not meet
the clarified requirement for two recent IEEE Transactions source papers.
Use `tpami_project_proposal.md` for the current project.

**Title:** Compute-efficient landfill material classification with a frozen SigLIP 2 image encoder.

**Objective:** Evaluate SigLIP 2 ViT-B/16 on RealWaste's nine-class classification
task, using an existing encoder and existing dataset. Compare against a frozen
ResNet-18 under the same classifier-selection and data-split protocol.

**Novelty:** A candidate model–dataset application gap. The model, dataset,
logistic classifier, and evaluation methods already exist. A targeted literature
search found no direct evaluation of this pair on 9 November 2026. This is a
qualified search result, not proof that nobody has tried it. Related work exists.

**Research question:** How do frozen SigLIP 2 and ResNet-18 features compare in
macro-F1 and label efficiency on authentic landfill material images?

**Dataset:** RealWaste, 4,752 RGB images, nine material classes. No new annotation.
The original UCI archive is downloaded inside the Kaggle notebook. Exact decoded
image duplicates are removed before splitting and label conflicts stop execution.

**Method:** Pin pretrained checkpoints; extract 224 × 224 image embeddings with
frozen encoders; L2-normalize per image; fit a class-balanced logistic regression.
Select regularization using validation macro-F1. No backbone fine-tuning is needed.

**Evaluation:** Three predeclared stratified 70/15/15 image-level splits, seeds
42/123/2026. Identical partitions and nested label subsets for each model.
Primary: 100% of training labels. Secondary: 10% and 25% of training labels,
still using the full validation partition. Report macro-F1, accuracy, balanced
accuracy, weighted-F1, per-class results, confusion matrices, and failures.

**Compute:** One available Kaggle GPU, batch size 32, inference with float16
autocast; feature extraction once per encoder; CPU linear classifiers. No
multi-GPU setup, large-model training, or paid compute. Runtime is measured by
the notebook; an unmeasured duration is not promised.

**Deliverables:** Executable Kaggle notebook, saved outputs and classifier,
reproducibility records, comparison tables, visual error analysis, and a generated
report based on real held-out predictions. A negative result is still reportable.

**Scope limits:** Single-label image classification, not object detection or a
recyclability service. Random image-level splits cannot establish cross-site
generalization. Exact deduplication cannot remove all scene similarity. Different
pretraining datasets and model sizes prevent architecture-only causal conclusions.
