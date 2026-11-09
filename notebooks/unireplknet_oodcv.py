# %% [markdown]
# # UniRepLKNet-P on OOD-CV-v2: an IEEE Transactions application study
#
# **Model from paper A:** Zhang, Ding & Yue, *Scaling Up Your Kernels: Large Kernel
# Design in ConvNets Toward Universal Representations*, IEEE TPAMI 47(12),
# 11692–11707, **2025**. DOI: https://doi.org/10.1109/TPAMI.2025.3600702
#
# **Dataset from paper B:** Zhao et al., *OOD-CV-v2: An Extended Benchmark for
# Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural
# Images*, IEEE TPAMI 46(12), 11104–11118, **2024**.
# DOI: https://doi.org/10.1109/TPAMI.2024.3462293
#
# Both source **journal publications** meet the 2024–2026 requirement. Earlier
# preprints/conference versions exist. We combine the existing UniRepLKNet-P
# checkpoint with the existing OOD-CV-v2 classification data; no architecture,
# loss, or dataset is invented. Targeted searches on 9 November 2026 found no
# direct prior evaluation of this pair; this is a candidate application gap,
# not proof of a world-first result.
#
# **Research question:** How well do the frozen features of a small large-kernel
# ConvNet transfer to real distribution shifts, compared with a similarly sized
# ImageNet ResNet-18? A negative result is reportable.
#
# **Kaggle:** enable Internet and a GPU, then Run All. One GPU is used. Outputs
# and the measured report are saved under `/kaggle/working/unireplknet_oodcv/`.

# %%
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "timm==1.0.31", "gdown==5.2.0"])

# %%
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import shutil
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import gdown
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from IPython.display import FileLink, Markdown, display
from PIL import Image, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

assert Path("/kaggle/working").is_dir(), "Execute on Kaggle."
assert torch.cuda.is_available(), "Enable the GPU setting, wait for it to apply, then start the session."
DEVICE = torch.device("cuda:0")
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

CFG = {"batch_size": 64, "num_workers": 2, "image_size": 224,
       "subset_seeds": [42, 123, 2026], "label_fractions": [0.1, 0.25, 1.0],
       "C_grid": [0.1, 1.0, 10.0, 100.0], "max_iter": 3000, "amp": True}
SOURCE = {
    "model_paper": "10.1109/TPAMI.2025.3600702",
    "dataset_paper": "10.1109/TPAMI.2024.3462293",
    "code_commit": "b5677d8fb0e7b8af61eac39c1bf02727f5afc1d5",
    "weight_repo": "DingXiaoH/UniRepLKNet",
    "weight_revision": "af190f72f79f02d3212ffacc6b1d221a1cb94632",
    "weight_file": "unireplknet_p_in1k_224_acc80.23.pth",
    "resnet_repo": "timm/resnet18.a1_in1k",
    "resnet_revision": "491b427b45c94c7fb0e78b5474cc919aff584bbf",
    "dataset_id": "1bOLZYmxw0H4oDBP_afdmsbMt5LRS_f17",
    "dataset_project": "https://genintel.de/ood-cv-v2.html",
}
CLASSES = ["aeroplane", "bicycle", "boat", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train"]
LABELS = np.arange(len(CLASSES))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
OUT = Path("/kaggle/working/unireplknet_oodcv")
DATA = Path("/kaggle/temp/oodcv_tpami")
OUT.mkdir(exist_ok=True, parents=True)
DATA.mkdir(exist_ok=True, parents=True)
(OUT / "run_complete.json").unlink(missing_ok=True)
START = datetime.now(timezone.utc).isoformat()
T0 = time.perf_counter()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

ENV = {"started_utc": START, "python": sys.version, "gpu": torch.cuda.get_device_name(0),
       "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
       "cuda": torch.version.cuda,
       "packages": {p: importlib.metadata.version(p) for p in
                    ["torch", "torchvision", "timm", "gdown", "numpy", "pandas", "scikit-learn", "Pillow"]}}
write_json(OUT / "config.json", {"experiment": CFG, "sources": SOURCE, "classes": CLASSES})
write_json(OUT / "environment.json", ENV)
(OUT / "requirements-lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
display(pd.Series(ENV).to_frame("value"))

# %% [markdown]
# ## Official release, labels, and leakage checks
#
# The authors provide `train`, `phase-1`, and `phase-2` with released class labels.
# Following their project-page suggestion, we train on `train`, select C using
# `phase-1` **validation** labels, and evaluate only after selection on `phase-2`.
# Thus this protocol uses labeled shifted-domain validation; it is not source-only
# domain generalization. Nuisance labels are used for descriptive test breakdowns.
#
# Exact decoded RGB hashes are audited. Ambiguous identical images with conflicting
# class labels are excluded and recorded. Remaining exact duplicates are removed
# with priority test → validation → training, preserving test representatives.
# Counts before and after filtering are reported. Near-duplicate crops and scene
# dependence are not fully removed. This evaluation does not reproduce the dataset
# paper's end-to-end training scores.

# %%
archive = DATA / "OOD-CV-Cls.zip"
if not all((DATA / p / "labels.csv").is_file() for p in ["train", "phase-1", "phase-2"]):
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        archive.unlink(missing_ok=True)
        result = gdown.download(id=SOURCE["dataset_id"], output=str(archive), quiet=False)
        assert result and zipfile.is_zipfile(archive), "Download failed: attach the official OOD-CV-Cls release under /kaggle/input."
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            assert (DATA / item.filename).resolve().is_relative_to(DATA.resolve())
        z.extractall(DATA)
archive_hash = None
if archive.is_file():
    with archive.open("rb") as f:
        archive_hash = hashlib.file_digest(f, "sha256").hexdigest()

records, nuisance_map, duplicate_annotation_rows, unreferenced_image_files = [], {}, {}, {}
for phase, split in [("train", "train"), ("phase-1", "validation"), ("phase-2", "test")]:
    phase_root = DATA / phase
    labels_table = pd.read_csv(phase_root / "labels.csv")
    assert {"imgs", "labels"}.issubset(labels_table.columns)
    # The official train CSV repeats 39 identical annotation rows. Record these
    # and keep each filename once; conflicting filename labels still stop the run.
    assert labels_table.groupby("imgs").labels.nunique().max() == 1
    duplicate_annotation_rows[phase] = int(labels_table.duplicated("imgs").sum())
    labels_table[labels_table.duplicated("imgs", keep=False)].to_csv(OUT / f"duplicate_annotation_rows_{phase}.csv", index=False)
    labels_table = labels_table.drop_duplicates("imgs").reset_index(drop=True)
    assert set(labels_table.labels) == set(CLASSES)
    candidates = {}
    for path in phase_root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            assert path.name not in candidates, f"Ambiguous duplicate filename in release: {path.name}"
            candidates[path.name] = path
    assert set(labels_table.imgs).issubset(candidates), f"Missing labeled images in {phase}"
    # Use only filenames explicitly listed by the authors' CSV. The train
    # archive also contains 92 image files not referenced by that CSV.
    unused = sorted(set(candidates) - set(labels_table.imgs))
    unreferenced_image_files[phase] = len(unused)
    write_json(OUT / f"unreferenced_image_files_{phase}.json", unused)
    nuisance_file = phase_root / "nuisance2filenames.json"
    if nuisance_file.exists():
        nuisance_map[split] = json.loads(nuisance_file.read_text())
    for row in tqdm(labels_table.itertuples(index=False), total=len(labels_table), desc=f"Audit {phase}"):
        path = candidates[row.imgs]
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        records.append({"path": str(path), "relative_path": str(path.relative_to(DATA)),
                        "filename": path.name, "class_name": row.labels,
                        "label": CLASS_TO_IDX[row.labels], "split": split, "pixel_sha256": pixel_hash})

raw_manifest = pd.DataFrame(records)
ambiguous = raw_manifest.groupby("pixel_sha256").label.nunique()
bad_hashes = set(ambiguous[ambiguous > 1].index)
raw_manifest[raw_manifest.pixel_sha256.isin(bad_hashes)].to_csv(OUT / "ambiguous_exact_duplicates.csv", index=False)
raw_manifest[raw_manifest.duplicated("pixel_sha256", keep=False)].to_csv(OUT / "exact_duplicate_groups.csv", index=False)
clean = raw_manifest[~raw_manifest.pixel_sha256.isin(bad_hashes)].copy()
clean["priority"] = clean.split.map({"test": 0, "validation": 1, "train": 2})
manifest = clean.sort_values(["priority", "relative_path"]).drop_duplicates("pixel_sha256").drop(columns="priority")
manifest = manifest.sort_values(["split", "relative_path"]).reset_index(drop=True)
manifest["image_id"] = np.arange(len(manifest))
manifest.to_csv(OUT / "manifest.csv", index=False)
counts = pd.DataFrame({"released": raw_manifest.groupby("split").size(), "evaluated": manifest.groupby("split").size()})
counts.to_csv(OUT / "split_counts.csv")
MANIFEST_HASH = stable_hash(manifest[["relative_path", "label", "split", "pixel_sha256"]].to_dict("records"))
AUDIT = {"archive_sha256": archive_hash, "manifest_sha256": MANIFEST_HASH,
         "duplicate_annotation_rows": duplicate_annotation_rows,
         "unreferenced_image_files": unreferenced_image_files,
         "released_images": len(raw_manifest), "evaluated_images": len(manifest),
         "excluded_ambiguous_images": int(raw_manifest.pixel_sha256.isin(bad_hashes).sum()),
         "removed_total_images": len(raw_manifest) - len(manifest),
         "counts": counts.to_dict("index")}
write_json(OUT / "data_audit.json", AUDIT)
write_json(OUT / "nuisance_groups.json", nuisance_map)
y = manifest.label.to_numpy()
SPLIT = {s: manifest.index[manifest.split == s].to_numpy() for s in ["train", "validation", "test"]}
assert all(set(y[ids]) == set(LABELS) for ids in SPLIT.values())
display(counts)
print(AUDIT)

# %% [markdown]
# ## Frozen encoders and fixed preprocessing
#
# Load the authors' pinned UniRepLKNet-P code and ImageNet-1K checkpoint, replace
# only its classification head with identity, and use the official equivalent
# reparameterization for inference. Both encoders use deterministic 224-pixel
# center crops, ImageNet normalization, and frozen weights. No custom CUDA
# extension is required. The ResNet-18 checkpoint is also ImageNet-1K pretrained.
#
# Features are computed once per encoder with float16 autocast and independently
# L2-normalized. Classifier fitting is CPU-only. C is selected from a predeclared
# grid using validation macro-F1, never phase-2 scores. No test-time adaptation.

# %%
code_url = f"https://raw.githubusercontent.com/AILab-CVC/UniRepLKNet/{SOURCE['code_commit']}/unireplknet.py"
response = requests.get(code_url, timeout=30)
response.raise_for_status()
code_path = OUT / "official_unireplknet.py"
code_path.write_bytes(response.content)
SOURCE["code_sha256"] = hashlib.sha256(response.content).hexdigest()
module_spec = importlib.util.spec_from_file_location("official_unireplknet", code_path)
official = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = official
module_spec.loader.exec_module(official)

TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
PREPROCESSING = {"resize_shorter_side": 256, "center_crop": 224, "interpolation": "bicubic",
                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "feature_normalization": "l2"}

class Images(Dataset):
    def __len__(self):
        return len(manifest)

    def __getitem__(self, index):
        with Image.open(manifest.iloc[index].path) as image:
            return TRANSFORM(ImageOps.exif_transpose(image).convert("RGB"))

def load_model(name):
    if name == "unireplknet_p":
        model = official.unireplknet_p(in_1k_pretrained=False, num_classes=1000, attempt_use_lk_impl=False)
        checkpoint_path = hf_hub_download(repo_id=SOURCE["weight_repo"], revision=SOURCE["weight_revision"],
                                          filename=SOURCE["weight_file"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state, strict=True)
        model.head = torch.nn.Identity()
        model.eval().reparameterize_unireplknet()
    else:
        model = timm.create_model(f"hf-hub:{SOURCE['resnet_repo']}@{SOURCE['resnet_revision']}", pretrained=True, num_classes=0)
    return model.eval().requires_grad_(False).to(DEVICE)

@torch.inference_mode()
def features_for(name):
    identity = {"model": name, "sources": SOURCE, "manifest": MANIFEST_HASH,
                "preprocessing": PREPROCESSING, "environment": ENV["packages"], "amp": CFG["amp"], "gpu": ENV["gpu"]}
    feature_file = OUT / f"features_{name}_{stable_hash(identity)[:16]}.npz"
    meta_file = feature_file.with_suffix(".json")
    if feature_file.exists() and meta_file.exists():
        with np.load(feature_file, allow_pickle=False) as cached:
            features = cached["features"]
            assert np.array_equal(cached["image_ids"], manifest.image_id)
        meta = json.loads(meta_file.read_text())
        assert meta["identity"] == identity
        return features, meta
    model = load_model(name)
    loader = DataLoader(Images(), batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=CFG["amp"]):
        model(torch.zeros(1, 3, 224, 224, device=DEVICE))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    chunks = []
    for batch in tqdm(loader, desc=name):
        with torch.autocast("cuda", dtype=torch.float16, enabled=CFG["amp"]):
            feature = model(batch.to(DEVICE, non_blocking=True))
        assert feature.ndim == 2
        chunks.append(feature.float().cpu().numpy())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - began
    features = normalize(np.concatenate(chunks).astype(np.float32), norm="l2", copy=False)
    assert len(features) == len(manifest) and np.isfinite(features).all()
    meta = {"identity": identity, "parameters": sum(p.numel() for p in model.parameters()),
            "feature_dimension": features.shape[1], "seconds": elapsed,
            "images_per_second": len(features) / elapsed,
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
    np.savez_compressed(feature_file, features=features, image_ids=manifest.image_id.to_numpy())
    write_json(meta_file, meta)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return features, meta

FEATURES, COMPUTE = {}, {}
for name in ["resnet18", "unireplknet_p"]:
    FEATURES[name], COMPUTE[name] = features_for(name)
write_json(OUT / "compute.json", COMPUTE)
write_json(OUT / "preprocessing.json", PREPROCESSING)
display(pd.DataFrame(COMPUTE).T.drop(columns="identity"))

# %% [markdown]
# ## Classifier selection and test evaluation
#
# Primary condition: all released training labels, evaluated once per encoder
# because this linear classifier is deterministic. Secondary: nested per-class
# 10% and 25% training-label subsets at seeds 42, 123, and 2026. The full phase-1
# validation labels are used in every condition; this is not a strict few-shot test.
# The sample SD for the smaller subsets measures label-selection variability.
#
# Save all validation candidates and held-out predictions. The majority baseline
# uses only the most common training class. Nuisance groups may overlap and are
# evaluated as supplied by the dataset authors.

# %%
def metric_dict(truth, pred):
    return {"accuracy": accuracy_score(truth, pred), "balanced_accuracy": balanced_accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth, pred, labels=LABELS, average="weighted", zero_division=0)}

SUBSETS = {}
for seed in CFG["subset_seeds"]:
    rng = np.random.default_rng(seed)
    by_class = {c: rng.permutation(SPLIT["train"][y[SPLIT["train"]] == c]) for c in LABELS}
    for fraction in [0.1, 0.25]:
        SUBSETS[(seed, fraction)] = np.sort(np.concatenate([ids[:max(1, int(np.ceil(fraction * len(ids))))]
                                                           for ids in by_class.values()]))
SUBSETS[(42, 1.0)] = SPLIT["train"]
write_json(OUT / "training_subsets.json", {f"{seed}_{fraction}": ids.tolist() for (seed, fraction), ids in SUBSETS.items()})
rows, searches, predictions, class_rows, nuisance_rows, PROBES = [], [], [], [], [], {}
for (seed, fraction), train_ids in SUBSETS.items():
    for name, X in FEATURES.items():
        best, best_C, best_score = None, None, -np.inf
        began = time.perf_counter()
        for C in CFG["C_grid"]:
            clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=CFG["max_iter"], tol=1e-4)
            with threadpool_limits(limits=2), warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                clf.fit(X[train_ids], y[train_ids])
            score = f1_score(y[SPLIT["validation"]], clf.predict(X[SPLIT["validation"]]),
                             labels=LABELS, average="macro", zero_division=0)
            searches.append({"model": name, "seed": seed, "label_fraction": fraction, "C": C, "validation_macro_f1": score})
            if score > best_score:
                best, best_C, best_score = clf, C, score
        test_ids = SPLIT["test"]
        pred = best.predict(X[test_ids])
        probability = best.predict_proba(X[test_ids])
        key = {"model": name, "seed": seed, "label_fraction": fraction}
        rows.append({**key, "C": best_C, "n_train": len(train_ids), "validation_macro_f1": best_score,
                     "probe_selection_seconds": time.perf_counter() - began, **metric_dict(y[test_ids], pred)})
        part = manifest.iloc[test_ids][["image_id", "relative_path", "filename", "label", "class_name"]].copy()
        for k, v in key.items():
            part[k] = v
        part["prediction"], part["confidence"] = pred, probability.max(axis=1)
        part["predicted_class"] = [CLASSES[c] for c in pred]
        predictions.append(part)
        PROBES[(name, seed, fraction)] = best
        if fraction == 1.0:
            detail = classification_report(y[test_ids], pred, labels=LABELS, target_names=CLASSES,
                                           output_dict=True, zero_division=0)
            class_rows.extend({"model": name, "class_name": c, **detail[c]} for c in CLASSES)
            for nuisance, filenames in nuisance_map.get("test", {}).items():
                mask = part.filename.isin(set(filenames)).to_numpy()
                if mask.any():
                    nuisance_rows.append({"model": name, "nuisance": nuisance, "n_test": int(mask.sum()),
                                          **metric_dict(y[test_ids][mask], pred[mask])})
        print({**key, "C": best_C, **metric_dict(y[test_ids], pred)})
        pd.DataFrame(rows).to_csv(OUT / "metrics_per_run.csv", index=False)
        pd.DataFrame(searches).to_csv(OUT / "validation_search.csv", index=False)

majority = int(np.bincount(y[SPLIT["train"]], minlength=len(CLASSES)).argmax())
rows.append({"model": "majority", "seed": 42, "label_fraction": 1.0,
             **metric_dict(y[SPLIT["test"]], np.full(len(SPLIT["test"]), majority))})
metrics = pd.DataFrame(rows)
predictions = pd.concat(predictions, ignore_index=True)
per_class, nuisances = pd.DataFrame(class_rows), pd.DataFrame(nuisance_rows)
metrics.to_csv(OUT / "metrics_per_run.csv", index=False)
predictions.to_csv(OUT / "test_predictions.csv", index=False)
per_class.to_csv(OUT / "per_class_metrics.csv", index=False)
nuisances.to_csv(OUT / "nuisance_metrics.csv", index=False)
primary = metrics[metrics.label_fraction == 1.0].set_index("model")
display(primary)
display(nuisances)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, name in zip(axes, FEATURES):
    part = predictions[(predictions.model == name) & (predictions.label_fraction == 1.0)]
    matrix = confusion_matrix(part.label, part.prediction, labels=LABELS)
    row_norm = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if row_norm[i, j] > 0.55 else "black")
    ax.set_xticks(LABELS, CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(LABELS, CLASSES, fontsize=8)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{name}: phase-2 counts, row-normalized color")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", dpi=170, bbox_inches="tight")
plt.show()
plt.close(fig)

if len(nuisances):
    pivot = nuisances.pivot(index="nuisance", columns="model", values="accuracy") * 100
    ax = pivot.plot.bar(figsize=(9, 4.5), ylabel="Test accuracy (%)", title="Accuracy by released nuisance group")
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "nuisance_accuracy.png", dpi=160)
    plt.show()
    plt.close(ax.figure)

label_summary = metrics[metrics.model != "majority"].groupby(["model", "label_fraction"]).macro_f1.agg(["mean", "std", "count"]).reset_index()
label_summary.to_csv(OUT / "label_efficiency.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
for name in FEATURES:
    part = label_summary[label_summary.model == name].sort_values("label_fraction")
    ax.errorbar(part.label_fraction * 100, part["mean"] * 100, yerr=part["std"].fillna(0) * 100,
                marker="o", capsize=3, label=name)
ax.set(xlabel="Training labels used (%)", ylabel="Phase-2 macro-F1 (%)", title="Label efficiency; SD for subset seeds only")
ax.set_xticks([10, 25, 100])
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "label_efficiency.png", dpi=160)
plt.show()
plt.close(fig)

errors = predictions[(predictions.model == "unireplknet_p") & (predictions.label_fraction == 1.0)
                     & (predictions.label != predictions.prediction)].sort_values("confidence", ascending=False)
errors.head(20).to_csv(OUT / "error_examples.csv", index=False)
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax in axes.flat:
    ax.axis("off")
for ax, row in zip(axes.flat, errors.head(6).itertuples()):
    with Image.open(manifest.iloc[row.image_id].path) as image:
        ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
    ax.set_title(f"True: {row.class_name}\nPred: {row.predicted_class} ({row.confidence:.2f})", fontsize=9)
fig.suptitle("UniRepLKNet-P: highest-confidence phase-2 errors (probabilities uncalibrated)")
fig.tight_layout()
fig.savefig(OUT / "error_examples.png", dpi=160)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Measured report and saved classifier
#
# The report is generated from actual held-out predictions, including an unfavorable
# outcome if observed. Published scores from different training protocols are not
# directly compared. The saved linear classifier is fitted only on the official
# training partition, with C selected on phase-1. Restore it with the pinned
# encoder, saved transforms, and L2 feature normalization for single-image inference.

# %%
joblib.dump({"classifier": PROBES[("unireplknet_p", 42, 1.0)], "classes": CLASSES,
             "sources": SOURCE, "preprocessing": PREPROCESSING}, OUT / "unireplknet_p_linear_probe.joblib")

def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(map(str, columns)) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)

delta = float(primary.loc["unireplknet_p", "macro_f1"] - primary.loc["resnet18", "macro_f1"])
metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
primary_table = primary[metric_names].mul(100).reset_index()
nuisance_table = nuisances.copy()
if len(nuisance_table):
    nuisance_table[metric_names] *= 100
compute_table = pd.DataFrame(COMPUTE).T.drop(columns="identity").reset_index(names="model")
report = f"""# UniRepLKNet-P on OOD-CV-v2: an IEEE Transactions application study

Run started: {START}. Report generated: {datetime.now(timezone.utc).isoformat()}.

## Abstract
We combine the existing UniRepLKNet-P model from a 2025 IEEE TPAMI journal paper
with the existing OOD-CV-v2 classification dataset from a separate 2024 IEEE TPAMI
journal paper. Both source journal publications meet the 2024–2026 requirement.
Using frozen ImageNet-pretrained features and a validation-selected linear
classifier, UniRepLKNet-P achieves {primary.loc['unireplknet_p', 'accuracy'] * 100:.2f}%
test accuracy and {primary.loc['unireplknet_p', 'macro_f1'] * 100:.2f}% macro-F1.
Its macro-F1 minus the ResNet-18 baseline is {delta * 100:+.2f} percentage points.
No new architecture, loss function, or dataset is proposed.

## Source papers and contribution
**A — model:** Zhang, Ding & Yue, Scaling Up Your Kernels: Large Kernel Design in
ConvNets Toward Universal Representations. IEEE Transactions on Pattern Analysis
and Machine Intelligence 47(12), 11692–11707, 2025.
https://doi.org/10.1109/TPAMI.2025.3600702

**B — dataset:** Zhao et al., OOD-CV-v2: An Extended Benchmark for Robustness to
Out-of-Distribution Shifts of Individual Nuisances in Natural Images. IEEE
Transactions on Pattern Analysis and Machine Intelligence 46(12), 11104–11118, 2024.
https://doi.org/10.1109/TPAMI.2024.3462293

The application question is whether a compact large-kernel ConvNet provides useful
frozen representations for real distribution shifts. Our targeted search on
9 November 2026 found no direct UniRepLKNet–OOD-CV-v2 evaluation. Neither available
source manuscript named the other component. This is a candidate previously
unreported pairing; it is not proof of a first-ever combination. Earlier
conference/preprint versions of both source papers exist; the dates above are
the IEEE journal publication dates.

## Data and protocol
Official authors' classification archive, downloaded from the dataset project:
https://genintel.de/ood-cv-v2.html

{markdown_table(counts.reset_index())}

Repeated annotation rows (counted once per filename): {duplicate_annotation_rows}.
Image files not referenced by released label CSVs (unused): {unreferenced_image_files}.
Removed images: {AUDIT['removed_total_images']}, including
{AUDIT['excluded_ambiguous_images']} with conflicting labels for identical pixels.
Manifest SHA-256: `{MANIFEST_HASH}`. Details and excluded image groups are saved.
Exact RGB duplicates are handled with test, validation, then training priority;
this does not detect all near-duplicates or correlated crops. Numbers after this
audit define our protocol and may differ from the unfiltered published benchmark.

Training uses `train`; C is selected on labeled `phase-1`; `phase-2` is held out
until selection. The authors recommend this partition use. Because validation is
from a shifted domain, this is not strictly source-only domain generalization.
Both encoders use ImageNet-1K pretrained weights, deterministic resize to 256,
center crop to 224, ImageNet normalization, and per-image L2 feature normalization.
We train only a class-balanced logistic regression, selecting C from {CFG['C_grid']}
using validation macro-F1. There is no backbone fine-tuning or test-time adaptation.
The official UniRepLKNet inference reparameterization is used without a custom
CUDA extension. Code/checkpoint commits, source hash, and environment are saved.

## Primary held-out results (all training labels)
All four score columns are percentages. Full-data fits are deterministic and run
once per encoder; no artificial standard deviation over repeated identical fits
is reported.

{markdown_table(primary_table)}

![Confusion matrices](confusion_matrices.png)

## Released nuisance groups
Groups may overlap. Scores are percentages, and group macro-F1 averages over all
ten labels with absent classes assigned zero; accuracy is the primary group metric.

{markdown_table(nuisance_table) if len(nuisance_table) else 'No nuisance mapping was supplied.'}

![Nuisance accuracy](nuisance_accuracy.png)

## Training-label efficiency
10% and 25% subsets are nested per class, at seeds {CFG['subset_seeds']}; 100% uses
all training labels once. The full validation labels are required in each condition,
so these results are not strict few-shot learning. Mean and SD below are proportions;
SD is undefined for the single full-data fit.

{markdown_table(label_summary)}

![Label efficiency](label_efficiency.png)

## Error analysis
Per-class precision, recall, F1, support, and all held-out predictions are saved.
The highest-confidence UniRepLKNet-P mistakes are shown without adjusting the
model afterward. Probabilities are not calibrated. Visual examples describe
failures but do not establish their causes.

![Error examples](error_examples.png)

## Compute
Kaggle hardware: {ENV['gpu']}, {ENV['gpu_memory_gib']:.2f} GiB device memory.
Batch size {CFG['batch_size']}; one GPU; float16 autocast for frozen inference.
Extraction time includes loading, transforms, inference, and transfers, and excludes
checkpoint loading and warmup. Memory is peak PyTorch allocated GPU memory, not all
driver allocations. Linear-probe fitting runs on the CPU.

{markdown_table(compute_table)}

Elapsed work after setup: {time.perf_counter() - T0:.1f} seconds.

## Limitations
- This is a model–dataset application, not a novel model or a reproduction of
  end-to-end trained benchmark scores.
- Both encoders saw ImageNet-1K in pretraining. The benchmark training source also
  includes ImageNet/PASCAL3D+ imagery; pretraining overlap cannot be treated as
  evidence of generalization to previously unseen object categories.
- Exact deduplication does not eliminate all scene/crop dependence. No independent
  capture-group uncertainty estimate is available here.
- Model sizes and learned representations differ; this comparison cannot isolate
  kernel size as a causal reason for performance differences.
- Literature searches cannot guarantee that no unindexed prior combination exists.
- No state-of-the-art claim is made against scores under other protocols.

## Conclusion
For this fixed frozen-transfer protocol, UniRepLKNet-P changes phase-2 macro-F1 by
{delta * 100:+.2f} percentage points relative to ResNet-18. This measured outcome,
the per-nuisance results, compute records, and error examples document the proposed
combination of two existing IEEE Transactions research contributions.

## Artifacts and sources
The output bundle contains metrics, validation searches, held-out predictions,
duplicate and split manifests, plots, the saved linear classifier, pinned official
model source, and environment records. Raw dataset images and pretrained weights
are obtained from their authors and are not redistributed in the results bundle.
Model code and checkpoints: https://github.com/AILab-CVC/UniRepLKNet
Dataset: https://genintel.de/ood-cv-v2.html
"""
(OUT / "experiment_report.md").write_text(report, encoding="utf-8")
display(Markdown(report))
assert len(metrics) == 15 and metrics[metric_names].notna().all().all()
assert ((metrics[metric_names] >= 0) & (metrics[metric_names] <= 1)).all().all()
write_json(OUT / "run_complete.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                                       "evaluation_conditions": len(metrics), "manifest_sha256": MANIFEST_HASH})
bundle_path = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
print("IEEE-SOURCED EXPERIMENT COMPLETE")
display(FileLink(str(OUT / "experiment_report.md")))
display(FileLink(bundle_path))
