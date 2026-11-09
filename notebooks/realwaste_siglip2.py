# %% [markdown]
# # SigLIP 2 on RealWaste: a compute-efficient transfer study
#
# **Question:** Can a frozen SigLIP 2 image encoder classify authentic landfill
# materials better than a frozen ImageNet ResNet-18 under the same linear-probe protocol?
#
# **Contribution:** evaluate an existing model on an existing dataset. No new
# architecture, loss, or dataset is claimed. A targeted search on 9 November 2026
# found no direct SigLIP 2–RealWaste evaluation; this is a candidate application
# gap, not proof of a world-first result. Related foundation-model waste studies exist.
#
# **Kaggle:** select a GPU accelerator, enable Internet, then **Run All**.
# Downloads are public; no Kaggle or Hugging Face API token is needed inside this notebook.
# Everything computational runs here. A single GPU is sufficient; a second GPU is unused.
# Results are written to `/kaggle/working/realwaste_siglip2/` and bundled at the end.
# No scores below are invented: the report is generated only after evaluation succeeds.

# %%
import subprocess
import sys

# Preserve Kaggle's installed CUDA/PyTorch stack. Install only the small model library.
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "timm==1.0.31"])

# %%
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import time
import warnings
import zipfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import timm
import torch
from IPython.display import FileLink, Markdown, display
from PIL import Image, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

assert Path("/kaggle/working").is_dir(), "Run this notebook on Kaggle, with a GPU and Internet enabled."
assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator before Run All."
DEVICE = torch.device("cuda:0")
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CFG = {
    "seeds": [42, 123, 2026],
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "label_fractions": [0.10, 0.25, 1.0],
    "C_grid": [0.01, 0.1, 1.0, 10.0],
    "batch_size": 32,   # Lower to 16 if the selected GPU runs out of memory.
    "num_workers": 2,   # Set to 0 if Kaggle reports a worker error.
    "max_iter": 2000,
    "image_size": 224,
    "amp": True,
    "dataset_url": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
}
MODEL_SPECS = {
    "resnet18": {
        "repo": "timm/resnet18.a1_in1k",
        "revision": "491b427b45c94c7fb0e78b5474cc919aff584bbf",
    },
    "siglip2": {
        "repo": "timm/vit_base_patch16_siglip_224.v2_webli",
        "revision": "4c3661e5ac879a276ddc5ddc6d3f0ecc78fd5d82",
    },
}
EXPECTED_COUNTS = {
    "Cardboard": 461, "Food Organics": 411, "Glass": 420,
    "Metal": 790, "Miscellaneous Trash": 495, "Paper": 500,
    "Plastic": 921, "Textile Trash": 318, "Vegetation": 436,
}
CLASSES = sorted(EXPECTED_COUNTS)
LABELS = np.arange(len(CLASSES))
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
OUT = Path("/kaggle/working/realwaste_siglip2")
DATA = Path("/kaggle/temp/realwaste_siglip2_data")
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
(OUT / "run_complete.json").unlink(missing_ok=True)
STARTED = datetime.now(timezone.utc).isoformat()
SESSION_T0 = time.perf_counter()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
ENVIRONMENT = {
    "started_utc": STARTED, "python": sys.version, "platform": platform.platform(),
    "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
    "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    "packages": {p: importlib.metadata.version(p) for p in
                 ["torch", "torchvision", "timm", "scikit-learn", "numpy", "pandas", "Pillow"]},
}
write_json(OUT / "environment.json", ENVIRONMENT)
write_json(OUT / "config.json", {"experiment": CFG, "models": MODEL_SPECS})
(OUT / "requirements-lock.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
display(pd.Series(ENVIRONMENT).to_frame("value"))

# %% [markdown]
# ## Data and integrity audit
#
# RealWaste contains 4,752 images at 524 × 524 pixels, across nine classes,
# collected in an Australian landfill environment. The notebook downloads the
# original UCI archive or uses a matching RealWaste folder attached under
# `/kaggle/input`. It requires the documented class counts to prevent silently
# evaluating a different dataset.
#
# Decoded RGB pixels are hashed before splitting. Exact duplicates are removed;
# conflicting labels for identical images stop the run. This does **not** detect
# near-duplicates or identify acquisition sessions. The dataset does not provide
# capture-group IDs, so these are stratified image-level splits, not a claim of
# independence between scenes or generalization to other landfill sites.

# %%
def find_realwaste(base):
    if not base.exists():
        return None
    for cardboard in sorted(base.rglob("Cardboard")):
        root = cardboard.parent
        if cardboard.is_dir() and all((root / name).is_dir() for name in CLASSES):
            return root
    return None

root = find_realwaste(Path("/kaggle/input")) or find_realwaste(DATA)
archive_sha256 = None
if root is None:
    archive = DATA / "realwaste.zip"
    for attempt in range(3):
        try:
            with requests.get(CFG["dataset_url"], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with archive.open("wb") as handle, tqdm(total=total or None, unit="B", unit_scale=True,
                                                       desc="Downloading official RealWaste") as bar:
                    for chunk in response.iter_content(2**20):
                        if chunk:
                            handle.write(chunk)
                            bar.update(len(chunk))
            with zipfile.ZipFile(archive) as zf:
                # Do not extract paths outside the data directory.
                for member in zf.infolist():
                    target = (DATA / member.filename).resolve()
                    if not target.is_relative_to(DATA.resolve()):
                        raise ValueError(f"Unsafe archive path: {member.filename}")
                zf.extractall(DATA)
            break
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            archive.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError("Download failed. Enable Internet or attach the original RealWaste dataset.") from exc
            time.sleep(2 ** attempt)
    with archive.open("rb") as handle:
        archive_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    archive.unlink()  # Raw images stay in /kaggle/temp, outside saved notebook outputs.
    root = find_realwaste(DATA)
assert root is not None, "No valid nine-class RealWaste folder found."

records, unreadable = [], []
for class_name in CLASSES:
    for path in sorted((root / class_name).iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                pixel_hash = hashlib.sha256(
                    str(image.size).encode() + image.tobytes()).hexdigest()
                width, height = image.size
            records.append({"path": str(path), "relative_path": str(path.relative_to(root)),
                            "class_name": class_name, "label": CLASS_TO_IDX[class_name],
                            "pixel_sha256": pixel_hash, "width": width, "height": height})
        except (OSError, ValueError) as exc:
            unreadable.append({"path": str(path), "error": str(exc)})
write_json(OUT / "unreadable_images.json", unreadable)
assert not unreadable, "Unreadable images found; see unreadable_images.json. Fix the source before evaluation."
raw = pd.DataFrame(records)
assert len(raw), "No images found."
actual_counts = raw.groupby("class_name").size().to_dict()
assert actual_counts == EXPECTED_COUNTS, f"Dataset version mismatch: {actual_counts}"
conflicts = raw.groupby("pixel_sha256").label.nunique()
assert conflicts.max() == 1, "Identical pixels have conflicting labels; resolve these before training."
duplicates = raw[raw.duplicated("pixel_sha256", keep=False)].copy()
duplicates.to_csv(OUT / "exact_duplicate_groups.csv", index=False)
manifest = raw.drop_duplicates("pixel_sha256", keep="first").reset_index(drop=True)
manifest["image_id"] = np.arange(len(manifest))
manifest.to_csv(OUT / "manifest.csv", index=False)
MANIFEST_HASH = stable_hash(manifest[["relative_path", "label", "pixel_sha256"]].to_dict("records"))
AUDIT = {"source_url": CFG["dataset_url"], "archive_sha256": archive_sha256,
         "manifest_sha256": MANIFEST_HASH, "raw_images": len(raw), "unique_images": len(manifest),
         "removed_exact_duplicates": len(raw) - len(manifest), "unreadable_images": len(unreadable)}
write_json(OUT / "data_audit.json", AUDIT)
y = manifest.label.to_numpy()
display(pd.DataFrame({"original": pd.Series(actual_counts),
                      "after_exact_deduplication": manifest.groupby("class_name").size()}))
print(AUDIT)

# %%
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, class_name in zip(axes.flat, CLASSES):
    row = manifest[manifest.class_name == class_name].iloc[0]
    with Image.open(row.path) as image:
        ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
    ax.set_title(class_name)
    ax.axis("off")
fig.suptitle("RealWaste: one example per class", fontsize=16)
fig.tight_layout()
fig.savefig(OUT / "class_examples.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# ## Protocol fixed before results
#
# For each seed (42, 123, 2026), create a 70% train / 15% validation / 15% test
# stratified split after exact deduplication. Rounding may change counts slightly.
# Both models use exactly the same image IDs and label subsets. Their backbones
# remain frozen. At 224 × 224, each checkpoint uses its own documented normalization
# and crop settings; the ResNet evaluation size is explicitly overridden to 224
# for a shared resolution. The comparison is between pretrained systems, not an
# architecture-only experiment: pretraining data, model sizes, and transforms differ.
#
# Extract deterministic features once, L2-normalize per image, then fit a class-balanced
# multinomial logistic regression. No normalization statistics are fitted on test data.
# Select C from {0.01, 0.1, 1, 10} using **validation macro-F1 only**. Test labels are
# used only after that selection. Report accuracy, balanced accuracy, macro-F1,
# weighted-F1, per-class precision/recall/F1, and confusion matrices.
#
# Repeat with 10%, 25%, and 100% of the training labels; validation and test remain
# fixed. These are fractions of the training partition, and the full validation
# labels are still required for tuning. This is not a strict few-shot benchmark.
# Subsets are nested and identical for both encoders. The 100% condition is primary.
#
# Mean ± sample standard deviation across three splits describes split sensitivity.
# Since the repeated test splits overlap, the three scores are not independent
# replications. The first seed is preselected for plots and example errors.

# %%
SPLITS, SUBSETS = {}, {}
split_rows = []
assert np.isclose(sum(CFG[k] for k in ("train_fraction", "validation_fraction", "test_fraction")), 1.0)
heldout_fraction = CFG["validation_fraction"] + CFG["test_fraction"]
for seed in CFG["seeds"]:
    all_ids = np.arange(len(manifest))
    train_ids, heldout = train_test_split(all_ids, test_size=heldout_fraction, stratify=y, random_state=seed)
    val_ids, test_ids = train_test_split(heldout, test_size=CFG["test_fraction"] / heldout_fraction,
                                       stratify=y[heldout], random_state=seed)
    train_ids, val_ids, test_ids = [np.sort(ids) for ids in (train_ids, val_ids, test_ids)]
    assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids))
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(manifest)
    SPLITS[seed] = {"train": train_ids, "validation": val_ids, "test": test_ids}
    for split_name, ids in SPLITS[seed].items():
        assert np.array_equal(np.unique(y[ids]), LABELS)
        part = manifest.iloc[ids][["image_id", "relative_path", "pixel_sha256", "class_name", "label"]].copy()
        part["seed"], part["split"] = seed, split_name
        split_rows.append(part)
    # A single seeded per-class permutation makes the smaller subsets nested.
    rng = np.random.default_rng(seed)
    per_class = {label: rng.permutation(train_ids[y[train_ids] == label]) for label in LABELS}
    for fraction in CFG["label_fractions"]:
        subset = np.sort(np.concatenate([ids[:max(1, int(np.ceil(len(ids) * fraction)))]
                                        for ids in per_class.values()]))
        SUBSETS[(seed, fraction)] = subset
split_manifest = pd.concat(split_rows, ignore_index=True)
split_manifest.to_csv(OUT / "split_manifest.csv", index=False)
write_json(OUT / "training_subsets.json", {
    f"seed_{seed}_fraction_{fraction}": ids.tolist() for (seed, fraction), ids in SUBSETS.items()})
display(split_manifest.groupby(["seed", "split"]).size().unstack())

# %% [markdown]
# ## Frozen feature extraction (the only substantial GPU work)
#
# The SigLIP 2 **image tower** is used here; there is no text prompting or zero-shot
# claim. Checkpoints are pinned to Hugging Face commit hashes. AMP uses float16 on
# CUDA, then features are converted to float32. Cache identity includes the decoded
# data manifest, checkpoint revision, preprocessing, library versions, and GPU type.
# An interrupted run can reuse complete cached features within the same session.

# %%
class RealWasteImages(Dataset):
    def __init__(self, table, transform):
        self.paths = table.path.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(ImageOps.exif_transpose(image).convert("RGB"))

def load_encoder(name):
    spec = MODEL_SPECS[name]
    model = timm.create_model(f"hf-hub:{spec['repo']}@{spec['revision']}",
                              pretrained=True, num_classes=0)
    model.eval().requires_grad_(False)
    data_config = timm.data.resolve_model_data_config(model)
    data_config["input_size"] = (3, CFG["image_size"], CFG["image_size"])
    transform = timm.data.create_transform(**data_config, is_training=False)
    return model.to(DEVICE), transform, data_config

@torch.inference_mode()
def extract_features(name):
    model, transform, data_config = load_encoder(name)
    identity = {"model": MODEL_SPECS[name], "manifest": MANIFEST_HASH,
                "preprocessing": data_config, "amp": CFG["amp"],
                "packages": ENVIRONMENT["packages"], "gpu": ENVIRONMENT["gpu"]}
    identity_hash = stable_hash(identity)
    cache_dir = OUT / "features"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{name}_{identity_hash[:16]}.npz"
    meta_file = cache_file.with_suffix(".json")
    if cache_file.exists() and meta_file.exists():
        with np.load(cache_file, allow_pickle=False) as cache:
            features = cache["features"]
            assert np.array_equal(cache["image_ids"], manifest.image_id.to_numpy())
        metadata = json.loads(meta_file.read_text())
        assert metadata["identity_hash"] == identity_hash
        assert features.shape[0] == len(manifest) and np.isfinite(features).all()
        print(f"Reusing verified {name} features: {features.shape}")
    else:
        loader = DataLoader(RealWasteImages(manifest, transform),
                            batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)
        # Warm up before measuring full extraction (includes data loading and preprocessing).
        dummy = torch.zeros(1, 3, CFG["image_size"], CFG["image_size"], device=DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=CFG["amp"]):
            model(dummy)
        del dummy
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        chunks = []
        for batch in tqdm(loader, desc=f"Frozen {name}"):
            with torch.autocast("cuda", dtype=torch.float16, enabled=CFG["amp"]):
                encoded = model(batch.to(DEVICE, non_blocking=True))
            assert encoded.ndim == 2, f"Expected pooled image features, got {encoded.shape}"
            chunks.append(encoded.float().cpu().numpy())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        features = normalize(np.concatenate(chunks).astype(np.float32), norm="l2", copy=False)
        assert len(features) == len(manifest) and np.isfinite(features).all()
        metadata = {"identity": identity, "identity_hash": identity_hash,
                    "num_images": len(manifest), "feature_dim": features.shape[1],
                    "backbone_parameters": sum(p.numel() for p in model.parameters()),
                    "extraction_seconds": elapsed,
                    "extraction_images_per_second": len(features) / elapsed,
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "timing_note": "Includes image loading, preprocessing, GPU inference, transfers; excludes model download/load and warmup.",
                    "gpu": ENVIRONMENT["gpu"]}
        tmp = cache_file.with_name(cache_file.stem + ".partial.npz")
        np.savez_compressed(tmp, features=features, image_ids=manifest.image_id.to_numpy())
        tmp.replace(cache_file)
        write_json(meta_file, metadata)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return features, metadata

FEATURES, FEATURE_METADATA = {}, {}
for name in MODEL_SPECS:
    FEATURES[name], FEATURE_METADATA[name] = extract_features(name)
write_json(OUT / "feature_extraction.json", FEATURE_METADATA)
display(pd.DataFrame(FEATURE_METADATA).T[["feature_dim", "backbone_parameters", "extraction_seconds",
                                        "extraction_images_per_second", "peak_gpu_memory_gib"]])

# %%
def metrics_for(truth, prediction):
    return {"accuracy": accuracy_score(truth, prediction),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "macro_f1": f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth, prediction, labels=LABELS, average="weighted", zero_division=0)}

def fit_probe(X, train_ids, val_ids):
    candidates, best_model, best_score, best_C = [], None, -np.inf, None
    for C in CFG["C_grid"]:
        classifier = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                                        max_iter=CFG["max_iter"], tol=1e-4)
        # A nonconverged fit must not silently enter the results table.
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(X[train_ids], y[train_ids])
        score = f1_score(y[val_ids], classifier.predict(X[val_ids]),
                         labels=LABELS, average="macro", zero_division=0)
        candidates.append({"C": C, "validation_macro_f1": float(score)})
        # Fixed grid order breaks ties in favor of stronger regularization (smaller C).
        if score > best_score:
            best_model, best_score, best_C = classifier, score, C
    return best_model, best_C, best_score, candidates

metrics_rows, tuning_rows, prediction_rows, class_rows = [], [], [], []
PROBES = {}
for seed in CFG["seeds"]:
    val_ids, test_ids = SPLITS[seed]["validation"], SPLITS[seed]["test"]
    for fraction in CFG["label_fractions"]:
        train_ids = SUBSETS[(seed, fraction)]
        for name, X in FEATURES.items():
            t0 = time.perf_counter()
            classifier, best_C, val_f1, candidates = fit_probe(X, train_ids, val_ids)
            fit_seconds = time.perf_counter() - t0
            prediction = classifier.predict(X[test_ids])
            probabilities = classifier.predict_proba(X[test_ids])
            scores = metrics_for(y[test_ids], prediction)
            key = {"model": name, "seed": seed, "label_fraction": fraction}
            metrics_rows.append({**key, "n_train": len(train_ids), "n_val": len(val_ids),
                                 "n_test": len(test_ids), "best_C": best_C,
                                 "validation_macro_f1": val_f1, "probe_selection_seconds": fit_seconds,
                                 **scores})
            tuning_rows.extend({**key, **candidate} for candidate in candidates)
            part = manifest.iloc[test_ids][["image_id", "relative_path", "class_name", "label"]].copy()
            for k, v in key.items():
                part[k] = v
            part["prediction"] = prediction
            part["predicted_class"] = [CLASSES[i] for i in prediction]
            part["confidence"] = probabilities.max(axis=1)
            prediction_rows.append(part)
            report = classification_report(y[test_ids], prediction, labels=LABELS,
                                           target_names=CLASSES, output_dict=True, zero_division=0)
            class_rows.extend({**key, "class_name": c, **report[c]} for c in CLASSES)
            PROBES[(name, seed, fraction)] = classifier
            print(f"{name:8s} seed={seed} labels={fraction:.0%} C={best_C:g} "
                  f"val F1={val_f1:.4f} test F1={scores['macro_f1']:.4f}")
            # Persist after every successful condition, so partial runs retain useful evidence.
            pd.DataFrame(metrics_rows).to_csv(OUT / "metrics_per_run.csv", index=False)
            pd.DataFrame(tuning_rows).to_csv(OUT / "validation_search.csv", index=False)
    train_ids = SPLITS[seed]["train"]
    majority_label = int(np.bincount(y[train_ids], minlength=len(CLASSES)).argmax())
    metrics_rows.append({"model": "majority", "seed": seed, "label_fraction": 1.0,
                         "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
                         **metrics_for(y[test_ids], np.full(len(test_ids), majority_label))})

metrics = pd.DataFrame(metrics_rows)
predictions = pd.concat(prediction_rows, ignore_index=True)
per_class = pd.DataFrame(class_rows)
metrics.to_csv(OUT / "metrics_per_run.csv", index=False)
predictions.to_csv(OUT / "test_predictions.csv", index=False)
per_class.to_csv(OUT / "per_class_metrics.csv", index=False)
METRIC_NAMES = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
summary = metrics.groupby(["model", "label_fraction"])[METRIC_NAMES].agg(["mean", "std"]).reset_index()
summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
summary.to_csv(OUT / "metrics_summary.csv", index=False)
display(summary)

# %% [markdown]
# ## Results and error analysis
#
# All values in the following cells come from the held-out predictions above.
# Scores are proportions in CSV files and percentages in the written report.
# Confidence is the classifier's softmax probability; it is not calibrated.
# Error examples are descriptive, and no test-based tuning is performed afterward.

# %%
FIRST_SEED = CFG["seeds"][0]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, name in zip(axes, MODEL_SPECS):
    part = predictions[(predictions.model == name) & (predictions.seed == FIRST_SEED)
                       & (predictions.label_fraction == 1.0)]
    matrix = confusion_matrix(part.label, part.prediction, labels=LABELS)
    row_normalized = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(row_normalized, cmap="Blues", vmin=0, vmax=1)
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if row_normalized[i, j] > 0.55 else "black")
    ax.set_xticks(LABELS, CLASSES, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(LABELS, CLASSES, fontsize=8)
    ax.set_xlabel("Predicted material")
    ax.set_ylabel("True material")
    ax.set_title(f"{name} — seed {FIRST_SEED}; counts, row-normalized color")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", dpi=170, bbox_inches="tight")
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for name in MODEL_SPECS:
    part = summary[summary.model == name].sort_values("label_fraction")
    ax.errorbar(part.label_fraction * 100, part.macro_f1_mean * 100,
                yerr=part.macro_f1_std * 100, marker="o", capsize=4, label=name)
ax.set(xlabel="Training partition labels used (%)", ylabel="Test macro-F1 (%)",
       title="Label efficiency; mean ± sample SD across three splits")
ax.set_xticks([10, 25, 100])
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "label_efficiency.png", dpi=170)
plt.show()
plt.close(fig)

errors = predictions[(predictions.model == "siglip2") & (predictions.seed == FIRST_SEED)
                     & (predictions.label_fraction == 1.0)
                     & (predictions.label != predictions.prediction)].sort_values("confidence", ascending=False)
errors.head(20).to_csv(OUT / "siglip2_errors_seed42.csv", index=False)
if len(errors):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.head(6).itertuples()):
        with Image.open(manifest.iloc[row.image_id].path) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title(f"True: {row.class_name}\nPredicted: {row.predicted_class} ({row.confidence:.2f})", fontsize=9)
    fig.suptitle("SigLIP 2: highest-confidence errors, preselected seed 42")
    fig.tight_layout()
    fig.savefig(OUT / "error_examples.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %%
# Descriptive paired bootstrap for the preselected first split only.
# This interval reflects resampling test images, not model-training or site uncertainty.
first_test_ids = SPLITS[FIRST_SEED]["test"]
truth = y[first_test_ids]
first_predictions = {name: PROBES[(name, FIRST_SEED, 1.0)].predict(FEATURES[name][first_test_ids])
                     for name in MODEL_SPECS}
rng = np.random.default_rng(2026)
bootstrap_deltas = []
for _ in range(1000):
    ids = rng.integers(0, len(truth), size=len(truth))
    bootstrap_deltas.append(
        f1_score(truth[ids], first_predictions["siglip2"][ids], labels=LABELS,
                 average="macro", zero_division=0)
        - f1_score(truth[ids], first_predictions["resnet18"][ids], labels=LABELS,
                   average="macro", zero_division=0))
BOOTSTRAP = {"seed": FIRST_SEED, "resamples": 1000, "bootstrap_rng_seed": 2026,
             "delta_macro_f1_95_percentile_interval": np.quantile(bootstrap_deltas, [0.025, 0.975]).tolist(),
             "limitation": "Conditional on one split and fitted models; image-level resampling assumes independent test images."}
write_json(OUT / "paired_bootstrap.json", BOOTSTRAP)
print(BOOTSTRAP)

# %% [markdown]
# ## Save the classifier and reproduce single-image inference
#
# The saved classifier is from the **preselected first seed, 100% training labels**,
# not whichever seed scored highest. It was not refit on validation or test images.
# To use it later, download the pinned pretrained encoder, load this trusted local
# joblib artifact, apply the saved preprocessing, L2-normalize its image embedding,
# and call `classifier.predict`. Do not load untrusted joblib files.

# %%
bundle = {
    "classifier": PROBES[("siglip2", FIRST_SEED, 1.0)], "classes": CLASSES,
    "model_spec": MODEL_SPECS["siglip2"],
    "preprocessing": FEATURE_METADATA["siglip2"]["identity"]["preprocessing"],
    "feature_normalization": "l2", "manifest_sha256": MANIFEST_HASH,
    "seed": FIRST_SEED, "label_fraction": 1.0,
}
joblib.dump(bundle, OUT / "siglip2_linear_probe_seed42.joblib")

@torch.inference_mode()
def predict_image(image_path, saved_bundle, encoder):
    transform = timm.data.create_transform(**saved_bundle["preprocessing"], is_training=False)
    with Image.open(image_path) as image:
        tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", dtype=torch.float16, enabled=CFG["amp"]):
        features = encoder(tensor).float().cpu().numpy()
    features = normalize(features, norm="l2")
    probs = saved_bundle["classifier"].predict_proba(features)[0]
    predicted_label = int(saved_bundle["classifier"].classes_[probs.argmax()])
    return {"predicted_class": saved_bundle["classes"][predicted_label],
            "uncalibrated_probability": float(probs.max())}

encoder, _, _ = load_encoder("siglip2")
example_id = int(first_test_ids[0])
inference_example = predict_image(manifest.iloc[example_id].path, bundle, encoder)
write_json(OUT / "inference_example.json", {"image_id": example_id, **inference_example})
print(inference_example)
del encoder
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## Generate the experiment report and downloadable artifacts
#
# The report describes measured outcomes even if SigLIP 2 loses. It does not compare
# our scores directly against published scores obtained using different splits.
# Publication-ready claims require a broader literature review and, ideally,
# capture-group or external-site evaluation.

# %%
primary = summary[summary.label_fraction == 1.0].set_index("model")
delta = float(primary.loc["siglip2", "macro_f1_mean"] - primary.loc["resnet18", "macro_f1_mean"])
direction = "higher" if delta > 0 else "lower" if delta < 0 else "equal"

def score_cell(row, metric):
    return f"{row[metric + '_mean'] * 100:.2f} ± {row[metric + '_std'] * 100:.2f}"

table = ["| Model | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) |",
         "|---|---:|---:|---:|---:|"]
for name in ["majority", "resnet18", "siglip2"]:
    row = primary.loc[name]
    table.append("| " + " | ".join([name] + [score_cell(row, metric) for metric in METRIC_NAMES]) + " |")
efficiency_table = ["| Model | Training labels (%) | Macro-F1 (%) |", "|---|---:|---:|"]
for row in summary[summary.model != "majority"].to_dict("records"):
    efficiency_table.append(f"| {row['model']} | {100 * row['label_fraction']:.0f} | {score_cell(row, 'macro_f1')} |")
timing_table = ["| Encoder | Backbone parameters | Extraction (s) | Images/s | Peak allocated GPU GiB |",
                "|---|---:|---:|---:|---:|"]
for name, meta in FEATURE_METADATA.items():
    timing_table.append(f"| {name} | {meta['backbone_parameters']:,} | {meta['extraction_seconds']:.1f} | "
                        f"{meta['extraction_images_per_second']:.1f} | {meta['peak_gpu_memory_gib']:.2f} |")
interval = np.asarray(BOOTSTRAP["delta_macro_f1_95_percentile_interval"]) * 100
first_class = per_class[(per_class.model == "siglip2") & (per_class.seed == FIRST_SEED)
                        & (per_class.label_fraction == 1.0)].sort_values("f1-score")
weakest = first_class.iloc[0]
perclass_table = ["| Material | Precision (%) | Recall (%) | F1 (%) | Support |",
                  "|---|---:|---:|---:|---:|"]
for row in first_class.to_dict("records"):
    perclass_table.append(f"| {row['class_name']} | {row['precision'] * 100:.2f} | {row['recall'] * 100:.2f} | "
                          f"{row['f1-score'] * 100:.2f} | {int(row['support'])} |")

report_text = f"""# Frozen SigLIP 2 for authentic landfill material classification

Experiment started: {STARTED}. Report generated: {datetime.now(timezone.utc).isoformat()}.

## Abstract
We evaluate an existing SigLIP 2 ViT-B/16 image encoder on the existing RealWaste
dataset using a frozen encoder and a class-balanced linear classifier. An ImageNet
ResNet-18 and a majority-class rule provide baselines. Under three fixed stratified
image-level splits, SigLIP 2 obtains mean test macro-F1
{primary.loc['siglip2', 'macro_f1_mean'] * 100:.2f}% (sample SD
{primary.loc['siglip2', 'macro_f1_std'] * 100:.2f} percentage points).
Its mean macro-F1 is {abs(delta) * 100:.2f} percentage points {direction} than ResNet-18.
This is an empirical model–dataset application study; no new model architecture is proposed.

## Motivation and contribution
Authentic discarded materials differ from clean product photographs. We ask whether
the transferable representations of a pretrained SigLIP 2 image tower are useful for
the nine-class material task without updating the backbone. The contribution is a
reproducible evaluation of this specific model–dataset pairing, including a baseline,
label-efficiency conditions, per-class analysis, and measured compute costs.
A targeted search on 9 November 2026 found no direct SigLIP 2–RealWaste evaluation.
Search coverage is incomplete: this is not a guaranteed first-ever combination.
Foundation-model waste classification, including DINOv3 on RealWaste, is prior work.

## Data and reproducibility
Source: UCI RealWaste (DOI 10.24432/C5SS4G), {AUDIT['raw_images']} original images;
{AUDIT['unique_images']} after removing {AUDIT['removed_exact_duplicates']} exact
decoded-pixel duplicates. Dataset manifest SHA-256: `{MANIFEST_HASH}`.
Class labels: {', '.join(CLASSES)}.
Images, duplicate groups, split membership and label subsets are recorded in CSV/JSON.
Raw data are not included in this output archive; use the cited source.

## Methods
Frozen encoders: `timm/resnet18.a1_in1k` and
`timm/vit_base_patch16_siglip_224.v2_webli`, pinned to commits in config.json.
Images use 224 × 224 inputs and checkpoint-specific deterministic transforms.
ResNet's default evaluation resolution is explicitly overridden to 224.
Every embedding is L2-normalized independently. No encoder weights are trained.
For each seed {CFG['seeds']}, stratify 70/15/15 into train/validation/test;
select logistic regression C from {CFG['C_grid']} using validation macro-F1 only,
with class_weight='balanced', solver='lbfgs', max_iter={CFG['max_iter']}.
Do not refit on validation or test data. Both encoders share all partitions and
training subsets. Conditions use 10%, 25%, or 100% of training labels, while the
full validation partition remains available. The 100% condition is primary.
Model size, pretraining data, and transforms differ, so this is a system-level
comparison rather than an isolated test of architecture or training objective.

## Primary held-out results
Mean ± sample SD across the three repeated splits:

{chr(10).join(table)}

The SigLIP 2 minus ResNet-18 mean macro-F1 difference is {delta * 100:+.2f}
percentage points. For preselected seed 42, a paired image-level percentile
bootstrap with 1,000 resamples gives a conditional 95% difference interval of
[{interval[0]:+.2f}, {interval[1]:+.2f}] percentage points. This is conditional on
one fitted pair and split and assumes independent test images; it does not
measure training, capture-session, or site uncertainty. The three repeated test
sets overlap, so their SD is descriptive split sensitivity, not an independent-run CI.

## Label-efficiency results
Full validation labels are used in all conditions. Percentages refer to the train
partition; no strict few-shot or zero-shot claim is made.

{chr(10).join(efficiency_table)}

![Label efficiency](label_efficiency.png)

## Error analysis (preselected seed 42, all training labels)
The lowest per-class F1 for SigLIP 2 is {weakest['class_name']}
({weakest['f1-score'] * 100:.2f}%). Confusion matrices report counts with row-normalized
color. The highest-confidence errors, when present, appear in error_examples.png;
classifier probabilities are uncalibrated. These examples describe observed failures
and do not establish their causes.

{chr(10).join(perclass_table)}

![Confusion matrices](confusion_matrices.png)

## Compute
Hardware: {ENVIRONMENT['gpu']} ({ENVIRONMENT['gpu_memory_gib']:.1f} GiB device memory).
One CUDA device; batch size {CFG['batch_size']}; float16 autocast for encoder inference.
Full feature extraction includes data loading, preprocessing and transfers, but excludes
checkpoint downloading/loading and warmup. Peak memory is PyTorch allocated memory,
not all driver allocations. Classifier fitting is on the CPU. Cached timing values
refer to the original extraction recorded in each cache's metadata.

{chr(10).join(timing_table)}

Elapsed notebook work after package setup: {time.perf_counter() - SESSION_T0:.1f} s.
Package versions and the full environment lock are saved with the results.

## Limitations and interpretation
- Exact deduplication does not eliminate near-duplicates or scene correlations.
- No capture-group labels or independent landfill-site test set are available here.
- The pretrained WebLI dataset is not fully inspectable for benchmark overlap;
  contamination cannot be ruled out. ImageNet and WebLI pretraining differ substantially.
- The experiment tests frozen transfer. It does not establish optimal fine-tuned performance.
- Three overlapping image-level splits provide limited uncertainty assessment.
- Single-label material classification does not detect multiple objects, identify every
  material in an image, determine local recyclability, or detect unknown classes.
- Published RealWaste scores with other splits/protocols are background context only.

## Conclusion
Under this fixed evaluation protocol, SigLIP 2 has {direction} mean macro-F1 than
the frozen ResNet-18 baseline by {abs(delta) * 100:.2f} percentage points.
This measured outcome supports a scoped discussion of the selected model–dataset
application, with the generalization and novelty limitations above.

## References and source terms
1. Tschannen et al. (2025), SigLIP 2. https://arxiv.org/abs/2502.14786
2. SigLIP 2 image encoder and Apache-2.0 model card:
   https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli
3. Single, Iranmanesh & Raad (2023), RealWaste dataset, UCI:
   https://doi.org/10.24432/C5SS4G
4. Single et al. (2023), RealWaste: A Novel Real-Life Data Set for Landfill Waste
   Classification Using Deep Learning. https://doi.org/10.3390/info14120633
5. ResNet-18 checkpoint: https://huggingface.co/timm/resnet18.a1_in1k
6. Funk et al. (2026), Evaluation of vision–language models for waste classification:
   Zero- and few-shot with a training-free tip-adapter. https://doi.org/10.1016/j.clwas.2026.100475

UCI lists CC BY 4.0 for RealWaste. The authors' GitHub README lists CC BY-NC-SA 4.0
(https://github.com/sam-single/realwaste). Preserve attribution and verify applicable
terms before redistributing the images; this project downloads them for an academic study.
"""
(OUT / "experiment_report.md").write_text(report_text, encoding="utf-8")
display(Markdown(report_text))

# %%
expected_conditions = len(CFG["seeds"]) * (len(MODEL_SPECS) * len(CFG["label_fractions"]) + 1)
assert len(metrics) == expected_conditions
assert metrics[METRIC_NAMES].notna().all().all()
assert ((metrics[METRIC_NAMES] >= 0) & (metrics[METRIC_NAMES] <= 1)).all().all()
write_json(OUT / "run_complete.json", {
    "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
    "conditions": len(metrics), "manifest_sha256": MANIFEST_HASH,
    "config_sha256": stable_hash({"experiment": CFG, "models": MODEL_SPECS}),
})
# This archive contains results/features and a small classifier, not raw images or encoder weights.
archive_path = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
print("Experiment complete. Save a Kaggle version to retain these outputs.")
display(FileLink(str(OUT / "experiment_report.md")))
display(FileLink(archive_path))

# %% [markdown]
# To retain the run, use **Save Version → Save & Run All** (or save the completed
# interactive session using Kaggle's available output-saving option). Download
# `realwaste_siglip2.zip` from the Output files pane. The archive contains the
# experiment report, metrics, test predictions, data/split manifests, feature caches,
# plots, environment details, and the seed-42 SigLIP 2 classifier.
