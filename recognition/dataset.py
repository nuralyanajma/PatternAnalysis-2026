import glob
import json
import os
import random
import re
 
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
 
PATIENT_RE = re.compile(r"(Case_\d+)", re.IGNORECASE)
WEEK_RE = re.compile(r"(Week\d+)", re.IGNORECASE)
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]

def _key(path):
    name = os.path.basename(path)
    p = PATIENT_RE.search(name)
    w = WEEK_RE.search(name)
    stem = name.split(".")[0]
    return (p.group(1) if p else stem, w.group(1) if w else "")
 
 
def find_pairs(root):
    img_files = sorted(glob.glob(os.path.join(root, "semantic_MRs", "*.nii*")))
    lab_files = sorted(glob.glob(os.path.join(root, "semantic_labels_only", "*.nii*")))
    if not img_files or not lab_files:
        raise FileNotFoundError(f"No NIfTI files found under {root} "
                                "(expected semantic_MRs/ and semantic_labels_only/)")
    labels = {_key(f): f for f in lab_files}
    pairs, unmatched = [], []
    for f in img_files:
        k = _key(f)
        if k in labels:
            pairs.append({"patient": k[0], "week": k[1], "image": f, "label": labels[k]})
        else:
            unmatched.append(f)
    if unmatched:
        print(f"WARNING: {len(unmatched)} image(s) had no matching label, e.g. {unmatched[0]}")
    return pairs
 
 
def split_by_patient(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    patients = sorted({p["patient"] for p in pairs})
    random.Random(seed).shuffle(patients)
    n = len(patients)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])
    train_p = set(patients[n_test + n_val:])
    splits = {
        "train": [p for p in pairs if p["patient"] in train_p],
        "val": [p for p in pairs if p["patient"] in val_p],
        "test": [p for p in pairs if p["patient"] in test_p],
    }
    check_no_leakage(splits)
    return splits
 
 
def check_no_leakage(splits):
    sets = {k: {p["patient"] for p in v} for k, v in splits.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                raise AssertionError(f"Patient leakage between {names[i]} and {names[j]}: {sorted(overlap)}")
 
 
def save_splits(splits, path):
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
 
 
def load_splits(path):
    with open(path) as f:
        splits = json.load(f)
    check_no_leakage(splits)
    return splits
 
 
def describe_splits(splits):
    for k, v in splits.items():
        print(f"{k:>5}: {len({p['patient'] for p in v}):3d} patients, {len(v):3d} volumes")