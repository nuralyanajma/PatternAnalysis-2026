import argparse
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
 
from dataset import CLASS_NAMES, HipMRI3DDataset, load_splits
from modules import build_model
from utils import dice_iou_per_class, slice_thirds_dice, third_slice_indices
 
PROSTATE = CLASS_NAMES.index("prostate")
THIRD_NAMES = ["third 1 (one end)", "third 2 (middle)", "third 3 (other end)"]
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=False, help="unused if --split-file holds full paths")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-file", required=True, help="splits.json written by train.py")
    p.add_argument("--out-dir", default="predictions")
    p.add_argument("--n-worst", type=int, default=4)
    p.add_argument("--slice-axis", type=int, default=2)
    return p.parse_args()
 
 
def take_slice(vol, idx, axis):
    """Select one 2D slice from a 3D tensor along `axis` and return a NumPy array."""
    return torch.select(vol, axis, idx).cpu().numpy()
 
 
def overlay_error(img2d, gt2d, pred2d):
    """RGB image: grey MRI with FP (red) and FN (blue) pixels for the prostate label."""
    lo, hi = img2d.min(), img2d.max()
    grey = (img2d - lo) / (hi - lo + 1e-8)
    rgb = np.stack([grey] * 3, axis=-1)
    fp = (pred2d == PROSTATE) & (gt2d != PROSTATE)
    fn = (gt2d == PROSTATE) & (pred2d != PROSTATE)
    rgb[fp] = [1.0, 0.0, 0.0]
    rgb[fn] = [0.0, 0.3, 1.0]
    return rgb