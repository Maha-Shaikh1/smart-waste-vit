"""Shared helpers for the Smart Waste ViT project (same logic as the notebook)."""
import hashlib
import io
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_ID = "huaweilin/waste-classification"
MODEL_ID = "google/vit-base-patch16-224"
SEED = 42
HAZ_NAME = "Hazardous"

PALETTE = {"forest": "#1F4D3A", "mint": "#9FD9B8", "cream": "#F7F3EA", "charcoal": "#2E2E2E", "orange": "#D97B29"}
CLASS_COLOURS = [PALETTE["orange"], PALETTE["charcoal"], PALETTE["mint"], PALETTE["forest"]]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def thumb_hash(img: Image.Image) -> str:
    small = img.convert("L").resize((64, 64), Image.BILINEAR)
    return hashlib.md5(small.tobytes()).hexdigest()


def clean_dataset(raw):
    """Drop corrupted images and exact duplicates. Returns (dataset, stats)."""
    keep, seen, n_corrupt, n_dup = [], set(), 0, 0
    for i in range(len(raw)):
        try:
            img = raw[i]["image"].convert("RGB")
        except Exception:
            n_corrupt += 1
            continue
        h = thumb_hash(img)
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        keep.append(i)
    return raw.select(keep), {"total_raw": len(raw), "corrupted": n_corrupt, "duplicates": n_dup, "kept": len(keep)}


def make_splits(ds, seed: int = SEED):
    """Stratified 70/15/15 split by subclass. Returns (train_idx, val_idx, test_idx)."""
    sub = np.array(ds["subclass"])
    cls = np.array(ds["class"])
    key = sub.copy()
    vc = pd.Series(key).value_counts()
    for r in vc[vc < 7].index:
        key[key == r] = "RARE_" + cls[key == r]
    idx = np.arange(len(ds))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=key)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=key[tmp])
    return tr, va, te


def build_transforms(processor):
    _sz = processor.size  # dict, SizeDict or int depending on transformers version
    size = int(_sz["height"] if isinstance(_sz, dict) else getattr(_sz, "height", None) or _sz)
    mean, std = list(processor.image_mean), list(processor.image_std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class WasteDataset(Dataset):
    def __init__(self, hf_ds, tf, label2id, extra_tf=None):
        self.ds, self.tf, self.label2id, self.extra_tf = hf_ds, tf, label2id, extra_tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        if self.extra_tf is not None:
            img = self.extra_tf(img)
        return {"pixel_values": self.tf(img), "labels": self.label2id[ex["class"]]}


def collate_fn(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch])}


# ---- degradations for the image-quality experiment ----
def blur_tf(img):
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def dark_tf(img):
    return ImageEnhance.Brightness(img).enhance(0.5)


def jpeg_tf(img, quality=20):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CONDITIONS = {"Original": None, "Blur (r=2)": blur_tf, "Dark (x0.5)": dark_tf, "JPEG (q=20)": jpeg_tf}


def summarise(preds, labels, haz_id, name="Original"):
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    haz_mask = labels == haz_id
    haz_recall = float((preds[haz_mask] == haz_id).mean()) if haz_mask.sum() else float("nan")
    return {
        "condition": name, "accuracy": float(acc), "macro_precision": float(p), "macro_recall": float(r),
        "macro_f1": float(f1), "hazardous_recall": haz_recall, "hazardous_fn_rate": 1 - haz_recall,
        "hazardous_false_negatives": int(((labels == haz_id) & (preds != haz_id)).sum()),
        "hazardous_false_positives": int(((labels != haz_id) & (preds == haz_id)).sum()),
        "n_hazardous": int(haz_mask.sum()),
    }


def load_labels(model_dir: str):
    with open(os.path.join(model_dir, "labels.json")) as f:
        d = json.load(f)
    return {int(k): v for k, v in d["id2label"].items()}, d["label2id"], d["class_names"]
