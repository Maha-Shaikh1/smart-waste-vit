"""Evaluate the saved model on the held-out test split and run the image-quality experiment.

Usage:  python src/evaluate.py [--model saved_model] [--results results]
Requires results/split_indices.json produced by train.py (same seed / same cleaning).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

from common import (CONDITIONS, DATASET_ID, HAZ_NAME, PALETTE, WasteDataset, build_transforms, clean_dataset,
                    collate_fn, load_labels, summarise)


def predict(model, torch_ds, device, bs=64):
    model.eval()
    loader = DataLoader(torch_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    logits, labels = [], []
    with torch.no_grad():
        for b in loader:
            logits.append(model(pixel_values=b["pixel_values"].to(device)).logits.float().cpu())
            labels.append(b["labels"])
    logits = torch.cat(logits)
    probs = torch.softmax(logits, -1).numpy()
    return probs.argmax(-1), torch.cat(labels).numpy(), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saved_model")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    fig_dir = os.path.join(a.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    id2label, label2id, class_names = load_labels(a.model)
    haz_id = label2id[HAZ_NAME]

    raw = load_dataset(DATASET_ID, split="cleaned")
    ds, _ = clean_dataset(raw)
    split = json.load(open(os.path.join(a.results, "split_indices.json")))
    test_ds = ds.select(split["test"])

    processor = ViTImageProcessor.from_pretrained(a.model)
    _, eval_tf = build_transforms(processor)
    model = ViTForImageClassification.from_pretrained(a.model).to(device)

    preds, labels, probs = predict(model, WasteDataset(test_ds, eval_tf, label2id), device)
    base = summarise(preds, labels, haz_id, "Original")
    print(json.dumps(base, indent=2))
    print(classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0))
    json.dump(base, open(os.path.join(a.results, "test_metrics_original.json"), "w"), indent=2)

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(a.results, "confusion_matrix.csv"))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(PALETTE["forest"], as_cmap=True), cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=200); plt.close()

    # hazardous subclass analysis
    sub = np.array(test_ds["subclass"])
    df = pd.DataFrame({"true": [id2label[i] for i in labels], "pred": [id2label[i] for i in preds],
                       "subclass": sub, "confidence": probs.max(-1)})
    df["correct"] = df["true"] == df["pred"]
    df.to_csv(os.path.join(a.results, "test_predictions.csv"), index=False)
    haz = df[df["true"] == HAZ_NAME]
    haz_sub = haz.groupby("subclass")["correct"].agg(["sum", "count"])
    haz_sub["recall"] = haz_sub["sum"] / haz_sub["count"]
    print("\nHazardous recall per subclass:\n", haz_sub)
    haz_sub.to_csv(os.path.join(a.results, "hazardous_subclass_recall.csv"))
    print("\nMissed hazardous items:\n", haz[~haz["correct"]].groupby(["subclass", "pred"]).size())

    # robustness
    rows = []
    for name, fn in CONDITIONS.items():
        p = preds if fn is None else predict(model, WasteDataset(test_ds, eval_tf, label2id, extra_tf=fn), device)[0]
        rows.append(summarise(p, labels, haz_id, name))
    rob = pd.DataFrame(rows).set_index("condition")
    print("\nRobustness:\n", rob[["accuracy", "macro_f1", "hazardous_recall"]].round(4))
    rob.to_csv(os.path.join(a.results, "robustness_results.csv"))


if __name__ == "__main__":
    main()
