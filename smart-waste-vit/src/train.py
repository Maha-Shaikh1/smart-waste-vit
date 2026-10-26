"""Fine-tune google/vit-base-patch16-224 on huaweilin/waste-classification.

Usage:  python src/train.py [--epochs 5] [--batch 32] [--lr 5e-5] [--out saved_model]
Saves: saved_model/ (weights, processor, labels.json) and results/split_indices.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

from common import (DATASET_ID, HAZ_NAME, MODEL_ID, SEED, WasteDataset, build_transforms, clean_dataset,
                    collate_fn, make_splits, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="saved_model")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    set_seed()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    raw = load_dataset(DATASET_ID, split="cleaned")
    ds, stats = clean_dataset(raw)
    print("Cleaning:", stats)
    json.dump(stats, open(os.path.join(a.results, "data_cleaning.json"), "w"), indent=2)

    class_names = sorted(set(ds["class"]))
    label2id = {c: i for i, c in enumerate(class_names)}
    id2label = {i: c for c, i in label2id.items()}
    haz_id = label2id[HAZ_NAME]

    tr, va, te = make_splits(ds)
    json.dump({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
              open(os.path.join(a.results, "split_indices.json"), "w"))
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    train_tf, eval_tf = build_transforms(processor)
    train_ds = WasteDataset(ds.select(tr), train_tf, label2id)
    val_ds = WasteDataset(ds.select(va), eval_tf, label2id)

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID, num_labels=len(class_names), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        haz_recall = ((preds == haz_id) & (labels == haz_id)).sum() / max((labels == haz_id).sum(), 1)
        return {"accuracy": accuracy_score(labels, preds), "macro_precision": p, "macro_recall": r,
                "macro_f1": f1, "hazardous_recall": haz_recall}

    import inspect
    steps_per_epoch = int(np.ceil(len(train_ds) / a.batch))
    kw = dict(output_dir="checkpoints", per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
              num_train_epochs=a.epochs, learning_rate=a.lr, weight_decay=0.01,
              warmup_steps=int(0.1 * steps_per_epoch * a.epochs), lr_scheduler_type="cosine",
              logging_strategy="epoch", eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
              load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
              fp16=(device == "cuda"), dataloader_num_workers=2, remove_unused_columns=False, report_to="none", seed=SEED)
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    kw = {k: v for k, v in kw.items() if k in accepted}
    args = TrainingArguments(**kw)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate_fn, compute_metrics=compute_metrics)
    t0 = time.time()
    trainer.train()
    print(f"Training time: {(time.time() - t0) / 60:.1f} min")

    trainer.save_model(a.out)
    processor.save_pretrained(a.out)
    json.dump({"id2label": id2label, "label2id": label2id, "class_names": class_names},
              open(os.path.join(a.out, "labels.json"), "w"), indent=2)
    json.dump(trainer.state.log_history, open(os.path.join(a.results, "log_history.json"), "w"), indent=2, default=float)
    print("Saved model to", a.out)


if __name__ == "__main__":
    main()
