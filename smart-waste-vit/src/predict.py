"""Classify a single image with the fine-tuned model.

Usage:  python src/predict.py path/to/photo.jpg [--model saved_model]
"""
import argparse
import sys

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from common import build_transforms, load_labels


def predict_image(path: str, model_dir: str = "saved_model"):
    id2label, _, _ = load_labels(model_dir)
    processor = ViTImageProcessor.from_pretrained(model_dir)
    _, eval_tf = build_transforms(processor)
    model = ViTForImageClassification.from_pretrained(model_dir).eval()
    x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(pixel_values=x).logits.float(), -1)[0].numpy()
    top = int(probs.argmax())
    return id2label[top], float(probs[top]), {id2label[i]: float(p) for i, p in enumerate(probs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="saved_model")
    a = ap.parse_args()
    label, conf, allp = predict_image(a.image, a.model)
    print(f"Prediction: {label}  (confidence {conf:.1%})")
    for k, v in sorted(allp.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v:.3f}")
    if label != "Hazardous" and allp.get("Hazardous", 0) > 0.2:
        print("  NOTE: non-trivial hazardous probability - treat with caution / human check.", file=sys.stderr)
