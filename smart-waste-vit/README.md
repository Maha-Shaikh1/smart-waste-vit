# Smart Waste Classification Using Vision Transformers

**Evaluating Hazardous Waste Detection and Classification Reliability**

Shaikh Maha (1560980) · MAAI901 Deep Learning for Visual Data · University of Wollongong in Dubai
Individual coursework – Applied Vision Transformer Report (20%).

## Project overview

This project fine-tunes a pretrained Vision Transformer (ViT-Base, patch 16, 224×224) to classify photos of household waste into four categories: **Hazardous, Non-Recyclable, Organic, Recyclable**. Beyond overall accuracy, it focuses on the errors that matter most in a sorting system: **hazardous items (batteries, e-waste, paints, pesticides) that the model wrongly predicts as a non-hazardous class** (false negatives). It also tests how the model behaves when test images are blurred, darkened or heavily compressed.

## Research question

> How reliably can a Vision Transformer classify waste images into different waste categories, and what types of errors could create risks when identifying hazardous waste?

## Dataset and model

| | Link | Notes |
|---|---|---|
| Dataset | https://huggingface.co/datasets/huaweilin/waste-classification | ~1,976 real photos, single split `cleaned`, 4 classes / 18 subclasses. Repackaged from the Kaggle "Waste Classification" dataset by PhenomSG: https://www.kaggle.com/datasets/phenomsg/waste-classification |
| Model | https://huggingface.co/google/vit-base-patch16-224 | ViT-B/16, pretrained on ImageNet-21k, fine-tuned on ImageNet-1k. Licence: Apache-2.0 |

### Dataset licence and attribution

The Hugging Face card for `huaweilin/waste-classification` does not state its own licence and asks users to follow the original Kaggle dataset's terms. The original Kaggle dataset is released under the **MIT License**. The dataset is used here for non-commercial academic coursework only. All images remain the property of their original owners.

## Repository structure

```
smart-waste-vit/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── smart_waste_vit.ipynb     # main, self-contained Colab notebook (run this)
├── src/
│   ├── train.py                  # script version of the training pipeline
│   ├── evaluate.py               # test-set evaluation + robustness experiment
│   └── predict.py                # classify a single image with the saved model
├── results/                      # metrics (CSV/JSON) and figures produced by the notebook
├── report/
│   └── technical_report.md
└── presentation/
    └── presentation_outline.md
```

## Setup

The easiest way is Google Colab (free T4 GPU):

1. Open https://colab.research.google.com → *File → Upload notebook* → choose `notebooks/smart_waste_vit.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*. Expected runtime ≈ 15–25 min.
4. Download `results.zip` from the Files panel when finished and unzip it into `results/`.

Local run (needs a CUDA GPU with ≥ 8 GB, or be patient on CPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py              # trains and saves to saved_model/
python src/evaluate.py           # test metrics, confusion matrix, robustness
python src/predict.py path/to/photo.jpg
```

## Training and evaluation steps (what the notebook does)

1. Load the `cleaned` split from Hugging Face.
2. Remove corrupted images and exact duplicates (MD5 of a 64×64 thumbnail) **before** splitting, to avoid leakage.
3. Stratified 70 / 15 / 15 train / validation / test split (seed 42, stratified by subclass).
4. Preprocess with the ViT image processor (224×224, mean/std 0.5); augment training images only (random resized crop, flip, mild colour jitter).
5. Replace the 1000-class ImageNet head with a 4-class head and fine-tune for 5 epochs (batch 32, lr 5e-5, cosine schedule, warm-up 10 %, fp16). Best epoch chosen by validation macro F1.
6. Evaluate once on the held-out test set: accuracy, macro precision / recall / F1, per-class report, confusion matrix, hazardous recall and false-negative rate, per-subclass hazardous recall.
7. Robustness experiment on the same test images: Gaussian blur (r = 2), brightness × 0.5, JPEG quality 20.

## Results

From `results/summary.json` (Google Colab, Tesla T4, seed 42, 5 epochs, 5.2 min training; 1,788 images after de-duplication, 1,251 / 268 / 269 split).

| Metric (test set, original images) | Value |
|---|---|
| Accuracy | 92.6 % (249 / 269) |
| Macro precision / recall / F1 | 0.929 / 0.922 / 0.925 |
| Hazardous precision / recall | 0.964 / 0.898 |
| Hazardous false-negative rate | 10.2 % (6 of 59 missed: 3 paints, 3 e-waste) |
| Hazardous recall by subclass | batteries 1.00 · pesticides 1.00 · e-waste 0.89 · paints 0.75 |

| Condition | Accuracy | Macro F1 | Hazardous recall | Hazardous FN / FP |
|---|---|---|---|---|
| Original | 0.926 | 0.925 | 0.898 | 6 / 2 |
| Blur (r=2) | 0.914 | 0.914 | 0.898 | 6 / 6 |
| Dark (×0.5) | 0.903 | 0.903 | 0.915 | 5 / 6 |
| JPEG (q=20) | 0.918 | 0.917 | 0.881 | 7 / 2 |

Figures: `results/figures/fig01…fig11` (class distribution, samples, training curves, confusion matrix, hazardous recall per subclass, degradation examples, robustness chart, correct/incorrect examples, prediction demo).

## Example prediction

```python
from src.predict import predict_image
label, confidence, all_probs = predict_image("my_photo.jpg", model_dir="saved_model")
print(label, confidence)
```

## Limitations and ethical considerations

- **Small dataset (~2k images)** with uneven class sizes; results have noticeable statistical uncertainty.
- Images are mostly clean, single-object, product-style photos. A model trained on them may learn **background shortcuts** and perform worse on real bin contents (dirt, mixed items, occlusion). The degradation experiment only partly simulates this.
- The "Hazardous" category covers only four subclasses; many real hazardous items (medicines, syringes, chemicals, gas canisters) are **absent**, so the model cannot be expected to recognise them.
- A hazardous item predicted as Organic or Recyclable (false negative) could enter the wrong waste stream and cause fires (batteries), soil/water contamination (pesticides, paints) or injury to workers. Such a model must **not** be used as the only decision-maker; it is a support tool that needs human oversight and a conservative "when unsure, treat as hazardous" policy.
- The dataset's geographic origin and labelling process are not documented, so cultural/regional bias (packaging types, local recycling rules) cannot be ruled out.
