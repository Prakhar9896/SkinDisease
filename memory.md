# Project Memory — Skin Disease Classifier for Rural India

## Project Goal
Build a skin disease classification model that runs **fully offline** inside a **React Native mobile app** for rural India deployment. Images are taken with smartphone cameras by non-clinician health workers. Patients are predominantly Indian skin tones (Fitzpatrick Skin Type 4–5).

---

## Final Model Choice
- **Backbone:** EfficientNet-B2 (via `timm` library)
- **Input size:** 260×260
- **Export format:** PyTorch Mobile `.ptl` for React Native
- **Target size:** under 50MB (full app under 80MB)
- **Reason over MobileNetV3:** Better accuracy for fine-grained medical classification, 260×260 input captures more lesion detail, fits budget comfortably (~63MB total with app)

---

## Training Pipeline — 3 Stages

### Stage 1 — HAM10000 (baseline)
- **Images:** ~10,015 dermoscopy images
- **Classes:** mel, nv, bcc, akiec, bkl, df, vasc (7 classes)
- **Camera:** Dermoscopy
- **FST:** Not available, set fst_group=0
- **Purpose:** Learn strong disease features from high quality clinical images
- **Label file:** `HAM10000_metadata.csv` (columns: image_id, dx, lesion_id, patient_id)
- **Image dirs:** `HAM10000_images_part1/` and `HAM10000_images_part2/`
- **Status:** To build — Copilot prompt written (Stage 1 prompt)

### Stage 2 — ISIC 2019 + PAD-UFES-20 (already done)
- **ISIC 2019:** ~25,331 dermoscopy images, 8 classes — adds broader disease coverage
  - **Important:** ISIC 2019 is ALL dermoscopy, NO smartphone images
  - Contains all HAM10000 images — must deduplicate before training
  - Download: `kaggle datasets download andrewmvd/isic-2019`
- **PAD-UFES-20:** ~2,298 smartphone images, 6 classes — Brazilian population FST 3–4
  - Fully smartphone collected (6 different phone models, real clinical conditions)
  - Already has FST labels in metadata CSV — use them, do NOT set fst_group=0
  - Has patient IDs — use patient-level splits
  - Download: Mendeley Data
- **Purpose:** ISIC broadens disease classes. PAD-UFES-20 is the ONLY mobile camera bridge in Stage 2
- **Sampling:** Oversample PAD-UFES-20 (50:50 or 40:60 ISIC:PAD ratio) — PAD is smaller but critical
- **Status:** Already completed

### Stage 3 — Fitzpatrick17k-C + DermNet (skin tone robustness)
- **Fitzpatrick17k-C:** ~16,577 images, FST labels 1–6
  - Use the CLEANED version (Fitzpatrick17k-C), NOT the original
  - Has pre-built standardised train/val/test splits
  - Download: `zenodo.org/doi/10.5281/zenodo.11101337`
  - Images: Download ZIP from `derm.cs.sfu.ca/critique` (Fitzpatrick17k-Renamed ZIP)
  - Images organised in folders named after diagnosis abbreviations
  - Filenames encode FST labels
- **DermNet:** ~19,500 images, 23 disease categories
  - Use Kaggle scraped version: `kaggle.com/datasets/shubhamgoel27/dermnet`
  - Folder structure: `dermnet_images/<condition_name>/<image_file>`
  - Labels are noisier — use label_smoothing=0.1
  - No FST labels — set fst_group=0
  - No official splits — use 'train' for all
- **Purpose:** Fill FST 4–6 gap (HAM/ISIC are FST 1–3). DermNet adds tropical/inflammatory diseases common in rural India (tinea, fungal, melasma, vitiligo, psoriasis, eczema)
- **Status:** Preprocessing in progress

---

## Dataset Summary

| Stage | Dataset | Images | Camera | FST |
|---|---|---|---|---|
| 1 | HAM10000 | 10,015 | Dermoscopy | 1–3 |
| 2 | ISIC 2019 | 25,331 | Dermoscopy | 1–3 |
| 2 | PAD-UFES-20 | 2,298 | Smartphone | 3–4 |
| 3 | Fitzpatrick17k-C | 16,577 | Clinical mixed | 1–6 |
| 3 | DermNet (Kaggle) | 19,500 | Clinical mixed | unknown |
| **Total** | | **~73,721** | | |

### Optional Stage 4 (future)
- Indian clinic data (~500–1,000 labelled images from rural Karnataka)
- Collected from local clinics, labelled by dermatologist
- Last-mile fine-tuning — highest real-world impact despite small size
- Scaffolded as TODO in code — not yet available

---

## Unified Class Taxonomy

```python
UNIFIED_CLASSES = [
    # Malignant
    'MEL', 'BCC', 'SCC', 'AK',
    # Benign
    'NEV', 'BKL', 'DF', 'VASC', 'SEK',
    # Inflammatory / Infectious (important for rural India)
    'TINEA', 'PSORIASIS', 'VITILIGO',
    'MELASMA', 'FUNGAL', 'ECZEMA', 'URTICARIA'
]
```

### Label mapping per dataset
| Dataset | Raw label | Unified |
|---|---|---|
| HAM10000 | mel, nv, bcc, akiec, bkl, df, vasc | MEL, NEV, BCC, AK, BKL, DF, VASC |
| PAD-UFES-20 | bcc, mel, scc, ack, sek, nev | BCC, MEL, SCC, AK, SEK, NEV |
| ISIC 2019 | MEL, NV, BCC, AK, BKL, SCC, VASC, DF | direct map |
| Fitzpatrick17k | 114 conditions | via DiagnosisMapping file |
| DermNet | folder names | keyword matching |

---

## Key Technical Decisions

### Loss function
- `MaskedCrossEntropyLoss` — masks invalid classes per dataset source before softmax
- Each dataset only contributes loss for classes it actually contains
- `label_smoothing=0.1` for Fitzpatrick17k and DermNet (noisier labels)

### Data splits
- Always split by **patient_id**, never by image
- Same patient must not appear in train and val/test
- HAM10000 and ISIC 2019 have patient_id columns — use them
- Fitzpatrick17k-C has pre-built splits — use them directly

### Sampling
- `WeightedSampler` balancing by both disease class AND FST band
- FST bands: light (1–2), medium (3–4), dark (5–6)
- Stage 2: oversample PAD-UFES-20 to compensate for small size

### Deduplication
- ISIC 2019 contains all HAM10000 images
- Must deduplicate ISIC 2019 against HAM10000 training set by MD5 hash before Stage 2

### Augmentation
- Stage 1: Standard dermoscopy (flip, rotate, ColorJitter, RandomErasing)
- Stage 2: Adds mobile simulation (GaussianBlur, RandomPerspective, stronger ColorJitter, RandomAdjustSharpness)
- Stage 3: Adds skin tone simulation (LAB colorspace L* reduction for FST≥4)

---

## Model Architecture

```python
# EfficientNet-B2 via timm
backbone = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
# feature_dim = 1408

head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.BatchNorm1d(1408),
    nn.Dropout(0.3),
    nn.Linear(1408, len(UNIFIED_CLASSES))
)
```

### Fine-tuning strategy per stage
| Stage | Frozen | Unfrozen | Backbone LR | Head LR |
|---|---|---|---|---|
| 1 | all except last 3 blocks | last 3 blocks + head | — | 1e-3 |
| 2 | — | last 5 blocks + head | 1e-5 | 1e-4 |
| 3 | — | all layers | 5e-5 | 5e-5 |

---

## PREPROCESSING_CONFIG
```python
PREPROCESSING_CONFIG = {
    'input_size': 260,
    'mean': [0.485, 0.456, 0.406],
    'std':  [0.229, 0.224, 0.225],
}
```
**Critical:** React Native app must use identical values for inference preprocessing.

---

## Inference Output
```python
# predict() returns:
{
    'top3_classes':  [...],   # top 3 UNIFIED_CLASSES names
    'top3_probs':    [...],   # top 3 softmax probabilities
    'confidence':    float,
    'referral':      str      # one of:
    # >= 0.85  → 'LOW RISK - monitor, recheck in 4 weeks'
    # 0.60-0.85 → 'UNCERTAIN - visit nearest clinic'
    # < 0.60   → 'LOW CONFIDENCE - visit clinic'
    # blur/bad lighting → 'POOR IMAGE - retake'
}
```

Image quality check runs BEFORE disease prediction:
- Laplacian variance < 100 → blurry
- Mean brightness < 40 or > 220 → bad lighting

---

## React Native Integration
- Export via `torch.jit.script` + `optimize_for_mobile()` → `.ptl` file
- Runtime: `react-native-pytorch-core` (PyTorch Mobile)
- `InferenceWrapper` module wraps model — accepts normalised `[1,3,260,260]` tensor, returns softmax probs + confidence + top3
- React Native developer must match `PREPROCESSING_CONFIG` exactly

---

## File Structure (generated by Copilot)
```
config.py           — UNIFIED_CLASSES, LABEL_MAP, PREPROCESSING_CONFIG
dataset.py          — SkinDataset, build_splits, get_augmentation,
                      WeightedSampler, MaskedCrossEntropyLoss
model.py            — build_efficientnet_b2()
train_stage1.py     — Stage 1 training loop
train_stage2.py     — Stage 2 training loop
train_stage3.py     — Stage 3 training loop
evaluate.py         — evaluate() with per-FST-band metrics
predict.py          — predict() with image quality check
export.py           — export_to_pytorch_mobile()
checkpoints/
  stage1.pth
  stage2.pth
  stage3.pth
  skin_model.ptl    — final React Native model
```

---

## Fitzpatrick17k-C Files Downloaded
From `derm.cs.sfu.ca/critique` and `zenodo.org/doi/10.5281/zenodo.11101337`:

| File | Status | Purpose |
|---|---|---|
| Fitzpatrick17k-C.csv | ✓ Have | Cleaned labels + FST + splits |
| Fitzpatrick17k_DiagnosisMapping.xlsx | ✓ Have | Maps 114 diagnoses to categories |
| Fitzpatrick17k-Renamed ZIP | Downloading | All 16,577 images in diagnosis folders |
| Dataset Metadata Raw CSV | Download | Updated filenames matching ZIP |

---

## Copilot Prompts Written
1. **Full 3-stage prompt** — complete pipeline overview (use as reference)
2. **Stage 1 prompt** — HAM10000 only, generates 6 files (config.py, dataset.py, model.py, train_stage1.py, evaluate.py, predict.py)
3. **Stage 3 preprocessing prompt (Jupyter)** — 11 cells for Fitzpatrick17k-C + DermNet preprocessing → outputs `stage3_merged.csv`

---

## Important Warnings
- **Never hardcode 7 classes** — always use `len(UNIFIED_CLASSES)`
- **ISIC 2018 = HAM10000** — never download both, it is the same data
- **ISIC 2019 is dermoscopy only** — PAD-UFES-20 is the only smartphone data in Stage 2
- **Deduplicate ISIC 2019 vs HAM10000** by MD5 hash before Stage 2
- **Use Fitzpatrick17k-C not original** — original has 6,600+ duplicates and label errors
- **DermNet is Kaggle scraped version** — not the paid official DermNet NZ dataset
- **Patient-level splits always** — never split by image when patient_id is available

---

## Next Steps
1. Wait for Fitzpatrick17k-Renamed ZIP to finish downloading
2. Download Dataset Metadata Raw CSV from same page
3. Check ZIP folder structure (screenshot and verify)
4. Run Stage 3 preprocessing Jupyter notebook → generate `stage3_merged.csv`
5. Start Stage 1 training with HAM10000 using Copilot Stage 1 prompt
6. Write Stage 3 training prompt (after preprocessing is confirmed working)
