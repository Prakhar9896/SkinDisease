# Skin Disease Classifier — Complete Project Memory
## For Claude Opus 4.6 (Thinking)

---

## Project Overview

A 3-stage transfer learning skin disease classifier for rural India deployment. Runs fully offline inside a React Native mobile app via PyTorch Mobile. Images taken with smartphone cameras by non-clinician rural health workers. Target population is predominantly Indian skin tones (Fitzpatrick Skin Type 4–5).

**Current status:** Training complete. Best checkpoint identified (FitzTuned). Ready for PyTorch Mobile export and React Native integration.

---

## Deployment Context

- **Device:** Android + iOS smartphones (mid-range, rural India)
- **App framework:** React Native
- **Inference runtime:** PyTorch Mobile (`react-native-pytorch-core`)
- **Connectivity:** Fully offline, no internet required
- **Users:** Rural health workers, non-clinicians
- **Purpose:** Screening and triage tool — NOT diagnosis. Always refers to doctor.
- **App size budget:** Under 80MB total (model under 50MB)

---

## Model Architecture

**Backbone:** EfficientNet-B2 via `timm`
```python
timm.create_model('efficientnet_b2', pretrained=False, num_classes=0, global_pool="")
```

**Custom head (attached as `model.head`):**
```python
nn.Sequential(
    nn.AdaptiveAvgPool2d(1),   # head.0
    nn.Flatten(),               # head.1
    nn.BatchNorm1d(1408),      # head.2
    nn.Dropout(0.3),            # head.3
    nn.Linear(1408, 16)        # head.4
)
```

**Forward pass:**
```python
def forward(self, x):
    feats = self.backbone.forward_features(x)
    return self.head(feats)
```

**Checkpoint structure:**
```python
ckpt = torch.load('path.pth', map_location='cpu', weights_only=False)
# ckpt.keys(): ['model_state', 'classes', 'stage', 'best_composite']
# ckpt['model_state'] — load with strict=True
# ckpt['classes']     — list of 16 unified class names
```

**Key state dict layer names:**
- `backbone.conv_stem.weight` — first conv layer
- `backbone.blocks.0-6.*` — 7 MBConv block groups
- `backbone.conv_head.weight` — final backbone conv
- `head.2.*` — BatchNorm1d weights
- `head.4.weight / head.4.bias` — Linear classifier

---

## Preprocessing Contract (CRITICAL — must match exactly in React Native)

```python
PREPROCESSING_CONFIG = {
    'input_size': 260,
    'mean': [0.485, 0.456, 0.406],
    'std':  [0.229, 0.224, 0.225],
    'color': 'RGB'  # NOT BGR
}

transforms.Compose([
    transforms.Resize(260),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
```

React Native must apply identical normalisation before passing tensor to model.

---

## Unified Class Taxonomy (16 classes, fixed order)

```python
UNIFIED_CLASSES = [
    # Malignant (index 0-3)
    'MEL',   # 0 — melanoma
    'BCC',   # 1 — basal cell carcinoma
    'SCC',   # 2 — squamous cell carcinoma
    'AK',    # 3 — actinic keratosis (pre-malignant)
    # Benign lesions (index 4-8)
    'NEV',   # 4 — nevus/mole
    'BKL',   # 5 — benign keratosis-like lesion
    'DF',    # 6 — dermatofibroma
    'VASC',  # 7 — vascular lesion
    'SEK',   # 8 — seborrheic keratosis
    # Inflammatory/Infectious (index 9-15)
    'TINEA',      # 9
    'PSORIASIS',  # 10
    'VITILIGO',   # 11
    'MELASMA',    # 12
    'FUNGAL',     # 13
    'ECZEMA',     # 14
    'URTICARIA',  # 15
]
```

**Label mapping per source dataset:**
| Dataset | Raw labels | Mapped to |
|---|---|---|
| HAM10000 | mel, nv, bcc, akiec, bkl, df, vasc | MEL, NEV, BCC, AK, BKL, DF, VASC |
| PAD-UFES-20 | bcc, mel, scc, ack, sek, nev | BCC, MEL, SCC, AK, SEK, NEV |
| ISIC 2019 | MEL, NV, BCC, AK, BKL, SCC, VASC, DF | direct map |
| Fitzpatrick17k-C | 114 conditions | filtered to 16, ~2,553 usable |
| DermNet (Kaggle) | folder names | keyword matching to 16 |

---

## Training Pipeline — 3 Stages

### Stage 1 — HAM10000 (dermoscopy baseline)
- **Data:** 10,015 dermoscopy images, 7 classes, FST 1-3
- **Label file:** `HAM10000_metadata.csv` (columns: image_id, dx, lesion_id, patient_id)
- **Image dirs:** `HAM10000_images_part1/` and `HAM10000_images_part2/`
- **Setup:** Freeze all except last 3 blocks + head
- **Optimizer:** Adam lr=1e-3, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR
- **Loss:** MaskedCrossEntropyLoss with class frequency weights
- **Checkpoint:** `checkpoints/stage1.pth`

### Stage 2 — ISIC 2019 + PAD-UFES-20 (mobile domain)
- **ISIC 2019:** 25,331 dermoscopy images, 8 classes — adds broader disease coverage
  - **IMPORTANT:** ISIC 2019 is entirely dermoscopy, NO smartphone images
  - Contains all HAM10000 images — deduplicated by MD5 hash before training
  - Label file: `ISIC_2019_Training_GroundTruth.csv` (one-hot encoded)
  - Metadata: `ISIC_2019_Training_Metadata.csv` (has patient_id)
- **PAD-UFES-20:** 1,426 smartphone images, 6 classes, Brazilian population FST 3-4
  - **Only smartphone dataset in pipeline** — critical for mobile deployment
  - Has FST labels in metadata CSV — use them (NOT fst_group=0)
  - Has patient_id — use patient-level splits
  - Label: `ack` maps to AK (not `actinic keratosis`)
- **Setup:** Unfreeze last 5 blocks + head
- **Param groups:** backbone lr=1e-5, head lr=1e-4
- **Augmentation:** Adds mobile camera simulation (GaussianBlur, RandomPerspective, stronger ColorJitter)
- **Sampling:** Oversample PAD-UFES-20 relative to ISIC (50:50 or 40:60 ratio)
- **Checkpoint:** `checkpoints/stage2.pth`
- **Status:** Already done

### Stage 3 — Fitzpatrick17k-C + DermNet (skin tone robustness)
- **Fitzpatrick17k-C:** ~2,553 usable images after label filtering (from 16,577 total)
  - Uses CLEANED version (Fitzpatrick17k-C) not original
  - Has pre-built train/val/test splits — use them directly
  - FST labels available (1-6)
  - Label smoothing=0.1 (crowdsourced labels)
  - Image zip from `derm.cs.sfu.ca/critique` (Fitzpatrick17k-Renamed)
  - CSV from `zenodo.org/doi/10.5281/zenodo.11101337`
- **DermNet (Kaggle):** 8,831 rows used
  - Kaggle scraped version: `kaggle.com/datasets/shubhamgoel27/dermnet`
  - Folder structure: `dermnet_images/<condition>/<image>`
  - Label smoothing=0.15 (web scraped)
  - No FST labels — fst_group=0
  - No official splits — all set to 'train'
- **Training CSV schema:** `image_path, unified_label, fst_group, split, dataset_source, has_disease_label`
- **Dataset source values:** `ISIC2019`, `dermnet`, `fitzpatrick17k`, `PAD-UFES-20`
- **Replay:** Stage 2 data included at 40% of batches to prevent catastrophic forgetting
- **Setup:** Full unfreeze, lr=5e-5
- **Source-aware label smoothing:**
  ```python
  SOURCE_SMOOTHING = {
      'ISIC2019': 0.0, 'PAD-UFES-20': 0.05,
      'fitzpatrick17k': 0.1, 'dermnet': 0.15
  }
  ```
- **FST-weighted sampling:** dark (FST 5-6) oversampled 3x, medium 1.5x, light 1x
- **Composite checkpoint score:**
  ```python
  score = (0.35 * pad_macro_f1
         + 0.25 * avg(MEL_recall, SCC_recall, AK_recall)
         + 0.20 * fst_dark_f1
         + 0.20 * balanced_acc)
  ```
- **Checkpoints available:**
  - `checkpoints/stage3.pth` — original Stage 3
  - `checkpoints/stage3_replay_best.pth` — with experience replay (composite 0.364)
  - `checkpoints/stage3_fitz_tuned.pth` (or similar) — FitzTuned, BEST checkpoint

---

## Checkpoints Summary

| Checkpoint | Composite (PAD) | Composite (Fitz) | High conf wrong (Fitz) | Status |
|---|---|---|---|---|
| stage2.pth | 0.25 | — | — | Baseline |
| stage3.pth (old) | 0.27 | 0.27 | 0.16 | Superseded |
| stage3_replay_best.pth | 0.31 | 0.18 | 0.70 | Superseded |
| FitzTuned | **0.33** | **0.25** | **0.29** | **BEST — USE THIS** |

---

## Final Evaluation Results (FitzTuned vs ReplayBest)

### On Fitzpatrick/DermNet val set:
| Metric | ReplayBest | FitzTuned | Delta |
|---|---|---|---|
| Accuracy | 0.193 | 0.340 | +0.147 |
| PAD macro F1 | 0.147 | 0.210 | +0.063 |
| Balanced acc | 0.188 | 0.322 | +0.133 |
| MEL recall | 0.233 | 0.384 | +0.151 |
| SCC recall | 0.232 | 0.522 | +0.290 |
| AK recall | 0.276 | 0.172 | -0.103 ← regression |
| High conf wrong | 0.704 | 0.286 | -0.418 |
| FST dark F1 | 0.128 | 0.136 | +0.008 |
| Composite | 0.177 | 0.255 | +0.078 |

### On PAD-UFES-20 test set:
| Metric | ReplayBest | FitzTuned | Delta |
|---|---|---|---|
| Accuracy | 0.651 | 0.641 | -0.010 |
| Balanced acc | 0.586 | 0.618 | +0.031 |
| MEL recall | 0.200 | 0.400 | +0.200 |
| SCC recall | 0.524 | 0.524 | 0.000 |
| AK recall | 0.772 | 0.728 | -0.043 |
| High conf wrong | 0.124 | 0.158 | +0.034 |
| Composite | 0.306 | 0.327 | +0.021 |

**FitzTuned wins on both test sets.**

---

## Known Failure Modes

**1. SEK over-prediction on dark skin clinical photos (DDI)**
- SCC on FST V-VI skin predicted as SEK with 88% confidence
- Root cause: model trained mostly on dermoscopy + clinical atlas, not smartphone dark skin
- Mitigation: confidence threshold + malignancy safety net in app

**2. AK recall regression in FitzTuned**
- AK dropped from 0.77 to 0.73 on PAD
- Check if AK is being confused with SCC (acceptable) or NEV/BKL (dangerous)
- Not a deployment blocker if confusion is with SCC

**3. Inflammatory class weakness (TINEA, ECZEMA, PSORIASIS etc.)**
- These classes came from DermNet (clinical atlas) and Fitzpatrick only
- Model has never seen smartphone images of inflammatory conditions
- Domain mismatch causes poor performance on real phone photos
- Mitigation: output "Unknown condition — visit clinic" for low confidence inflammatory predictions

**4. BKL/NEV/SEK triangle confusion**
- Visual ambiguity between these three classes is genuine — even dermatologists struggle
- Not fully solvable with retraining — label boundary is inherently blurry

**5. FST dark F1 remains low (0.14)**
- Only ~200-300 FST 5-6 images mapped to 16 classes after Fitzpatrick filtering
- Not enough dark skin training data for reliable FST V-VI performance
- Requires DDI training data or local Indian clinic data to fix properly

---

## DDI/DDI-2 Fairness Evaluation Results

**DDI skin_tone column values:** 12=FST I-II, 34=FST III-IV, 56=FST V-VI (NOT ITA scores)

**Correct FST mapping:**
```python
FST_MAP = {12: 2, 34: 4, 56: 6}
ddi_df['fst_group'] = ddi_df['skin_tone'].map(FST_MAP)
```

**Evaluation was run but FST V-VI showed 0 samples due to incorrect ITA conversion.**
Needs re-running with correct FST mapping above to get valid light vs dark comparison.

**DDI label mapping:** `disease` column has hyphenated strings e.g. `melanoma-in-situ`
Clean with: `disease.replace('-', ' ').lower().strip()`

**DDI-2 FST column:** `fitzpatrick_skin_type` is string, may be `"3, 4"` — take first value as int.

---

## Datasets Downloaded and Available

| Dataset | Location | Status |
|---|---|---|
| HAM10000 | `data/HAM10000_images_part1/`, `data/HAM10000_images_part2/` | Available |
| ISIC 2019 | `data/isic/ISIC_2019_Training_Input/` | Available |
| PAD-UFES-20 | `data/PAD-UFES-20/archive/` | Available |
| Fitzpatrick17k-C | `data/Fitzpatrick17k_CategorizedAbbrvs/` | Available |
| DermNet (Kaggle) | `data/DermNet/` | Available |
| DDI | `data/ddidiversedermatologyimages/` | Available |
| DDI-2 | `data/ddi2diversedermatologyimages2/` | Available |

**Fitzpatrick17k-C files:**
- `fitzpatrick17k_c.csv` — cleaned labels + FST + pre-built splits
- `fitzpatrick17k_diagnosismapping.xlsx` — diagnosis mapping file
- `Fitzpatrick17k_DiagnosisMapping.xlsx` — abbreviation mapping
- Image ZIP extracted to `data/Fitzpatrick17k_CategorizedAbbrvs/`

**DDI metadata:** `data/ddi_metadata.csv`
- Columns: `Unnamed: 0, DDI_ID, DDI_file, skin_tone, malignant, disease`

**DDI-2 metadata:** `data/ddi2diversedermatologyimages2/final_DDI2_Asian_spreadsheet.csv`
- Key columns: `photo_id, fitzpatrick_skin_type, diagnosis_detailed, diagnosis_general, benign/malignant`

---

## Training Data CSVs

| File | Contents | Rows |
|---|---|---|
| `stage3_train.csv` | ISIC2019 + dermnet + fitzpatrick17k + PAD-UFES-20 | ~26,065 |
| `stage3_val.csv` | Val split (mostly Fitz-C + PAD val) | ~4,729 |

**stage3_train.csv dataset_source value_counts:**
```
ISIC2019         13,255
dermnet           8,831
fitzpatrick17k    2,553
PAD-UFES-20       1,426
```

---

## PyTorch Mobile Export

**Target file:** `checkpoints/skin_model_final.ptl`
**Export method:** `torch.jit.script` + `optimize_for_mobile()` + `_save_for_lite_interpreter()`
**Backend:** Default CPU (works on both Android and iOS)
**Size target:** Under 50MB

**InferenceWrapper for export:**
```python
class InferenceWrapper(nn.Module):
    def forward(self, x: torch.Tensor):
        # x: float32 [1, 3, 260, 260] normalised
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values
        top3_idx = probs.topk(3, dim=1).indices.squeeze(0)
        return probs, confidence, top3_idx
        # Returns: Tuple[Tensor[1,16], Tensor[1], Tensor[3]]
        # MUST return tuple of tensors only — no dicts, no strings
```

**React Native class index mapping (fixed):**
```
0=MEL  1=BCC  2=SCC  3=AK   4=NEV  5=BKL
6=DF   7=VASC 8=SEK  9=TINEA 10=PSORIASIS
11=VITILIGO 12=MELASMA 13=FUNGAL 14=ECZEMA 15=URTICARIA
```

---

## App Referral Logic (Safety-Critical)

```python
MALIGNANT_CLASSES = {'MEL', 'BCC', 'SCC', 'AK'}

# Check top 3 for malignant — safety net
top3_labels = [UNIFIED_CLASSES[i] for i in top3_idx]
top3_has_malignant = any(c in MALIGNANT_CLASSES for c in top3_labels)

if top3_has_malignant:
    referral = "REFER TO CLINIC — possible malignant lesion"
elif confidence >= 0.75:
    referral = "Possible [CLASS] — confirm with health worker"
elif confidence >= 0.55:
    referral = "Unclear — visit nearest clinic"
else:
    referral = "Poor image or unclear — retake photo"

# Image quality checks BEFORE model inference:
# Laplacian variance < 100 → blurry → retake
# Mean brightness < 40 or > 220 → bad lighting → retake
```

**NEVER output "low risk — monitor".** Every output must encourage human confirmation.

---

## Environment

- **Python:** 3.10.20
- **Framework:** PyTorch + timm + torchvision
- **IDE:** Jupyter notebooks (VS Code)
- **OS:** Windows (paths use backslash)
- **Device:** CPU for inference export
- **Key libraries:** torch, timm, torchvision, scikit-learn, pandas, Pillow, opencv-python, tqdm, matplotlib, seaborn

---

## File Structure

```
checkpoints/
  stage1.pth
  stage1_finetuned.pth
  stage2.pth
  stage3.pth
  stage3_replay_best.pth
  [fitz_tuned checkpoint — BEST]
  skin_model_final.ptl  ← to be generated

data/
  HAM10000_images_part1/
  HAM10000_images_part2/
  HAM10000_metadata.csv
  isic/ISIC_2019_Training_Input/
  isic/ISIC_2019_Training_GroundTruth.csv
  isic/ISIC_2019_Training_Metadata.csv
  PAD-UFES-20/archive/
  Fitzpatrick17k_CategorizedAbbrvs/
  fitzpatrick17k_c.csv
  fitzpatrick17k_diagnosismapping.xlsx
  DermNet/
  ddidiversedermatologyimages/
  ddi_metadata.csv
  ddi2diversedermatologyimages2/
  ddi2diversedermatologyimages2/final_DDI2_Asian_spreadsheet.csv

stage3_train.csv
stage3_val.csv
```

---

## Immediate Next Steps

1. **Export FitzTuned to `.ptl`** — use 4-cell export notebook (already written)
   - Rebuild model with `EfficientNetB2Unified` class
   - Load FitzTuned checkpoint with `strict=True`
   - Wrap in `InferenceWrapper`
   - Script → optimize → save → verify size < 50MB

2. **Fix DDI evaluation** — re-run with correct FST mapping:
   `FST_MAP = {12: 2, 34: 4, 56: 6}`
   to get valid light vs dark skin performance gap

3. **Check AK confusion in FitzTuned** — confirm AK regression
   is SCC confusion (acceptable) not NEV/BKL confusion (dangerous)

4. **Hand `.ptl` + preprocessing contract to React Native developer**

---

## V2 Roadmap (Post Deployment)

**High priority:**
- Add SCIN dataset (GitHub, free, no registration) for smartphone inflammatory class coverage
  - 10,000+ smartphone images with FST labels
  - Covers TINEA, ECZEMA, URTICARIA, FUNGAL, PSORIASIS
  - Download: `github.com/google-research-datasets/scin`
- Collect 300-500 images from Karnataka clinics (highest real-world impact)
- Re-attempt DDI access for dark skin SCC training data

**Architecture consideration:**
- Two-model approach: separate lesion classifier + inflammatory classifier
- Expand taxonomy to include: ACNE, SCABIES, INSECT_BITE, WOUND_INF

**Fairness:**
- Re-run DDI evaluation with corrected FST mapping
- Publish FST-stratified performance metrics with the app

---

## Key Decisions and Rationale

| Decision | Rationale |
|---|---|
| EfficientNet-B2 not B3 | B2 fits 80MB budget comfortably, B3 too close to limit |
| PyTorch Mobile not TFLite | Avoids lossy PyTorch→ONNX→TFLite conversion |
| Default CPU backend | Works on both Android and iOS |
| 16 unified classes | Covers malignant + benign + Indian-relevant inflammatory |
| Patient-level splits always | Medical datasets — same patient must not cross train/test |
| Fitzpatrick17k-C not original | Original has 6,600+ duplicates and label errors |
| DermNet = Kaggle scraped | Paid official DermNet NZ not used |
| ISIC 2019 not 2018 | 2018 = HAM10000 exactly, would cause data leakage |
| Never "low risk — monitor" | Model not validated enough for safety verdicts |
| Stop training, deploy v1 | Data ceiling hit — more training gives diminishing returns |

---

## Important Warnings

- **ISIC 2018 = HAM10000** — never download both, identical data
- **ISIC 2019 is dermoscopy only** — no smartphone images
- **PAD-UFES-20 FST labels exist** — use `fitzpatrick` column, do NOT set fst_group=0
- **Fitzpatrick17k skin_tone=56** is FST V-VI group code, NOT ITA score 56
- **DDI skin_tone values:** 12=FST I-II, 34=FST III-IV, 56=FST V-VI
- **Never hardcode 16** — always use `len(UNIFIED_CLASSES)`
- **InferenceWrapper forward must return tuple of tensors only** — TorchScript requirement
- **FitzTuned has AK recall regression** — check confusion before declaring safe
- **Inflammatory classes are weak on smartphone** — model trained only on clinical atlas for these
