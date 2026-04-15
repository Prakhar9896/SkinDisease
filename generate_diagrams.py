"""
Generate block diagram images for the project report using matplotlib.
Run: python generate_diagrams.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import numpy as np

OUT_DIR = Path(__file__).parent / "report_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Color Palette ──
C_BLUE = '#2E4057'
C_TEAL = '#048A81'
C_ORANGE = '#E76F51'
C_PURPLE = '#7B2D8E'
C_GREEN = '#2D6A4F'
C_LIGHT = '#F0F4F8'
C_RED = '#C1121F'
C_GOLD = '#E9C46A'
C_DARK = '#264653'
C_WHITE = '#FFFFFF'

def draw_box(ax, x, y, w, h, text, color=C_BLUE, text_color='white', fontsize=9,
             fontweight='bold', style='round,pad=0.3', alpha=1.0, text_wrap=True):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=style, facecolor=color, edgecolor='#333333',
                         linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=fontweight, zorder=3,
            wrap=text_wrap, family='sans-serif')

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'),
                zorder=1)

def draw_arrow_label(ax, x1, y1, x2, y2, label='', color='#555555'):
    """Draw an arrow with a label."""
    draw_arrow(ax, x1, y1, x2, y2, color=color)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx, my + 0.15, label, ha='center', va='bottom', fontsize=7,
            color=color, style='italic', family='sans-serif')


# ══════════════════════════════════════════
# FIGURE 1: System Architecture
# ══════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('EfficientNet-B2 System Architecture', fontsize=14, fontweight='bold',
                 pad=15, family='sans-serif')

    # Input
    draw_box(ax, 1, 2, 1.8, 1.2, 'Input Image\n260×260\nRGB', C_DARK, fontsize=9)
    # Preprocessing
    draw_arrow(ax, 1.9, 2, 2.8, 2)
    draw_box(ax, 3.8, 2, 1.8, 1.2, 'Preprocessing\nResize + Normalize\n(ImageNet μ, σ)', C_TEAL, fontsize=8)
    # Backbone
    draw_arrow(ax, 4.7, 2, 5.5, 2)
    draw_box(ax, 6.6, 2, 2.0, 1.6, 'EfficientNet-B2\nBackbone\n(1408-dim\nfeatures)', C_BLUE, fontsize=9)
    # Head
    draw_arrow(ax, 7.6, 2, 8.4, 2)
    # Head components
    head_x = 9.2
    draw_box(ax, head_x, 3.0, 1.4, 0.5, 'AdaptiveAvgPool(1)', '#4A7C59', fontsize=7, fontweight='normal')
    draw_box(ax, head_x, 2.4, 1.4, 0.5, 'BatchNorm1d(1408)', '#4A7C59', fontsize=7, fontweight='normal')
    draw_box(ax, head_x, 1.8, 1.4, 0.5, 'Dropout(p=0.3)', '#4A7C59', fontsize=7, fontweight='normal')
    draw_box(ax, head_x, 1.2, 1.4, 0.5, 'Linear(1408→16)', '#4A7C59', fontsize=7, fontweight='normal')
    draw_arrow(ax, head_x, 2.75, head_x, 2.65, lw=1.0)
    draw_arrow(ax, head_x, 2.15, head_x, 2.05, lw=1.0)
    draw_arrow(ax, head_x, 1.55, head_x, 1.45, lw=1.0)
    # Label head block
    ax.text(head_x, 3.55, 'Custom Head', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color=C_DARK, family='sans-serif')

    # Output
    draw_arrow(ax, 9.9, 1.2, 10.3, 1.2)
    draw_box(ax, 11.2, 2, 1.8, 2.4, '', C_LIGHT, text_color=C_DARK, fontsize=7, alpha=0.8)
    ax.text(11.2, 3.0, 'Softmax → 16 Classes', ha='center', va='center', fontsize=8,
            fontweight='bold', color=C_DARK, family='sans-serif')
    # Neoplastic
    ax.text(11.2, 2.5, 'Neoplastic:', ha='center', va='center', fontsize=7,
            fontweight='bold', color=C_BLUE, family='sans-serif')
    ax.text(11.2, 2.1, 'MEL BCC SCC AK\nNEV BKL DF VASC SEK', ha='center', va='center',
            fontsize=6.5, color=C_DARK, family='sans-serif')
    # Inflammatory
    ax.text(11.2, 1.4, 'Inflammatory:', ha='center', va='center', fontsize=7,
            fontweight='bold', color=C_ORANGE, family='sans-serif')
    ax.text(11.2, 1.0, 'TINEA PSORIASIS\nVITILIGO MELASMA\nFUNGAL ECZEMA\nURTICARIA', ha='center', va='center',
            fontsize=6, color=C_DARK, family='sans-serif')

    fig.tight_layout()
    path = OUT_DIR / 'fig1_architecture.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


# ══════════════════════════════════════════
# FIGURE 2: Training Pipeline
# ══════════════════════════════════════════
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    ax.set_title('Three-Stage Progressive Transfer Learning Pipeline', fontsize=14,
                 fontweight='bold', pad=15, family='sans-serif')

    # Stage 1
    s1_x, s1_y = 2.2, 3.0
    draw_box(ax, s1_x, s1_y, 3.6, 3.5, '', C_GREEN, alpha=0.15, text_color=C_GREEN)
    ax.text(s1_x, 4.5, 'Stage 1: Dermoscopy Baseline', ha='center', fontsize=10,
            fontweight='bold', color=C_GREEN, family='sans-serif')
    ax.text(s1_x, 3.8, 'HAM10000 (10,015 images)', ha='center', fontsize=8, color=C_DARK, family='sans-serif')
    ax.text(s1_x, 3.3, '7 classes → 16 unified', ha='center', fontsize=7.5, color='#555', family='sans-serif')
    ax.text(s1_x, 2.8, '25 epochs, Adam+CosineAnnealingLR', ha='center', fontsize=7, color='#555', family='sans-serif')
    ax.text(s1_x, 2.3, 'Backbone frozen\nLast 3 blocks + head unfrozen', ha='center', fontsize=7, color='#555', family='sans-serif')
    ax.text(s1_x, 1.6, 'WeightedRandomSampler', ha='center', fontsize=7, color='#555', family='sans-serif')

    # Arrow S1 -> S2
    draw_arrow_label(ax, 4.0, 3.0, 5.0, 3.0, 'stage1.pth', C_GREEN)

    # Stage 2
    s2_x = 7.3
    draw_box(ax, s2_x, s1_y, 3.6, 3.5, '', C_BLUE, alpha=0.15, text_color=C_BLUE)
    ax.text(s2_x, 4.5, 'Stage 2: Domain Bridge', ha='center', fontsize=10,
            fontweight='bold', color=C_BLUE, family='sans-serif')
    ax.text(s2_x, 3.8, 'ISIC 2019 (18,277) + PAD (2,298)', ha='center', fontsize=8, color=C_DARK, family='sans-serif')
    ax.text(s2_x, 3.3, 'MD5 dedup vs HAM (7,054 removed)', ha='center', fontsize=7.5, color='#555', family='sans-serif')
    ax.text(s2_x, 2.8, '20 epochs, Diff. LR (1e-5/1e-4)', ha='center', fontsize=7, color='#555', family='sans-serif')
    ax.text(s2_x, 2.3, 'Last 5 blocks + head unfrozen', ha='center', fontsize=7, color='#555', family='sans-serif')
    ax.text(s2_x, 1.6, 'Head expanded: 7→16 classes', ha='center', fontsize=7, color='#555', family='sans-serif')

    # Arrow S2 -> S3
    draw_arrow_label(ax, 9.1, 3.0, 10.1, 3.0, 'stage2.pth', C_BLUE)

    # Stage 3
    s3_x = 12.3
    draw_box(ax, s3_x, s1_y, 3.6, 3.5, '', C_PURPLE, alpha=0.15, text_color=C_PURPLE)
    ax.text(s3_x, 4.5, 'Stage 3: Skin Tone Robustness', ha='center', fontsize=10,
            fontweight='bold', color=C_PURPLE, family='sans-serif')
    ax.text(s3_x, 3.8, 'Fitz17k (2,553) + DermNet (8,831)', ha='center', fontsize=8, color=C_DARK, family='sans-serif')
    ax.text(s3_x, 3.3, '+ Replay: Stage 2 data (14,681)', ha='center', fontsize=7.5, color='#555', family='sans-serif')
    ax.text(s3_x, 2.8, '12 epochs, Full unfreeze', ha='center', fontsize=7, color='#555', family='sans-serif')
    ax.text(s3_x, 2.3, 'FST-weighted sampling\nSource-aware label smoothing', ha='center', fontsize=7, color='#555', family='sans-serif')
    ax.text(s3_x, 1.5, 'Composite checkpoint scoring', ha='center', fontsize=7, color='#555', family='sans-serif')

    # Output checkpoints
    draw_arrow(ax, s3_x - 0.5, 1.0, s3_x - 0.5, 0.25, C_PURPLE)
    draw_box(ax, s3_x - 0.8, -0.15, 2.5, 0.5, 'stage3_replay_best.pth', C_PURPLE, fontsize=7)
    draw_arrow(ax, s3_x + 0.5, 1.0, s3_x + 0.5, 0.25, C_ORANGE)
    draw_box(ax, s3_x + 1.0, -0.15, 2.5, 0.5, 'fitz_tuned.pth (BEST)', C_ORANGE, fontsize=7)

    fig.tight_layout()
    path = OUT_DIR / 'fig2_pipeline.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


# ══════════════════════════════════════════
# FIGURE 3: Data Preprocessing
# ══════════════════════════════════════════
def fig3_preprocessing():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')
    ax.set_title('Data Preprocessing Pipeline', fontsize=14, fontweight='bold',
                 pad=15, family='sans-serif')

    # Dataset boxes at top
    datasets = [
        (1.5, 6.5, 'HAM10000\n10,015\nDermoscopy', C_GREEN),
        (4.5, 6.5, 'ISIC 2019\n25,331\nDermoscopy', C_BLUE),
        (7.5, 6.5, 'PAD-UFES-20\n2,298\nSmartphone', C_ORANGE),
        (10.5, 6.5, 'Fitz17k + DermNet\n11,394 + 10,390\nClinical Atlas', C_PURPLE),
    ]
    for x, y, text, color in datasets:
        draw_box(ax, x, y, 2.5, 1.2, text, color, fontsize=8)

    # Arrows down
    for x, _, _, _ in datasets:
        draw_arrow(ax, x, 5.9, x, 5.4)

    # Step 1: Label Harmonization
    draw_box(ax, 6, 5.0, 10.5, 0.7, 'Step 1: Label Harmonization — Map diverse labels → 16 unified classes\n'
             '(HAM: 7 classes | Fitz: 114 diagnoses | DermNet: folder keywords)',
             '#3A5A7C', fontsize=8, fontweight='normal')
    draw_arrow(ax, 6, 4.65, 6, 4.2)

    # Step 2: Quality Control
    draw_box(ax, 6, 3.8, 10.5, 0.7, 'Step 2: Quality Control — MD5 deduplication (ISIC vs HAM: 7,054 removed)\n'
             'Filter UNKNOWN labels | Image validation (corrupt/missing)',
             '#3A5A7C', fontsize=8, fontweight='normal')
    draw_arrow(ax, 6, 3.45, 6, 3.0)

    # Step 3: Patient-Level Splitting
    draw_box(ax, 6, 2.6, 10.5, 0.7, 'Step 3: Patient-Level Splitting — GroupShuffleSplit (70/15/15)\n'
             'No patient crosses train/val/test boundaries',
             '#3A5A7C', fontsize=8, fontweight='normal')
    draw_arrow(ax, 6, 2.25, 6, 1.8)

    # Step 4: Output CSVs
    draw_box(ax, 3, 1.3, 3.0, 0.7, 'Train CSV\n(26,065 images)', C_GREEN, fontsize=8)
    draw_box(ax, 6, 1.3, 3.0, 0.7, 'Val CSV\n(4,729 images)', C_TEAL, fontsize=8)
    draw_box(ax, 9, 1.3, 3.0, 0.7, 'Test CSV\n(6,163 images)', C_ORANGE, fontsize=8)

    # Arrows to outputs
    draw_arrow(ax, 4.5, 1.75, 3, 1.65)
    draw_arrow(ax, 6, 1.75, 6, 1.65)
    draw_arrow(ax, 7.5, 1.75, 9, 1.65)

    # Column labels at bottom
    ax.text(6, 0.6, 'Columns: image_path | unified_label | fst_group | split | dataset_source',
            ha='center', fontsize=8, style='italic', color='#666', family='sans-serif')

    fig.tight_layout()
    path = OUT_DIR / 'fig3_preprocessing.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


# ══════════════════════════════════════════
# FIGURE 4: Inference Pipeline
# ══════════════════════════════════════════
def fig4_inference():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title('Mobile Inference & Clinical Decision Support Pipeline', fontsize=14,
                 fontweight='bold', pad=15, family='sans-serif')

    # Camera
    draw_box(ax, 1, 2.5, 1.8, 1.4, 'Smartphone\nCamera', C_DARK, fontsize=9)
    draw_arrow(ax, 1.9, 2.5, 2.6, 2.5)

    # Quality Check
    draw_box(ax, 3.5, 2.5, 1.6, 1.4, 'Quality\nCheck\n(blur/light)', C_TEAL, fontsize=8)
    # Fail path
    draw_arrow(ax, 3.5, 1.8, 3.5, 1.0, C_RED)
    draw_box(ax, 3.5, 0.5, 1.6, 0.6, 'Retake Photo', C_RED, fontsize=8)

    draw_arrow(ax, 4.3, 2.5, 5.0, 2.5)

    # Preprocessing
    draw_box(ax, 5.8, 2.5, 1.4, 1.4, 'Preprocess\n260×260\nNormalize', '#4A7C59', fontsize=8)
    draw_arrow(ax, 6.5, 2.5, 7.2, 2.5)

    # Model
    draw_box(ax, 8.2, 2.5, 1.8, 1.4, 'EfficientNet-B2\n(.ptl <50MB)\nOffline', C_BLUE, fontsize=8)
    draw_arrow(ax, 9.1, 2.5, 9.8, 2.5)

    # Post-processing
    draw_box(ax, 10.5, 2.5, 1.2, 1.4, 'Softmax\nTop-3\nConfidence', C_PURPLE, fontsize=8)
    draw_arrow(ax, 11.1, 2.5, 11.7, 2.5)

    # Decision boxes
    draw_box(ax, 12.8, 3.6, 2.0, 0.6, 'REFER TO CLINIC\n(malignant in top-3)', C_RED, fontsize=7)
    draw_box(ax, 12.8, 2.5, 2.0, 0.6, 'Possible [CLASS]\n(confidence > 75%)', C_ORANGE, fontsize=7)
    draw_box(ax, 12.8, 1.4, 2.0, 0.6, 'Visit clinic\n(confidence 55-75%)', C_GOLD, text_color=C_DARK, fontsize=7)

    draw_arrow(ax, 11.7, 2.8, 11.8, 3.6, C_RED)
    draw_arrow(ax, 11.7, 2.5, 11.8, 2.5, C_ORANGE)
    draw_arrow(ax, 11.7, 2.2, 11.8, 1.4, C_GOLD)

    # Legend
    ax.text(12.8, 0.6, 'Safety: Never outputs\n"low risk — monitor"',
            ha='center', fontsize=7, style='italic', color=C_RED, family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0', edgecolor=C_RED, alpha=0.5))

    fig.tight_layout()
    path = OUT_DIR / 'fig4_inference.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


if __name__ == '__main__':
    print("Generating block diagrams...")
    p1 = fig1_architecture()
    p2 = fig2_pipeline()
    p3 = fig3_preprocessing()
    p4 = fig4_inference()
    print(f"\nAll diagrams saved to: {OUT_DIR}")
