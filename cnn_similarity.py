"""
CNN / ResNet Similarity Pipeline
==================================
Uses ResNet (18/50/101) as a shared feature extractor to compare:
  - Person crops from PPE model   (ppe.pt labels)
  - Person crops from Person model (person labels)
  - PPE crops (helmet / coverall) → assign to correct person

Similarity Methods supported:
  1. Cosine Similarity          → angle between feature vectors
  2. Euclidean Distance         → L2 distance in feature space
  3. Siamese Network (optional) → learned similarity (fine-tunable)

Output:
  - Per-image match report
  - Similarity heatmaps
  - t-SNE embedding plot (visualize cluster separation)
  - JSON results
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
class Config:
    # Paths
    TEST_IMAGES_DIR = "test_images/"
    LABELS_DIR      = "labels/"          # labels/ppe/ and labels/person/
    OUTPUT_DIR      = "output_cnn/"

    # Class IDs in PPE model labels
    PPE_PERSON_CLASS_ID   = 0
    PPE_HELMET_CLASS_ID   = 1
    PPE_COVERALL_CLASS_ID = 2
    PPE_CLASS_NAMES       = {0: "person", 1: "helmet", 2: "coverall"}

    # Class ID in Person model labels
    PERSON_CLASS_ID = 0

    # CNN backbone: choose one → "resnet18" | "resnet50" | "resnet101"
    BACKBONE = "resnet50"

    # Similarity method: "cosine" | "euclidean" | "both"
    SIMILARITY_METHOD = "both"

    # Thresholds
    COSINE_THRESHOLD    = 0.60   # above → match
    EUCLIDEAN_THRESHOLD = 15.0   # below → match (tune per backbone)

    # Assignment blending
    ALPHA_SPATIAL    = 0.4   # weight for IoP spatial score
    ALPHA_EMBEDDING  = 0.6   # weight for CNN embedding score

    # t-SNE
    RUN_TSNE = True


# ─────────────────────────────────────────────────────────────
# BACKBONE FACTORY
# ─────────────────────────────────────────────────────────────
BACKBONE_CONFIGS = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.DEFAULT,  512),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.DEFAULT,  2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
}

def build_backbone(name: str, device: str):
    """
    Build ResNet backbone with FC head removed.
    Output: [B, embedding_dim] feature vectors.

    Architecture (ResNet50 example):
        Input [3,224,224]
            ↓
        Conv1 + BN + ReLU + MaxPool   → [64, 56, 56]
            ↓
        Layer1 (3 bottleneck blocks)  → [256, 56, 56]
            ↓
        Layer2 (4 bottleneck blocks)  → [512, 28, 28]
            ↓
        Layer3 (6 bottleneck blocks)  → [1024, 14, 14]
            ↓
        Layer4 (3 bottleneck blocks)  → [2048,  7,  7]
            ↓
        AdaptiveAvgPool2d(1,1)        → [2048,  1,  1]
            ↓
        Flatten                       → [2048]          ← embedding vector
            ↓
        [FC head REMOVED]
    """
    assert name in BACKBONE_CONFIGS, f"Unknown backbone: {name}"
    model_fn, weights, emb_dim = BACKBONE_CONFIGS[name]

    base = model_fn(weights=weights)

    # Remove the final FC classification layer
    # Keep everything up to and including AdaptiveAvgPool
    layers = list(base.children())[:-1]   # drop Linear(emb_dim → 1000)
    backbone = nn.Sequential(*layers)
    backbone.eval().to(device)

    print(f"  Backbone : {name}")
    print(f"  Emb dim  : {emb_dim}")
    print(f"  Device   : {device}")
    return backbone, emb_dim


# ─────────────────────────────────────────────────────────────
# SIAMESE NETWORK (optional fine-tunable head)
# ─────────────────────────────────────────────────────────────
class SiameseHead(nn.Module):
    """
    Lightweight projection head on top of CNN features.
    Projects high-dim embeddings → compact 128-d space
    optimized for similarity comparison.

    Can be fine-tuned with contrastive / triplet loss
    on your factory data if needed.
    """
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)   # L2-normalized output

    def similarity(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity (= dot product after L2-norm)"""
        return (emb_a * emb_b).sum(dim=-1)


# ─────────────────────────────────────────────────────────────
# CNN EMBEDDER
# ─────────────────────────────────────────────────────────────
class CNNEmbedder:
    """
    Extracts embeddings from image crops using a CNN backbone.

    Pipeline per crop:
        BGR image crop
            ↓
        RGB conversion
            ↓
        Resize to 224×224
            ↓
        ImageNet normalization
            ↓
        ResNet backbone (no FC)
            ↓
        AdaptiveAvgPool → flatten
            ↓
        L2-normalize
            ↓
        [embedding_dim] vector
    """

    def __init__(self, backbone_name: str = "resnet50",
                 use_siamese_head: bool = False,
                 device: str = "cpu"):
        self.device = device
        self.backbone, self.emb_dim = build_backbone(backbone_name, device)
        self.use_siamese = use_siamese_head

        if use_siamese_head:
            self.head = SiameseHead(in_dim=self.emb_dim, out_dim=128).to(device)
            self.head.eval()
            print("  Siamese projection head: ON (128-d output)")
        else:
            self.head = None
            print("  Siamese projection head: OFF (raw backbone features)")

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """
        Extract embedding for a single bounding box crop.

        Args:
            image : Full BGR frame (H, W, 3)
            bbox  : [x1, y1, x2, y2] in pixel coords

        Returns:
            embedding : 1D numpy array, L2-normalized
        """
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]

        # Guard: clamp to image bounds
        x2 = min(x2, image.shape[1])
        y2 = min(y2, image.shape[0])

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            dim = 128 if self.use_siamese else self.emb_dim
            return np.zeros(dim)

        # BGR → RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Transform + batch dimension
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)  # [1,3,224,224]

        with torch.no_grad():
            feat = self.backbone(tensor)          # [1, emb_dim, 1, 1]
            feat = feat.flatten(start_dim=1)      # [1, emb_dim]

            if self.head is not None:
                feat = self.head(feat)             # [1, 128]

        emb = feat.squeeze(0).cpu().numpy()       # [emb_dim] or [128]

        # L2 normalize
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def extract_batch(self, image: np.ndarray, bboxes: list) -> np.ndarray:
        """
        Extract embeddings for multiple bboxes efficiently in one batch.

        Returns:
            embeddings : shape [N, emb_dim]
        """
        if not bboxes:
            return np.empty((0, 128 if self.use_siamese else self.emb_dim))

        crops = []
        valid_idx = []
        dim = 128 if self.use_siamese else self.emb_dim

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            x2 = min(x2, image.shape[1])
            y2 = min(y2, image.shape[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(self.transform(crop_rgb))
            valid_idx.append(i)

        if not crops:
            return np.zeros((len(bboxes), dim))

        batch = torch.stack(crops).to(self.device)   # [N, 3, 224, 224]

        with torch.no_grad():
            feats = self.backbone(batch)              # [N, emb_dim, 1, 1]
            feats = feats.flatten(start_dim=1)        # [N, emb_dim]
            if self.head is not None:
                feats = self.head(feats)               # [N, 128]

        embs_valid = feats.cpu().numpy()

        # L2 normalize each row
        norms = np.linalg.norm(embs_valid, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embs_valid = embs_valid / norms

        # Place back into full array (zeros for invalid crops)
        result = np.zeros((len(bboxes), dim))
        for out_i, in_i in enumerate(valid_idx):
            result[in_i] = embs_valid[out_i]

        return result


# ─────────────────────────────────────────────────────────────
# SIMILARITY CALCULATOR
# ─────────────────────────────────────────────────────────────
class SimilarityCalculator:
    """
    Computes pairwise similarity between two sets of embeddings.
    Supports cosine, euclidean, and combined scoring.
    """

    @staticmethod
    def cosine_matrix(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
        """
        Cosine similarity matrix.
        Range: [-1, 1] → clipped to [0, 1] for PPE use case
        Higher = more similar.

        Shape: [len(a), len(b)]
        """
        if embs_a.size == 0 or embs_b.size == 0:
            return np.zeros((len(embs_a), len(embs_b)))
        sim = cosine_similarity(embs_a, embs_b)
        return np.clip(sim, 0, 1)

    @staticmethod
    def euclidean_matrix(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
        """
        Euclidean distance matrix, converted to similarity score.
        Lower distance = higher similarity.
        Normalized to [0, 1] via: sim = 1 / (1 + distance)

        Shape: [len(a), len(b)]
        """
        if embs_a.size == 0 or embs_b.size == 0:
            return np.zeros((len(embs_a), len(embs_b)))
        dist = cdist(embs_a, embs_b, metric="euclidean")
        return 1.0 / (1.0 + dist)   # maps [0, ∞) → (0, 1]

    @staticmethod
    def combined_matrix(embs_a: np.ndarray, embs_b: np.ndarray,
                        w_cos: float = 0.6, w_euc: float = 0.4) -> np.ndarray:
        """
        Weighted combination of cosine + euclidean similarity.
        Both are [0,1] normalized before combining.
        """
        cos = SimilarityCalculator.cosine_matrix(embs_a, embs_b)
        euc = SimilarityCalculator.euclidean_matrix(embs_a, embs_b)
        return w_cos * cos + w_euc * euc

    @staticmethod
    def hungarian_assign(sim_matrix: np.ndarray) -> list:
        """
        Optimal 1-to-1 assignment using Hungarian Algorithm.
        Maximizes total similarity (minimizes cost = 1 - sim).
        Returns list of (row_idx, col_idx, similarity_score)
        """
        if sim_matrix.size == 0:
            return []
        cost = 1.0 - sim_matrix
        row_idx, col_idx = linear_sum_assignment(cost)
        return [(int(r), int(c), float(sim_matrix[r, c]))
                for r, c in zip(row_idx, col_idx)]


# ─────────────────────────────────────────────────────────────
# LABEL LOADER
# ─────────────────────────────────────────────────────────────
def load_yolo_labels(label_path: str, img_w: int, img_h: int,
                     class_names: dict = None) -> list:
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            labels.append({
                "class_id"  : cls_id,
                "bbox"      : [x1, y1, x2, y2],
                "label_name": (class_names or {}).get(cls_id, f"cls_{cls_id}")
            })
    return labels


# ─────────────────────────────────────────────────────────────
# SPATIAL UTILITIES
# ─────────────────────────────────────────────────────────────
def intersection_over_ppe(person_box, ppe_box) -> float:
    xA = max(person_box[0], ppe_box[0])
    yA = max(person_box[1], ppe_box[1])
    xB = min(person_box[2], ppe_box[2])
    yB = min(person_box[3], ppe_box[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    ppe_area = (ppe_box[2]-ppe_box[0]) * (ppe_box[3]-ppe_box[1])
    return inter / ppe_area if ppe_area > 0 else 0.0

def compute_iou(a, b) -> float:
    xA, yA = max(a[0],b[0]), max(a[1],b[1])
    xB, yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    aA = (a[2]-a[0])*(a[3]-a[1])
    aB = (b[2]-b[0])*(b[3]-b[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# CORE COMPARATOR
# ─────────────────────────────────────────────────────────────
class CNNComparator:
    def __init__(self, embedder: CNNEmbedder):
        self.embedder = embedder
        self.sim_calc = SimilarityCalculator()

    def process_image(self, image: np.ndarray,
                      ppe_labels: list, person_labels: list) -> dict:
        """
        Full comparison pipeline for one image.
        Returns structured result dict.
        """
        # Split PPE labels by class
        ppe_persons  = [l for l in ppe_labels if l["class_id"] == Config.PPE_PERSON_CLASS_ID]
        helmets      = [l for l in ppe_labels if l["class_id"] == Config.PPE_HELMET_CLASS_ID]
        coveralls    = [l for l in ppe_labels if l["class_id"] == Config.PPE_COVERALL_CLASS_ID]
        det_persons  = [l for l in person_labels if l["class_id"] == Config.PERSON_CLASS_ID]

        print(f"\n    PPE model   → person:{len(ppe_persons)} helmet:{len(helmets)} coverall:{len(coveralls)}")
        print(f"    Person model → person:{len(det_persons)}")

        # ── Extract embeddings (batched) ──────────────────────────────
        print("    Extracting CNN embeddings...")
        ppe_person_embs  = self.embedder.extract_batch(image, [p["bbox"] for p in ppe_persons])
        det_person_embs  = self.embedder.extract_batch(image, [p["bbox"] for p in det_persons])
        helmet_embs      = self.embedder.extract_batch(image, [h["bbox"] for h in helmets])
        coverall_embs    = self.embedder.extract_batch(image, [c["bbox"] for c in coveralls])

        # ── Cross-model person identity check ─────────────────────────
        cross_cos_matrix = np.zeros((len(ppe_persons), len(det_persons)))
        cross_euc_matrix = np.zeros((len(ppe_persons), len(det_persons)))
        cross_combined   = np.zeros((len(ppe_persons), len(det_persons)))
        person_matches   = []

        if len(ppe_persons) > 0 and len(det_persons) > 0:
            cross_cos_matrix = self.sim_calc.cosine_matrix(ppe_person_embs, det_person_embs)
            cross_euc_matrix = self.sim_calc.euclidean_matrix(ppe_person_embs, det_person_embs)
            cross_combined   = self.sim_calc.combined_matrix(ppe_person_embs, det_person_embs)

            assignments = self.sim_calc.hungarian_assign(cross_combined)
            for (r, c, score) in assignments:
                cos_score = float(cross_cos_matrix[r, c])
                euc_score = float(cross_euc_matrix[r, c])
                iou_score = compute_iou(ppe_persons[r]["bbox"], det_persons[c]["bbox"])

                # Match decision: both cosine AND euclidean must agree
                cos_match = cos_score >= Config.COSINE_THRESHOLD
                euc_match = euc_score >= (1.0 / (1.0 + Config.EUCLIDEAN_THRESHOLD))
                is_match  = cos_match and euc_match

                person_matches.append({
                    "ppe_person_idx"    : r,
                    "det_person_idx"    : c,
                    "ppe_person_bbox"   : ppe_persons[r]["bbox"],
                    "det_person_bbox"   : det_persons[c]["bbox"],
                    "cosine_similarity" : round(cos_score, 4),
                    "euclidean_similarity": round(euc_score, 4),
                    "combined_score"    : round(score, 4),
                    "bbox_iou"          : round(iou_score, 4),
                    "cosine_match"      : cos_match,
                    "euclidean_match"   : euc_match,
                    "is_match"          : is_match,
                    "verdict"           : "✅ MATCH" if is_match else "❌ MISMATCH"
                })

        # ── PPE → Person assignment ───────────────────────────────────
        def assign_ppe(ppe_items, ppe_embs, ppe_type):
            results = []
            if not ppe_items or len(ppe_persons) == 0:
                return results

            # CNN embedding similarity: [num_persons × num_ppe]
            emb_sim = self.sim_calc.cosine_matrix(ppe_person_embs, ppe_embs)

            # Spatial IoP: [num_persons × num_ppe]
            spatial = np.array([
                [intersection_over_ppe(p["bbox"], e["bbox"]) for e in ppe_items]
                for p in ppe_persons
            ])

            # Hybrid score
            hybrid = Config.ALPHA_SPATIAL * spatial + Config.ALPHA_EMBEDDING * emb_sim

            assignments = self.sim_calc.hungarian_assign(hybrid)
            for (person_i, ppe_i, score) in assignments:
                results.append({
                    "ppe_type"       : ppe_type,
                    "ppe_bbox"       : ppe_items[ppe_i]["bbox"],
                    "person_idx"     : person_i,
                    "person_bbox"    : ppe_persons[person_i]["bbox"],
                    "hybrid_score"   : round(score, 4),
                    "spatial_score"  : round(float(spatial[person_i, ppe_i]), 4),
                    "cnn_emb_score"  : round(float(emb_sim[person_i, ppe_i]), 4),
                    "is_valid"       : score >= Config.COSINE_THRESHOLD * 0.8,
                })
            return results

        ppe_assignments = {
            "helmets"  : assign_ppe(helmets,   helmet_embs,   "helmet"),
            "coveralls": assign_ppe(coveralls, coverall_embs, "coverall"),
        }

        return {
            "person_matches"   : person_matches,
            "ppe_assignments"  : ppe_assignments,
            "matrices": {
                "cosine"    : cross_cos_matrix.tolist(),
                "euclidean" : cross_euc_matrix.tolist(),
                "combined"  : cross_combined.tolist(),
            },
            "embeddings": {
                "ppe_persons" : ppe_person_embs.tolist(),
                "det_persons" : det_person_embs.tolist(),
                "helmets"     : helmet_embs.tolist(),
                "coveralls"   : coverall_embs.tolist(),
            },
            "counts": {
                "ppe_persons": len(ppe_persons),
                "det_persons": len(det_persons),
                "helmets"    : len(helmets),
                "coveralls"  : len(coveralls),
            },
            "labels": {
                "ppe_persons": ppe_persons,
                "det_persons": det_persons,
                "helmets"    : helmets,
                "coveralls"  : coveralls,
            }
        }


# ─────────────────────────────────────────────────────────────
# VISUALIZER
# ─────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "person"   : "#00E676",
    "helmet"   : "#FFD600",
    "coverall" : "#FF6D00",
    "det_person": "#2196F3",
}

def draw_boxes(ax, labels, color_map, title):
    for lbl in labels:
        x1,y1,x2,y2 = lbl["bbox"]
        color = color_map.get(lbl.get("label_name", "person"), "#FFFFFF")
        rect = patches.Rectangle(
            (x1,y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(x1+2, y1-6, lbl.get("label_name","?"),
                color=color, fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")


def visualize_comparison(image, result, save_path):
    fig = plt.figure(figsize=(24, 14))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labels = result["labels"]

    # Panel 1: PPE model detections
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    all_ppe = labels["ppe_persons"] + labels["helmets"] + labels["coveralls"]
    draw_boxes(ax1, all_ppe, LABEL_COLORS, "PPE Model Detections")

    # Panel 2: Person model detections
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb)
    det_lbs = [dict(l, label_name="det_person") for l in labels["det_persons"]]
    draw_boxes(ax2, det_lbs, LABEL_COLORS, "Person Model Detections")

    # Panel 3: Cross-model match overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgb)
    for m in result["person_matches"]:
        color = "#00FF00" if m["is_match"] else "#FF1744"
        for bbox, style in [(m["ppe_person_bbox"], "-"), (m["det_person_bbox"], "--")]:
            x1,y1,x2,y2 = bbox
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor=color,
                                       facecolor="none", linestyle=style)
            ax3.add_patch(rect)
        mx = (m["ppe_person_bbox"][0] + m["det_person_bbox"][0]) // 2
        my = min(m["ppe_person_bbox"][1], m["det_person_bbox"][1]) - 10
        ax3.text(mx, my,
                 f"cos={m['cosine_similarity']:.2f}\neuc={m['euclidean_similarity']:.2f}",
                 color=color, fontsize=7, fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    ax3.set_title("Cross-Model Person Match", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # Panel 4: PPE assignments
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(rgb)
    for ppe_type, color in [("helmets","#FFD600"), ("coveralls","#FF6D00")]:
        for a in result["ppe_assignments"][ppe_type]:
            if not a["is_valid"]: continue
            px1,py1,px2,py2 = a["person_bbox"]
            ex1,ey1,ex2,ey2 = a["ppe_bbox"]
            ax4.plot([px1+(px2-px1)//2, ex1+(ex2-ex1)//2],
                     [py1+(py2-py1)//2, ey1+(ey2-ey1)//2],
                     color=color, linewidth=1.5, alpha=0.8)
            rect = patches.Rectangle((ex1,ey1), ex2-ex1, ey2-ey1,
                                       linewidth=2, edgecolor=color, facecolor="none")
            ax4.add_patch(rect)
    ax4.set_title("PPE → Person Assignment", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # Panel 5 & 6: Similarity matrices
    def plot_heatmap(ax, matrix, title, xlabel, ylabel):
        if not matrix or not matrix[0]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title); return
        mat = np.array(matrix)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.2 < mat[i,j] < 0.8 else "white")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels([f"Det-P{i}" for i in range(mat.shape[1])], fontsize=7)
        ax.set_yticklabels([f"PPE-P{i}" for i in range(mat.shape[0])], fontsize=7)

    ax5 = fig.add_subplot(gs[1, 0])
    plot_heatmap(ax5, result["matrices"]["cosine"],
                 "Cosine Similarity Matrix", "Person Model", "PPE Model")

    ax6 = fig.add_subplot(gs[1, 1])
    plot_heatmap(ax6, result["matrices"]["euclidean"],
                 "Euclidean Similarity Matrix", "Person Model", "PPE Model")

    ax7 = fig.add_subplot(gs[1, 2])
    plot_heatmap(ax7, result["matrices"]["combined"],
                 "Combined Similarity Matrix", "Person Model", "PPE Model")

    # Panel 8: Score summary bar chart
    ax8 = fig.add_subplot(gs[1, 3])
    if result["person_matches"]:
        idxs     = [f"P{m['ppe_person_idx']}↔P{m['det_person_idx']}" for m in result["person_matches"]]
        cos_vals = [m["cosine_similarity"]    for m in result["person_matches"]]
        euc_vals = [m["euclidean_similarity"] for m in result["person_matches"]]
        x = np.arange(len(idxs))
        w = 0.35
        bars1 = ax8.bar(x - w/2, cos_vals, w, label="Cosine",    color="#4CAF50", alpha=0.85)
        bars2 = ax8.bar(x + w/2, euc_vals, w, label="Euclidean", color="#2196F3", alpha=0.85)
        ax8.axhline(Config.COSINE_THRESHOLD, color="red", linestyle="--",
                    linewidth=1, label=f"Threshold ({Config.COSINE_THRESHOLD})")
        ax8.set_xticks(x); ax8.set_xticklabels(idxs, rotation=30, ha="right", fontsize=7)
        ax8.set_ylim(0, 1.1); ax8.set_ylabel("Similarity Score")
        ax8.set_title("Per-Pair Similarity Scores", fontsize=10, fontweight="bold")
        ax8.legend(fontsize=8)
    else:
        ax8.text(0.5, 0.5, "No matches found", ha="center", va="center", transform=ax8.transAxes)
        ax8.set_title("Per-Pair Scores")

    plt.suptitle("CNN Embedding Comparison — PPE vs Person Model",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved visualization → {save_path}")


# ─────────────────────────────────────────────────────────────
# t-SNE VISUALIZER
# ─────────────────────────────────────────────────────────────
def plot_tsne(all_embeddings: dict, save_path: str):
    """
    Projects all embeddings from all images to 2D using t-SNE.
    Good clusters = embeddings from both models represent the same person
    in nearby regions of the space.

    Legend:
        ● PPE-person     (green)
        ● Det-person     (blue)
        ▲ Helmet         (gold)
        ■ Coverall       (orange)
    """
    vectors, labels, markers, colors_list = [], [], [], []
    color_map = {
        "ppe_person":  "#00E676",
        "det_person":  "#2196F3",
        "helmet":      "#FFD600",
        "coverall":    "#FF6D00",
    }
    marker_map = {
        "ppe_person": "o", "det_person": "s",
        "helmet": "^",     "coverall":   "D",
    }

    for tag, emb_list in all_embeddings.items():
        for emb in emb_list:
            if np.any(emb):  # skip zero vectors
                vectors.append(emb)
                labels.append(tag)
                colors_list.append(color_map[tag])
                markers.append(marker_map[tag])

    if len(vectors) < 4:
        print("  ⚠️  Too few embeddings for t-SNE (need ≥ 4)")
        return

    print(f"  Running t-SNE on {len(vectors)} embeddings...")
    perp = min(30, len(vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    proj = tsne.fit_transform(np.array(vectors))

    fig, ax = plt.subplots(figsize=(10, 8))
    for tag in color_map:
        idxs = [i for i, l in enumerate(labels) if l == tag]
        if idxs:
            ax.scatter(proj[idxs, 0], proj[idxs, 1],
                       c=color_map[tag], marker=marker_map[tag],
                       s=80, alpha=0.85, label=tag, edgecolors="black", linewidths=0.5)

    ax.set_title("t-SNE: CNN Embedding Space\n(PPE model vs Person model)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved t-SNE plot → {save_path}")


# ─────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────
def print_report(image_name: str, result: dict):
    print(f"\n  {'═'*55}")
    print(f"  REPORT: {image_name}")
    print(f"  {'═'*55}")
    c = result["counts"]
    print(f"  PPE model   → persons:{c['ppe_persons']} helmets:{c['helmets']} coveralls:{c['coveralls']}")
    print(f"  Person model → persons:{c['det_persons']}")

    print(f"\n  [CROSS-MODEL PERSON IDENTITY]")
    if not result["person_matches"]:
        print("  ⚠️  No person pairs to compare")
    for m in result["person_matches"]:
        print(f"\n  PPE-Person[{m['ppe_person_idx']}] ↔ Det-Person[{m['det_person_idx']}]")
        print(f"    Cosine similarity    : {m['cosine_similarity']}  {'✅' if m['cosine_match'] else '❌'}")
        print(f"    Euclidean similarity : {m['euclidean_similarity']}  {'✅' if m['euclidean_match'] else '❌'}")
        print(f"    Combined score       : {m['combined_score']}")
        print(f"    BBox IoU             : {m['bbox_iou']}")
        print(f"    Verdict              : {m['verdict']}")

    print(f"\n  [PPE ASSIGNMENT]")
    for ppe_type, items in result["ppe_assignments"].items():
        for a in items:
            st = "✅" if a["is_valid"] else "⚠️"
            print(f"  {st} {a['ppe_type']} → Person[{a['person_idx']}]  "
                  f"hybrid={a['hybrid_score']}  spatial={a['spatial_score']}  cnn={a['cnn_emb_score']}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'═'*55}")
    print(f"  CNN/ResNet Embedding Comparator")
    print(f"{'═'*55}")
    print(f"  Backbone : {Config.BACKBONE}")
    print(f"  Method   : {Config.SIMILARITY_METHOD}")
    print(f"  Device   : {device}")

    # Load embedder
    embedder   = CNNEmbedder(backbone_name=Config.BACKBONE,
                              use_siamese_head=False, device=device)
    comparator = CNNComparator(embedder)

    # Collect images
    img_dir  = Path(Config.TEST_IMAGES_DIR)
    lbl_dir  = Path(Config.LABELS_DIR)
    exts     = {".jpg", ".jpeg", ".png", ".bmp"}
    images   = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])

    if not images:
        print(f"  ⚠️  No images found in {Config.TEST_IMAGES_DIR}")
        return

    all_results  = {}
    global_stats = defaultdict(list)
    # For t-SNE: collect all embeddings across images
    tsne_pool = defaultdict(list)

    for img_path in images:
        print(f"\n  Processing: {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Cannot read {img_path.name}")
            continue
        h, w = image.shape[:2]

        ppe_labels    = load_yolo_labels(
            str(lbl_dir / "ppe"    / (img_path.stem + ".txt")), w, h, Config.PPE_CLASS_NAMES)
        person_labels = load_yolo_labels(
            str(lbl_dir / "person" / (img_path.stem + ".txt")), w, h, {0: "det_person"})

        result = comparator.process_image(image, ppe_labels, person_labels)
        all_results[img_path.name] = {
            k: v for k, v in result.items() if k != "embeddings"
        }

        print_report(img_path.name, result)

        # Visualize
        vis_path = os.path.join(Config.OUTPUT_DIR, f"{img_path.stem}_cnn_comparison.png")
        visualize_comparison(image, result, vis_path)

        # Accumulate for global stats + t-SNE
        for m in result["person_matches"]:
            global_stats["cosine"].append(m["cosine_similarity"])
            global_stats["euclidean"].append(m["euclidean_similarity"])
            global_stats["combined"].append(m["combined_score"])
            global_stats["is_match"].append(int(m["is_match"]))

        embs = result["embeddings"]
        for e in embs["ppe_persons"]:  tsne_pool["ppe_person"].append(e)
        for e in embs["det_persons"]:  tsne_pool["det_person"].append(e)
        for e in embs["helmets"]:      tsne_pool["helmet"].append(e)
        for e in embs["coveralls"]:    tsne_pool["coverall"].append(e)

    # ── Global Summary ────────────────────────────────────────
    if global_stats["cosine"]:
        cos  = np.array(global_stats["cosine"])
        euc  = np.array(global_stats["euclidean"])
        comb = np.array(global_stats["combined"])
        mtch = np.array(global_stats["is_match"])

        print(f"\n{'═'*55}")
        print(f"  GLOBAL CNN SIMILARITY SUMMARY")
        print(f"{'═'*55}")
        print(f"  Total person pairs   : {len(cos)}")
        print(f"  Match rate           : {mtch.mean()*100:.1f}%")
        print(f"  Cosine  avg ± std    : {cos.mean():.4f} ± {cos.std():.4f}")
        print(f"  Euclidean sim avg    : {euc.mean():.4f} ± {euc.std():.4f}")
        print(f"  Combined avg         : {comb.mean():.4f} ± {comb.std():.4f}")

        # Distribution plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, vals, label, color in zip(
            axes,
            [cos, euc, comb],
            ["Cosine Similarity", "Euclidean Similarity", "Combined Score"],
            ["#4CAF50", "#2196F3", "#9C27B0"]
        ):
            ax.hist(vals, bins=20, color=color, edgecolor="black", alpha=0.8)
            ax.axvline(Config.COSINE_THRESHOLD, color="red", linestyle="--",
                       label=f"Threshold ({Config.COSINE_THRESHOLD})")
            ax.set_xlabel(label); ax.set_ylabel("Count")
            ax.set_title(f"{label} Distribution")
            ax.legend(fontsize=8)
        plt.suptitle(f"CNN ({Config.BACKBONE}) Similarity Distributions",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        dist_path = os.path.join(Config.OUTPUT_DIR, "cnn_similarity_distributions.png")
        plt.savefig(dist_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved distributions → {dist_path}")

    # t-SNE
    if Config.RUN_TSNE:
        tsne_path = os.path.join(Config.OUTPUT_DIR, "tsne_embedding_space.png")
        plot_tsne(tsne_pool, tsne_path)

    # Save JSON
    json_path = os.path.join(Config.OUTPUT_DIR, "cnn_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved JSON results   → {json_path}")
    print(f"\n✅ Done. All outputs → {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
