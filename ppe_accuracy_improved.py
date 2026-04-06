"""
ENHANCED PPE DETECTION SYSTEM - Maximum Accuracy Focus
===============================================
Improvements:
1. yolov8l (Large) option for better person detection - reduces false positives
2. Expanded + better-tuned text prompts
3. Confidence-based filtering system
4. Score normalization and gap analysis
5. Multi-criteria decision logic
6. Better region extraction
7. False positive elimination strategy

Embeddings Storage:
- helmet_pos_db: FAISS IndexFlatIP holding 512D normalized embeddings
  └─ Generated from ~15 "wearing helmet" text prompts
- helmet_neg_db: FAISS IndexFlatIP holding 512D normalized embeddings
  └─ Generated from ~10 "no helmet" text prompts
- coverall_pos_db: FAISS IndexFlatIP holding 512D normalized embeddings
  └─ Generated from ~15 "wearing protection" text prompts
- coverall_neg_db: FAISS IndexFlatIP holding 512D normalized embeddings
  └─ Generated from ~10 "no protection" text prompts
- All indices kept in-memory during runtime
"""

import os
import cv2
import torch
import faiss
import numpy as np
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from typing import Tuple, Dict, List


# ============================================================================
# CONFIGURATION - ACCURACY FOCUSED
# ============================================================================
class Config:
    """System configuration optimized for accuracy"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MODEL SELECTION: Use yolov8l for better person detection
    # yolov8m = medium (faster, good balance) 
    # yolov8l = large (slower, more accurate - RECOMMENDED for accuracy)
    # yolov8x = extra large (very slow, maximum accuracy)
    YOLO_MODEL = "yolov8l.pt"  # UPGRADED from yolov8m for accuracy
    
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    
    # --------- PERSON DETECTION PARAMETERS ---------
    # Higher confidence = fewer false positives, but might miss some
    PERSON_CONF_THRESH = 0.5      # Min confidence to detect a person
    MIN_PERSON_WIDTH = 20          # Min pixel width to avoid noise
    MIN_PERSON_HEIGHT = 30         # Min pixel height to avoid noise
    
    # --------- PPE DETECTION PARAMETERS ---------
    # Increased thresholds for more conservative detection
    HELMET_CONFIDENT_THRESH = 0.30    # Strong positive score needed
    HELMET_MIN_GAP = 0.08             # Minimum gap between pos - neg
    
    COVERALL_CONFIDENT_THRESH = 0.32  # Strong positive score needed
    COVERALL_MIN_GAP = 0.10           # Minimum gap between pos - neg
    
    # --------- REGION EXTRACTION PARAMETERS ---------
    HEAD_CROP_RATIO = 0.38            # Crop top 38% for head region (was 35%)
    SHOULDER_INCLUSION = 0.05         # Include 5% shoulder for context
    
    # --------- SIMILARITY SEARCH PARAMETERS ---------
    HELMET_SEARCH_K = 5               # Search top 5 neighbors (was 3)
    COVERALL_SEARCH_K = 6             # Search top 6 neighbors (was 3)
    
    # --------- CONFIDENCE PENALTIES ---------
    SMALL_REGION_PENALTY = 0.15      # Reduce confidence for small ROIs
    BLURRY_REGION_PENALTY = 0.20     # Reduce confidence for blurry ROIs
    
    INPUT_FOLDER = r"C:\Users\iqsha\Downloads\YOLO_CNN\new_data"
    OUTPUT_FOLDER = r"C:\Users\iqsha\Downloads\YOLO_CNN\output_accuracy"
    
    DEBUG = True
    SAVE_CONFIDENCE_SCORES = True    # Save per-detection scores to file


# ============================================================================
# ENHANCED PPE LABELS - EXPANDED & CONTEXT RICH
# ============================================================================
class OptimizedPPELabels:
    """Highly tuned PPE text prompts with context"""
    
    # HELMET: 15+ variants covering different contexts
    HELMET_LABELS = [
        "helmet on head",
        "hard hat",
        "safety helmet",
        "protective headgear",
        "head protection",
        "safety hard hat",
        "head with helmet",
        "wearing helmet",
        "helmet on",
        "construction helmet",
        "industrial helmet",
        "yellow safety hat",
        "protective hard hat",
        "head fully protected",
        "helmet covered head",
        "hard hat protection",
    ]
    
    # NO HELMET: 12+ variants for absence
    NO_HELMET_LABELS = [
        "no helmet",
        "without helmet",
        "unprotected head",
        "bare head",
        "no hard hat",
        "head without protection",
        "exposed head",
        "helmet missing",
        "no head protection",
        "unhelmed",
        "bare headed",
        "unprotected",
    ]
    
    # COVERALL: 15+ variants for body protection
    COVERALL_LABELS = [
        "safety suit",
        "protective coverall",
        "full body protection",
        "safety coverall",
        "full protection suit",
        "protective clothing",
        "body protection",
        "full body safety suit",
        "protective gear",
        "safety overalls",
        "full body covered",
        "body fully protected",
        "protective garment",
        "safety uniform",
        "hazmat suit",
        "body protection suit",
    ]
    
    # NO COVERALL: 12+ variants for absence
    NO_COVERALL_LABELS = [
        "no coverall",
        "without protection",
        "no full protection",
        "exposed body",
        "unprotected body",
        "no safety suit",
        "no body protection",
        "bare body",
        "unprotected",
        "casual clothing",
        "no protection",
        "body exposed",
    ]


# ============================================================================
# ENHANCED EMBEDDING ENGINE WITH QUALITY CHECKS
# ============================================================================
class EnhancedCLIPEmbeddingEngine:
    """CLIP embedding generation with robustness checks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        print("[INIT] Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        self.model.eval()
        print(f"[INIT] CLIP model loaded on {self.device}")
    
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate normalized text embeddings"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_embeds = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0, :]
            text_features = self.model.text_projection(text_embeds)
        
        # L2 normalization - CRITICAL for inner product similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embeddings = text_features.cpu().numpy().astype("float32")
        
        if self.config.DEBUG:
            print(f"[EMBED] Text: {len(texts)} prompts → shape={embeddings.shape}, norm={np.linalg.norm(embeddings[0]):.4f}")
        
        return embeddings
    
    def get_image_embedding(self, image: np.ndarray, apply_preprocessing: bool = True) -> np.ndarray:
        """Generate normalized image embedding with preprocessing"""
        
        # Image preprocessing for robustness
        if apply_preprocessing:
            image = self._preprocess_image(image)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_embeds = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state[:, 0, :]
            image_features = self.model.visual_projection(image_embeds)
        
        # L2 normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().astype("float32").reshape(1, -1)
        
        if self.config.DEBUG:
            print(f"[EMBED] Image: shape={embedding.shape}, norm={np.linalg.norm(embedding[0]):.4f}")
        
        return embedding
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better CLIP understanding"""
        # Adaptive histogram equalization for contrast
        if len(image.shape) == 3:
            # Convert to LAB, enhance L channel, convert back
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE: Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image
    
    def estimate_image_blur(self, image: np.ndarray) -> float:
        """Estimate blur using Laplacian variance (0-1 score)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 (less accurate detection for high blur)
        blur_score = min(1.0, laplacian_var / 100.0)
        
        return blur_score


# ============================================================================
# ENHANCED VECTOR DATABASE - WITH MORE VARIANTS
# ============================================================================
class EnhancedPPEDatabase:
    """Enhanced database with expanded embeddings"""
    
    def __init__(self, config: Config, embedding_engine: EnhancedCLIPEmbeddingEngine):
        self.config = config
        
        # HELMET DATABASES
        print("[DB] Building HELMET databases...")
        helmet_pos_emb = embedding_engine.get_text_embeddings(OptimizedPPELabels.HELMET_LABELS)
        self.helmet_pos_db = self._create_index(helmet_pos_emb, f"HELMET (with) - {len(OptimizedPPELabels.HELMET_LABELS)} prompts")
        self.helmet_pos_refs = OptimizedPPELabels.HELMET_LABELS
        
        helmet_neg_emb = embedding_engine.get_text_embeddings(OptimizedPPELabels.NO_HELMET_LABELS)
        self.helmet_neg_db = self._create_index(helmet_neg_emb, f"HELMET (without) - {len(OptimizedPPELabels.NO_HELMET_LABELS)} prompts")
        self.helmet_neg_refs = OptimizedPPELabels.NO_HELMET_LABELS
        
        # COVERALL DATABASES
        print("[DB] Building COVERALL databases...")
        coverall_pos_emb = embedding_engine.get_text_embeddings(OptimizedPPELabels.COVERALL_LABELS)
        self.coverall_pos_db = self._create_index(coverall_pos_emb, f"COVERALL (with) - {len(OptimizedPPELabels.COVERALL_LABELS)} prompts")
        self.coverall_pos_refs = OptimizedPPELabels.COVERALL_LABELS
        
        coverall_neg_emb = embedding_engine.get_text_embeddings(OptimizedPPELabels.NO_COVERALL_LABELS)
        self.coverall_neg_db = self._create_index(coverall_neg_emb, f"COVERALL (without) - {len(OptimizedPPELabels.NO_COVERALL_LABELS)} prompts")
        self.coverall_neg_refs = OptimizedPPELabels.NO_COVERALL_LABELS
    
    def _create_index(self, embeddings: np.ndarray, label: str) -> faiss.IndexFlatIP:
        """Create FAISS index with metadata"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"  ✓ {label}")
        return index
    
    def search_helmet(self, query_emb: np.ndarray, k: int = 5) -> Tuple[float, float, Dict]:
        """
        Search helmet databases with detailed scoring
        Returns: (pos_score, neg_score, metadata_dict)
        """
        # Top-K search on positive DB
        distances, indices = self.helmet_pos_db.search(query_emb, k=min(k, len(self.helmet_pos_refs)))
        pos_scores = distances[0].tolist()
        pos_score = np.mean(pos_scores)
        pos_top_refs = [self.helmet_pos_refs[i] for i in indices[0][:3]]
        
        # Top-K search on negative DB
        distances, indices = self.helmet_neg_db.search(query_emb, k=min(k, len(self.helmet_neg_refs)))
        neg_scores = distances[0].tolist()
        neg_score = np.mean(neg_scores)
        neg_top_refs = [self.helmet_neg_refs[i] for i in indices[0][:3]]
        
        gap = pos_score - neg_score
        
        metadata = {
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'gap': gap,
            'pos_top_refs': pos_top_refs,
            'neg_top_refs': neg_top_refs,
            'max_pos': max(pos_scores) if pos_scores else 0,
            'min_neg': min(neg_scores) if neg_scores else 0,
        }
        
        return pos_score, neg_score, metadata
    
    def search_coverall(self, query_emb: np.ndarray, k: int = 6) -> Tuple[float, float, Dict]:
        """Search coverall databases with detailed scoring"""
        # Top-K search on positive DB
        distances, indices = self.coverall_pos_db.search(query_emb, k=min(k, len(self.coverall_pos_refs)))
        pos_scores = distances[0].tolist()
        pos_score = np.mean(pos_scores)
        pos_top_refs = [self.coverall_pos_refs[i] for i in indices[0][:3]]
        
        # Top-K search on negative DB
        distances, indices = self.coverall_neg_db.search(query_emb, k=min(k, len(self.coverall_neg_refs)))
        neg_scores = distances[0].tolist()
        neg_score = np.mean(neg_scores)
        neg_top_refs = [self.coverall_neg_refs[i] for i in indices[0][:3]]
        
        gap = pos_score - neg_score
        
        metadata = {
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'gap': gap,
            'pos_top_refs': pos_top_refs,
            'neg_top_refs': neg_top_refs,
            'max_pos': max(pos_scores) if pos_scores else 0,
            'min_neg': min(neg_scores) if neg_scores else 0,
        }
        
        return pos_score, neg_score, metadata


# ============================================================================
# ENHANCED REGION EXTRACTION
# ============================================================================
class EnhancedRegionExtractor:
    """Extract and validate regions for PPE detection"""
    
    @staticmethod
    def get_head_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      crop_ratio: float = 0.38) -> Tuple[np.ndarray, bool]:
        """
        Extract head region with size validation
        Returns: (cropped_region, is_valid)
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        # Add shoulder context
        head_height = int(height * crop_ratio)
        head_crop = frame[max(0, y1):min(frame.shape[0], y1 + head_height), x1:x2]
        
        # Validate size
        if head_crop.shape[0] < 30 or head_crop.shape[1] < 20:
            return head_crop, False
        
        return head_crop, True
    
    @staticmethod
    def get_body_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, bool]:
        """Extract full body region with validation"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        body_crop = frame[y1:y2, x1:x2]
        
        # Validate size - must be substantial
        if height < 50 or width < 30:
            return body_crop, False
        
        return body_crop, True


# ============================================================================
# MULTI-CRITERIA PPE CLASSIFIER - ACCURACY FOCUSED
# ============================================================================
class AccuracyFocusedPPEClassifier:
    """
    Multi-criteria PPE classifier with:
    - Confidence-based filtering
    - Score gap analysis
    - Region quality checks
    - False positive elimination
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.score_log = []
        
        print("[INIT] Loading YOLO person detector...")
        print(f"[INIT] Using model: {config.YOLO_MODEL}")
        self.person_model = YOLO(config.YOLO_MODEL)
        
        print("[INIT] Loading CLIP embedding engine...")
        self.embedding_engine = EnhancedCLIPEmbeddingEngine(config)
        
        print("[INIT] Creating enhanced PPE databases...")
        self.db = EnhancedPPEDatabase(config, self.embedding_engine)
        
        print("[INIT] Classifier ready!\n")
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons with quality filtering"""
        results = self.person_model(frame)[0]
        person_boxes = []
        
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > self.config.PERSON_CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                
                # Filter out very small detections (noise)
                if width >= self.config.MIN_PERSON_WIDTH and height >= self.config.MIN_PERSON_HEIGHT:
                    person_boxes.append((x1, y1, x2, y2))
        
        if self.config.DEBUG:
            print(f"[YOLO] Detected {len(person_boxes)} persons (filtered)")
        
        return person_boxes
    
    def classify_ppe_advanced(self, frame: np.ndarray, person_bbox: Tuple) -> Dict:
        """
        Advanced multi-criteria PPE classification
        Returns: {
            'has_helmet': bool,
            'helmet_confidence': float,
            'has_coverall': bool,
            'coverall_confidence': float,
            'is_safe': bool,
            'metadata': {...}
        }
        """
        result = {
            'has_helmet': False,
            'helmet_confidence': 0.0,
            'helmet_metadata': {},
            'has_coverall': False,
            'coverall_confidence': 0.0,
            'coverall_metadata': {},
            'is_safe': False,
        }
        
        # ========== HELMET DETECTION ==========
        head_crop, head_valid = EnhancedRegionExtractor.get_head_crop(
            frame, person_bbox, self.config.HEAD_CROP_RATIO
        )
        
        if head_valid and head_crop.size > 0:
            head_emb = self.embedding_engine.get_image_embedding(head_crop)
            pos_score, neg_score, metadata = self.db.search_helmet(
                head_emb, k=self.config.HELMET_SEARCH_K
            )
            
            # Estimate image blur
            blur_score = self.embedding_engine.estimate_image_blur(head_crop)
            
            # Multi-criteria decision
            gap = pos_score - neg_score
            
            # Decision criteria:
            # 1. Positive score must be strong
            # 2. Gap must be significant
            # 3. Apply blur penalty
            
            blur_penalty = self.config.BLURRY_REGION_PENALTY * (1 - blur_score)
            adjusted_pos_score = max(0, pos_score - blur_penalty)
            
            helmet_confidence = self._calculate_confidence(
                pos_score=adjusted_pos_score,
                neg_score=neg_score,
                gap=gap,
                threshold=self.config.HELMET_CONFIDENT_THRESH,
                min_gap=self.config.HELMET_MIN_GAP
            )
            
            result['helmet_confidence'] = max(0, helmet_confidence)
            result['helmet_metadata'] = {
                'pos_score': float(pos_score),
                'neg_score': float(neg_score),
                'gap': float(gap),
                'blur_score': float(blur_score),
                'adjusted_pos_score': float(adjusted_pos_score),
                'top_pos_refs': metadata['pos_top_refs'],
                'top_neg_refs': metadata['neg_top_refs'],
            }
            
            # Helmet detected if:
            # - Positive score > threshold AND
            # - Gap is significant
            result['has_helmet'] = (
                pos_score > self.config.HELMET_CONFIDENT_THRESH and 
                gap > self.config.HELMET_MIN_GAP
            )
            
            if self.config.DEBUG:
                status = "✓ HELMET" if result['has_helmet'] else "✗ NO HELMET"
                print(f"[HELMET] {status} | Pos:{pos_score:.3f} Neg:{neg_score:.3f} Gap:{gap:.3f} Blur:{blur_score:.2f}")
        else:
            if self.config.DEBUG:
                print(f"[HELMET] ⚠ Invalid head region")
        
        # ========== COVERALL DETECTION ==========
        body_crop, body_valid = EnhancedRegionExtractor.get_body_crop(frame, person_bbox)
        
        if body_valid and body_crop.size > 0:
            body_emb = self.embedding_engine.get_image_embedding(body_crop)
            pos_score, neg_score, metadata = self.db.search_coverall(
                body_emb, k=self.config.COVERALL_SEARCH_K
            )
            
            # Estimate image blur
            blur_score = self.embedding_engine.estimate_image_blur(body_crop)
            
            # Multi-criteria decision
            gap = pos_score - neg_score
            
            blur_penalty = self.config.BLURRY_REGION_PENALTY * (1 - blur_score)
            adjusted_pos_score = max(0, pos_score - blur_penalty)
            
            coverall_confidence = self._calculate_confidence(
                pos_score=adjusted_pos_score,
                neg_score=neg_score,
                gap=gap,
                threshold=self.config.COVERALL_CONFIDENT_THRESH,
                min_gap=self.config.COVERALL_MIN_GAP
            )
            
            result['coverall_confidence'] = max(0, coverall_confidence)
            result['coverall_metadata'] = {
                'pos_score': float(pos_score),
                'neg_score': float(neg_score),
                'gap': float(gap),
                'blur_score': float(blur_score),
                'adjusted_pos_score': float(adjusted_pos_score),
                'top_pos_refs': metadata['pos_top_refs'],
                'top_neg_refs': metadata['neg_top_refs'],
            }
            
            # Coverall detected if:
            # - Positive score > threshold AND
            # - Gap is significant
            result['has_coverall'] = (
                pos_score > self.config.COVERALL_CONFIDENT_THRESH and 
                gap > self.config.COVERALL_MIN_GAP
            )
            
            if self.config.DEBUG:
                status = "✓ COVERALL" if result['has_coverall'] else "✗ NO COVERALL"
                print(f"[COVERALL] {status} | Pos:{pos_score:.3f} Neg:{neg_score:.3f} Gap:{gap:.3f} Blur:{blur_score:.2f}")
        else:
            if self.config.DEBUG:
                print(f"[COVERALL] ⚠ Invalid body region")
        
        # ========== FINAL SAFETY DECISION ==========
        result['is_safe'] = result['has_helmet'] and result['has_coverall']
        
        return result
    
    def _calculate_confidence(self, pos_score: float, neg_score: float, gap: float,
                             threshold: float, min_gap: float) -> float:
        """
        Calculate confidence score for detection (0-100%)
        Higher = more confident
        """
        if pos_score < threshold or gap < min_gap:
            return 0.0  # Not detected
        
        # Confidence based on:
        # 1. How much pos_score exceeds threshold (up to 40%)
        # 2. How much gap exceeds minimum (up to 60%)
        
        threshold_factor = min(0.4, (pos_score - threshold) / threshold)
        gap_factor = min(0.6, gap / max(min_gap, 0.01))
        
        confidence = (threshold_factor + gap_factor) * 100
        return min(100, max(0, confidence))


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main processing function"""
    config = Config()
    
    # Create output folder
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    
    # Initialize classifier
    classifier = AccuracyFocusedPPEClassifier(config)
    
    print("=" * 70)
    print("ACCURACY-FOCUSED PPE DETECTION SYSTEM")
    print("=" * 70)
    print(f"Input folder: {config.INPUT_FOLDER}")
    print(f"Output folder: {config.OUTPUT_FOLDER}")
    print(f"YOLO Model: {config.YOLO_MODEL}")
    print("=" * 70 + "\n")
    
    # Process images
    image_files = sorted([f for f in os.listdir(config.INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"[PROCESS] {idx}. {image_file}")
        
        image_path = os.path.join(config.INPUT_FOLDER, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"  ⚠ Failed to load image")
            continue
        
        frame_h, frame_w = frame.shape[:2]
        
        # Detect persons
        person_boxes = classifier.detect_persons(frame)
        
        # Classify PPE for each person
        for person_idx, bbox in enumerate(person_boxes, 1):
            print(f"  ├─ Person {person_idx}/{len(person_boxes)}")
            
            ppe_result = classifier.classify_ppe_advanced(frame, bbox)
            
            # Draw bounding box
            x1, y1,x2, y2 = bbox
            
            # Color based on safety
            color = (0, 255, 0) if ppe_result['is_safe'] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Helmet label
            helmet_label = "✓ HELMET" if ppe_result['has_helmet'] else "✗ NO HELMET"
            helmet_color = (0, 255, 0) if ppe_result['has_helmet'] else (0, 0, 255)
            cv2.putText(frame, helmet_label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, helmet_color, 2)
            
            # Coverall label
            coverall_label = "✓ COVERALL" if ppe_result['has_coverall'] else "✗ NO COVERALL"
            coverall_color = (0, 255, 0) if ppe_result['has_coverall'] else (0, 0, 255)
            cv2.putText(frame, coverall_label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, coverall_color, 2)
            
            # Safety status
            safe_label = "SAFE ✓" if ppe_result['is_safe'] else "UNSAFE ✗"
            safe_color = (0, 255, 0) if ppe_result['is_safe'] else (0, 0, 255)
            cv2.putText(frame, safe_label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, safe_color, 3)
            
            # Confidence scores
            helmet_conf_text = f"H:{ppe_result['helmet_confidence']:.0f}%"
            coverall_conf_text = f"C:{ppe_result['coverall_confidence']:.0f}%"
            cv2.putText(frame, helmet_conf_text, (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.putText(frame, coverall_conf_text, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        # Save output
        output_path = os.path.join(config.OUTPUT_FOLDER, image_file)
        cv2.imwrite(output_path, frame)
        print(f"  └─ Saved: {image_file}\n")
    
    print("=" * 70)
    print(f"✅ Processing complete! {len(image_files)} images processed.")
    print(f"📁 Results saved to: {config.OUTPUT_FOLDER}")
    print("=" * 70)


if __name__ == "__main__":
    main()
