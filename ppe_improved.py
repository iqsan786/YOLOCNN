"""
Improved CLIP PPE Detection - Separate Helmet & Coverall Embeddings
With optimized text prompts and aggressive detection
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
# CONFIGURATION
# ============================================================================
class Config:
    """System configuration"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_MODEL = "yolov8m.pt"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    
    PERSON_CONF_THRESH = 0.5
    
    # ⚠️ MUCH LOWER THRESHOLDS - Test phase
    HELMET_SIMILARITY_THRESH = 0.10      # Very lenient
    COVERALL_SIMILARITY_THRESH = 0.10    # Very lenient
    
    HEAD_CROP_RATIO = 0.35
    
    INPUT_FOLDER = r"C:\Users\iqsha\Downloads\YOLO_CNN\new_data"
    OUTPUT_FOLDER = r"C:\Users\iqsha\Downloads\YOLO_CNN\output_improved"
    
    DEBUG = True


# ============================================================================
# OPTIMIZED PPE LABELS
# ============================================================================
class PPELabels:
    """Highly optimized PPE text prompts for better CLIP matching"""
    
    # HELMET: Varied descriptions focusing on head protection
    HELMET_LABELS = [
        "helmet on head",
        "hard hat",
        "safety helmet",
        "protective headgear",
        "head protection",
        "safety hard hat",
        "head with helmet",
        "wearing helmet",
    ]
    
    # NO HELMET: Clear absence indicators
    NO_HELMET_LABELS = [
        "no helmet",
        "without helmet",
        "unprotected head",
        "bare head",
        "no hard hat",
        "head without protection",
    ]
    
    # COVERALL: Full body protection
    COVERALL_LABELS = [
        "safety suit",
        "protective coverall",
        "full body protection",
        "safety coverall",
        "full protection suit",
        "protective clothing",
        "body protection",
        "full body safety suit",
    ]
    
    # NO COVERALL: Absence indicators
    NO_COVERALL_LABELS = [
        "no coverall",
        "without protection",
        "no full protection",
        "exposed body",
        "unprotected body",
        "no safety suit",
    ]


# ============================================================================
# EMBEDDING ENGINE
# ============================================================================
class CLIPEmbeddingEngine:
    """CLIP embedding generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        print("[INIT] Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        self.model.to(self.device)
        self.model.eval()
        print(f"[INIT] CLIP model loaded on {self.device}")
    
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate normalized text embeddings"""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_embeds = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0, :]
            text_features = self.model.text_projection(text_embeds)
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embeddings = text_features.cpu().numpy().astype("float32")
        
        if self.config.DEBUG:
            print(f"[EMBED] Text embeddings: shape={embeddings.shape}")
        
        return embeddings
    
    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate normalized image embedding"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_embeds = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state[:, 0, :]
            image_features = self.model.visual_projection(image_embeds)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().astype("float32").reshape(1, -1)
        
        if self.config.DEBUG:
            print(f"[EMBED] Image embedding: shape={embedding.shape}")
        
        return embedding


# ============================================================================
# VECTOR DATABASE - SEPARATE POSITIVE & NEGATIVE
# ============================================================================
class SeparatePPEDatabase:
    """Separate databases for positive (wearing) and negative (not wearing) PPE"""
    
    def __init__(self, config: Config, embedding_engine: CLIPEmbeddingEngine):
        self.config = config
        
        print("[DB] Building HELMET databases...")
        # Positive: Wearing helmet
        helmet_pos_emb = embedding_engine.get_text_embeddings(PPELabels.HELMET_LABELS)
        self.helmet_pos_db = self._create_index(helmet_pos_emb, "HELMET (with)")
        
        # Negative: Not wearing helmet
        helmet_neg_emb = embedding_engine.get_text_embeddings(PPELabels.NO_HELMET_LABELS)
        self.helmet_neg_db = self._create_index(helmet_neg_emb, "HELMET (without)")
        
        print("[DB] Building COVERALL databases...")
        # Positive: Wearing coverall
        coverall_pos_emb = embedding_engine.get_text_embeddings(PPELabels.COVERALL_LABELS)
        self.coverall_pos_db = self._create_index(coverall_pos_emb, "COVERALL (with)")
        
        # Negative: Not wearing coverall
        coverall_neg_emb = embedding_engine.get_text_embeddings(PPELabels.NO_COVERALL_LABELS)
        self.coverall_neg_db = self._create_index(coverall_neg_emb, "COVERALL (without)")
    
    def _create_index(self, embeddings: np.ndarray, label: str) -> faiss.IndexFlatIP:
        """Create FAISS index"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"  ✓ {label}: {len(embeddings)} embeddings")
        return index
    
    def search_helmet(self, query_emb: np.ndarray, k: int = 3) -> Tuple[float, float]:
        """Search helmet databases, return (positive_score, negative_score)"""
        # Search positive DB
        _, pos_indices = self.helmet_pos_db.search(query_emb, k=min(k, 8))
        pos_score = np.mean([pos_indices[0][i] for i in range(min(k, len(pos_indices[0])))])
        
        # Search negative DB
        _, neg_indices = self.helmet_neg_db.search(query_emb, k=min(k, 6))
        neg_score = np.mean([neg_indices[0][i] for i in range(min(k, len(neg_indices[0])))])
        
        if self.config.DEBUG:
            print(f"[HELMET] Pos_score: {pos_score:.4f}, Neg_score: {neg_score:.4f}")
        
        return pos_score, neg_score
    
    def search_coverall(self, query_emb: np.ndarray, k: int = 3) -> Tuple[float, float]:
        """Search coverall databases, return (positive_score, negative_score)"""
        # Search positive DB
        _, pos_indices = self.coverall_pos_db.search(query_emb, k=min(k, 8))
        pos_score = np.mean([pos_indices[0][i] for i in range(min(k, len(pos_indices[0])))])
        
        # Search negative DB
        _, neg_indices = self.coverall_neg_db.search(query_emb, k=min(k, 6))
        neg_score = np.mean([neg_indices[0][i] for i in range(min(k, len(neg_indices[0])))])
        
        if self.config.DEBUG:
            print(f"[COVERALL] Pos_score: {pos_score:.4f}, Neg_score: {neg_score:.4f}")
        
        return pos_score, neg_score


# ============================================================================
# REGION EXTRACTION
# ============================================================================
class RegionExtractor:
    """Extract regions for PPE detection"""
    
    @staticmethod
    def get_head_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      crop_ratio: float = 0.35) -> np.ndarray:
        """Extract head region"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        head_height = int(height * crop_ratio)
        head_crop = frame[y1:y1 + head_height, x1:x2]
        return head_crop
    
    @staticmethod
    def get_body_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract full body region"""
        x1, y1, x2, y2 = bbox
        body_crop = frame[y1:y2, x1:x2]
        return body_crop


# ============================================================================
# IMPROVED PPE CLASSIFIER
# ============================================================================
class ImprovedPPEClassifier:
    """Classify PPE using separate positive/negative databases"""
    
    def __init__(self, config: Config):
        self.config = config
        
        print("[INIT] Loading YOLO person detector...")
        self.person_model = YOLO(config.YOLO_MODEL)
        
        print("[INIT] Loading CLIP embedding engine...")
        self.embedding_engine = CLIPEmbeddingEngine(config)
        
        print("[INIT] Creating separate PPE databases...")
        self.db = SeparatePPEDatabase(config, self.embedding_engine)
        
        print("[INIT] Classifier ready!")
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons"""
        results = self.person_model(frame)[0]
        person_boxes = []
        
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > self.config.PERSON_CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
        
        if self.config.DEBUG:
            print(f"[YOLO] Detected {len(person_boxes)} persons")
        
        return person_boxes
    
    def classify_ppe(self, frame: np.ndarray, person_bbox: Tuple) -> Dict[str, bool]:
        """
        Classify PPE using separate databases:
        1. Helmet: Compare with positive & negative helmet databases
        2. Coverall: Compare with positive & negative coverall databases
        """
        
        # -------- HELMET DETECTION --------
        has_helmet = False
        head_crop = RegionExtractor.get_head_crop(frame, person_bbox, self.config.HEAD_CROP_RATIO)
        
        if head_crop.size > 0:
            head_emb = self.embedding_engine.get_image_embedding(head_crop)
            pos_score, neg_score = self.db.search_helmet(head_emb, k=3)
            
            # Helmet detected if positive similarity > negative similarity
            if pos_score > neg_score:
                has_helmet = True
                if self.config.DEBUG:
                    print(f"  ✓ HELMET DETECTED (pos: {pos_score:.4f} > neg: {neg_score:.4f})")
            else:
                if self.config.DEBUG:
                    print(f"  ✗ NO HELMET (pos: {pos_score:.4f} <= neg: {neg_score:.4f})")
        
        # -------- COVERALL DETECTION --------
        has_coverall = False
        body_crop = RegionExtractor.get_body_crop(frame, person_bbox)
        
        if body_crop.size > 0:
            body_emb = self.embedding_engine.get_image_embedding(body_crop)
            pos_score, neg_score = self.db.search_coverall(body_emb, k=3)
            
            # Coverall detected if positive similarity > negative similarity
            if pos_score > neg_score:
                has_coverall = True
                if self.config.DEBUG:
                    print(f"  ✓ COVERALL DETECTED (pos: {pos_score:.4f} > neg: {neg_score:.4f})")
            else:
                if self.config.DEBUG:
                    print(f"  ✗ NO COVERALL (pos: {pos_score:.4f} <= neg: {neg_score:.4f})")
        
        return {
            "has_helmet": has_helmet,
            "has_coverall": has_coverall,
        }


# ============================================================================
# VISUALIZATION
# ============================================================================
class Visualizer:
    """Draw results"""
    
    @staticmethod
    def draw_result(frame: np.ndarray, bbox: Tuple, ppe_result: Dict) -> np.ndarray:
        """Draw safety classification"""
        x1, y1, x2, y2 = bbox
        
        has_helmet = ppe_result["has_helmet"]
        has_coverall = ppe_result["has_coverall"]
        
        is_safe = has_helmet and has_coverall
        color = (0, 255, 0) if is_safe else (0, 0, 255)
        status = "SAFE" if is_safe else "UNSAFE"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"H:{has_helmet} C:{has_coverall} {status}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    config = Config()
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    
    print("=" * 70)
    print("IMPROVED PPE DETECTION - Separate Positive/Negative Databases")
    print("=" * 70)
    
    classifier = ImprovedPPEClassifier(config)
    
    image_count = 0
    for filename in os.listdir(config.INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        image_count += 1
        image_path = os.path.join(config.INPUT_FOLDER, filename)
        frame = cv2.imread(image_path)
        
        if frame is None:
            continue
        
        output = frame.copy()
        
        print(f"\n[PROCESS] {image_count}. {filename}")
        
        person_boxes = classifier.detect_persons(frame)
        
        for person_idx, person_bbox in enumerate(person_boxes):
            print(f"  └─ Person {person_idx + 1}/{len(person_boxes)}")
            
            ppe_result = classifier.classify_ppe(frame, person_bbox)
            output = Visualizer.draw_result(output, person_bbox, ppe_result)
        
        output_path = os.path.join(config.OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, output)
        print(f"  ✅ Saved: {filename}")
    
    print("\n" + "=" * 70)
    print(f"🎉 Processing complete! {image_count} images processed.")
    print(f"📁 Results saved to: {config.OUTPUT_FOLDER}")
    print("=" * 70)


if __name__ == "__main__":
    main()
