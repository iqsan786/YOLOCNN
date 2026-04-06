"""
Production-Ready PPE Detection System
Hybrid Architecture: YOLO + CLIP + FAISS
Author: ML Engineering
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
    PPE_MODEL = r"C:\Users\iqsha\Downloads\YOLO_CNN\oldrig.pt"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    
    PERSON_CONF_THRESH = 0.5
    YOLO_HELMET_CONF_THRESH = 0.5
    YOLO_COVERALL_CONF_THRESH = 0.5
    CLIP_SIMILARITY_THRESH = 0.25
    
    # Region-specific thresholds
    HEAD_CROP_RATIO = 0.35  # Top 35% for helmet detection
    
    INPUT_FOLDER = r"C:\Users\iqsha\Downloads\YOLO_CNN\new_data"
    OUTPUT_FOLDER = r"C:\Users\iqsha\Downloads\YOLO_CNN\output"
    
    DEBUG = True


# ============================================================================
# PPE LABELS & PROMPTS
# ============================================================================
class PPELabels:
    """Optimized PPE description prompts"""
    HELMET_LABELS = [
        "a person wearing a safety helmet",
        "a worker with a hard hat",
        "person with protective headgear",
        "a person without a helmet",
    ]
    
    COVERALL_LABELS = [
        "a worker wearing a protective coverall",
        "a person in a safety suit",
        "worker wearing full-body protection",
        "a person without protective clothing",
    ]
    
    ALL_LABELS = HELMET_LABELS + COVERALL_LABELS


# ============================================================================
# EMBEDDING ENGINE
# ============================================================================
class CLIPEmbeddingEngine:
    """CLIP embedding generation with proper normalization"""
    
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
            print(f"[EMBED] Text embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        
        return embeddings
    
    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate normalized image embedding"""
        # Ensure BGR to RGB
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
            print(f"[EMBED] Image embedding: shape={embedding.shape}, dtype={embedding.dtype}")
        
        return embedding


# ============================================================================
# VECTOR DATABASE
# ============================================================================
class FAISSVectorDB:
    """FAISS vector database for similarity search"""
    
    def __init__(self, embeddings: np.ndarray, labels: List[str], config: Config):
        self.config = config
        self.labels = labels
        
        # Ensure 2D array
        assert embeddings.ndim == 2, f"Embeddings must be 2D, got {embeddings.ndim}D"
        assert embeddings.shape[0] == len(labels), "Mismatch: embeddings vs labels"
        
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        
        print(f"[FAISS] Index created: dim={self.dim}, vectors={len(labels)}")
        if self.config.DEBUG:
            print(f"[FAISS] Labels: {labels}")
    
    def search(self, query_embedding: np.ndarray, k: int = 4) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Search and return (scores, matched_labels, indices)"""
        assert query_embedding.shape[1] == self.dim, f"Dimension mismatch: {query_embedding.shape[1]} vs {self.dim}"
        
        distances, indices = self.index.search(query_embedding, k=min(k, len(self.labels)))
        
        scores = distances[0]
        matched_labels = [self.labels[i] for i in indices[0]]
        
        if self.config.DEBUG:
            print(f"[SEARCH] Scores: {scores}, Labels: {matched_labels}")
        
        return scores, matched_labels, indices[0]


# ============================================================================
# REGION EXTRACTION
# ============================================================================
class RegionExtractor:
    """Extract specialized regions for specific PPE detection"""
    
    @staticmethod
    def get_head_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      crop_ratio: float = 0.35) -> np.ndarray:
        """Extract head region (top portion of person bbox)"""
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
# PPE CLASSIFIER
# ============================================================================
class PPEClassifier:
    """Hybrid YOLO + CLIP classifier"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Load models
        print("[INIT] Loading YOLO person detector...")
        self.person_model = YOLO(config.YOLO_MODEL)
        
        print("[INIT] Loading YOLO PPE detector...")
        self.ppe_model = YOLO(config.PPE_MODEL)
        
        # Load CLIP
        self.embedding_engine = CLIPEmbeddingEngine(config)
        
        # Create FAISS indices for helmet and coverall
        print("[INIT] Building FAISS indices...")
        helmet_embeddings = self.embedding_engine.get_text_embeddings(PPELabels.HELMET_LABELS)
        self.helmet_db = FAISSVectorDB(helmet_embeddings, PPELabels.HELMET_LABELS, config)
        
        coverall_embeddings = self.embedding_engine.get_text_embeddings(PPELabels.COVERALL_LABELS)
        self.coverall_db = FAISSVectorDB(coverall_embeddings, PPELabels.COVERALL_LABELS, config)
        
        print("[INIT] PPE Classifier ready!")
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons using YOLO"""
        results = self.person_model(frame)[0]
        person_boxes = []
        
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > self.config.PERSON_CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
        
        if self.config.DEBUG:
            print(f"[YOLO] Detected {len(person_boxes)} persons")
        
        return person_boxes
    
    def detect_ppe_yolo(self, frame: np.ndarray) -> Dict[str, List[Tuple]]:
        """Detect PPE using YOLO model"""
        results = self.ppe_model(frame)[0]
        ppe_boxes = {"helmet": [], "coverall": []}
        
        for box in results.boxes:
            if float(box.conf[0]) < self.config.YOLO_HELMET_CONF_THRESH:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = self.ppe_model.names[int(box.cls[0])].lower()
            
            if "helmet" in name:
                ppe_boxes["helmet"].append((x1, y1, x2, y2))
            elif "coverall" in name:
                ppe_boxes["coverall"].append((x1, y1, x2, y2))
        
        if self.config.DEBUG:
            print(f"[YOLO-PPE] Helmets: {len(ppe_boxes['helmet'])}, Coveralls: {len(ppe_boxes['coverall'])}")
        
        return ppe_boxes
    
    def check_overlap(self, person_bbox: Tuple[int, int, int, int], 
                     ppe_boxes: List[Tuple[int, int, int, int]]) -> bool:
        """Check if any PPE box overlaps with person bbox"""
        px1, py1, px2, py2 = person_bbox
        
        for ppe_box in ppe_boxes:
            ex1, ey1, ex2, ey2 = ppe_box
            
            # Center of PPE box
            center_x = (ex1 + ex2) // 2
            center_y = (ey1 + ey2) // 2
            
            # Check if center is inside person box
            if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                return True
        
        return False
    
    def classify_ppe(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int],
                     ppe_boxes: Dict[str, List[Tuple]]) -> Dict[str, bool]:
        """
        Classify PPE using hybrid approach:
        1. YOLO detections for strong signals
        2. CLIP similarity for confirmation
        """
        x1, y1, x2, y2 = person_bbox
        
        # -------- HELMET DETECTION --------
        has_helmet_yolo = self.check_overlap(person_bbox, ppe_boxes["helmet"])
        has_helmet_clip = False
        
        # Head crop for helmet detection
        head_crop = RegionExtractor.get_head_crop(frame, person_bbox, self.config.HEAD_CROP_RATIO)
        
        if head_crop.size > 0:
            head_emb = self.embedding_engine.get_image_embedding(head_crop)
            scores, labels, _ = self.helmet_db.search(head_emb, k=4)
            
            # Check for high similarity with "wearing helmet" labels
            for score, label in zip(scores, labels):
                if "wearing" in label and "helmet" in label and score > self.config.CLIP_SIMILARITY_THRESH:
                    has_helmet_clip = True
                    break
        
        # Final decision: YOLO is primary, CLIP confirms
        has_helmet = has_helmet_yolo or has_helmet_clip
        
        # -------- COVERALL DETECTION --------
        has_coverall_yolo = self.check_overlap(person_bbox, ppe_boxes["coverall"])
        has_coverall_clip = False
        
        # Full body crop for coverall detection
        body_crop = RegionExtractor.get_body_crop(frame, person_bbox)
        
        if body_crop.size > 0:
            body_emb = self.embedding_engine.get_image_embedding(body_crop)
            scores, labels, _ = self.coverall_db.search(body_emb, k=4)
            
            # Check for high similarity with "wearing coverall" labels
            for score, label in zip(scores, labels):
                if "wearing" in label and "coverall" in label and score > self.config.CLIP_SIMILARITY_THRESH:
                    has_coverall_clip = True
                    break
        
        # Final decision: YOLO is primary, CLIP confirms
        has_coverall = has_coverall_yolo or has_coverall_clip
        
        if self.config.DEBUG:
            print(f"[CLASSIFY] Helmet: YOLO={has_helmet_yolo}, CLIP={has_helmet_clip}, Final={has_helmet}")
            print(f"[CLASSIFY] Coverall: YOLO={has_coverall_yolo}, CLIP={has_coverall_clip}, Final={has_coverall}")
        
        return {
            "has_helmet": has_helmet,
            "has_coverall": has_coverall,
            "helmet_yolo": has_helmet_yolo,
            "helmet_clip": has_helmet_clip,
            "coverall_yolo": has_coverall_yolo,
            "coverall_clip": has_coverall_clip,
        }


# ============================================================================
# VISUALIZATION
# ============================================================================
class Visualizer:
    """Draw bounding boxes and labels"""
    
    @staticmethod
    def draw_result(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   ppe_result: Dict[str, bool]) -> np.ndarray:
        """Draw safety classification on frame"""
        x1, y1, x2, y2 = bbox
        
        has_helmet = ppe_result["has_helmet"]
        has_coverall = ppe_result["has_coverall"]
        
        # Color: green if safe, red if unsafe
        is_safe = has_helmet and has_coverall
        color = (0, 255, 0) if is_safe else (0, 0, 255)
        status = "SAFE" if is_safe else "UNSAFE"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
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
    print("PPE DETECTION SYSTEM - Production Ready")
    print("=" * 70)
    
    # Initialize classifier
    classifier = PPEClassifier(config)
    
    # Process images
    image_count = 0
    for filename in os.listdir(config.INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        image_count += 1
        image_path = os.path.join(config.INPUT_FOLDER, filename)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"[WARN] Failed to read: {filename}")
            continue
        
        output = frame.copy()
        
        print(f"\n[PROCESS] {image_count}. {filename}")
        
        # Detect persons
        person_boxes = classifier.detect_persons(frame)
        
        # Detect PPE (YOLO)
        ppe_boxes = classifier.detect_ppe_yolo(frame)
        
        # Classify each person
        for person_idx, person_bbox in enumerate(person_boxes):
            print(f"  └─ Person {person_idx + 1}/{len(person_boxes)}")
            
            ppe_result = classifier.classify_ppe(frame, person_bbox, ppe_boxes)
            output = Visualizer.draw_result(output, person_bbox, ppe_result)
        
        # Save result
        output_path = os.path.join(config.OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, output)
        print(f"  ✅ Saved: {filename}")
    
    print("\n" + "=" * 70)
    print(f"🎉 Processing complete! {image_count} images processed.")
    print(f"📁 Results saved to: {config.OUTPUT_FOLDER}")
    print("=" * 70)


if __name__ == "__main__":
    main()
