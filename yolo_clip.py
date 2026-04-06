import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PATHS (EDIT THESE)
# -----------------------------
person_model_path = "yolov8m.pt"
ppe_model_path = r"C:\Users\iqsha\Downloads\YOLO_CNN\oldrig.pt"

input_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\new_data"
output_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\output"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
person_model = YOLO(person_model_path)
ppe_model = YOLO(ppe_model_path)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# -----------------------------
# TEXT PROMPTS
# -----------------------------
texts = [
    "a worker wearing a helmet",
    "a worker without a helmet",
    "a worker wearing coverall",
    "a worker without safety gear"
]

# Encode text once
text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)

text_features = text_features / text_features.norm(dim=-1, keepdim=True)
text_features = text_features.cpu().numpy()

# -----------------------------
# CLIP IMAGE EMBEDDING
# -----------------------------
def get_image_embedding(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inputs = clip_processor(images=image_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for file in os.listdir(input_folder):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_folder, file)
    frame = cv2.imread(path)

    if frame is None:
        print(f"❌ Skipping {file}")
        continue

    output = frame.copy()

    # -----------------------------
    # YOLO PERSON DETECTION
    # -----------------------------
    person_results = person_model(frame)[0]

    for box in person_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id != 0 or conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        # -----------------------------
        # CLIP EMBEDDING
        # -----------------------------
        img_emb = get_image_embedding(person_crop)

        # -----------------------------
        # COSINE SIMILARITY
        # -----------------------------
        similarity = cosine_similarity(img_emb, text_features)
        label_id = similarity.argmax()
        score = similarity[0][label_id]

        label = texts[label_id]

        # -----------------------------
        # FINAL DECISION
        # -----------------------------
        if "without" in label:
            color = (0, 0, 255)
            status = "UNSAFE"
        else:
            color = (0, 255, 0)
            status = "SAFE"

        # -----------------------------
        # DRAW OUTPUT
        # -----------------------------
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cv2.putText(output,
                    f"{label} ({score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        cv2.putText(output,
                    status,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    save_path = os.path.join(output_folder, file)
    cv2.imwrite(save_path, output)

    print(f"✅ Processed {file}")

print("🎉 DONE!")