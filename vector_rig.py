import os
import cv2
import torch
import faiss
import numpy as np
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# LOAD MODELS
# -----------------------------
person_model = YOLO("yolov8m.pt")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# -----------------------------
# PATHS
# -----------------------------
input_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\new_data"
output_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\output"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# PPE TEXT EMBEDDINGS (VECTOR DB)
# -----------------------------
labels = [
    "a person wearing a safety helmet",
    "a person without a helmet",
    "a worker wearing a protective coverall",
    "a worker without protective clothing"
]

def encode_text(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        text_embeds = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0, :]
        text_embeds = clip_model.text_projection(text_embeds)

    feats = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")

text_embeddings = encode_text(labels)

# -----------------------------
# FAISS VECTOR DB
# -----------------------------
dim = text_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(text_embeddings)

# -----------------------------
# IMAGE EMBEDDING
# -----------------------------
def get_embedding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = clip_processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        image_embeds = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state[:, 0, :]
        image_embeds = clip_model.visual_projection(image_embeds)

    feats = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32").reshape(1, -1)

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for file in os.listdir(input_folder):

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(input_folder, file)
    frame = cv2.imread(path)

    if frame is None:
        continue

    output = frame.copy()

    # YOLO → PERSON ONLY
    results = person_model(frame)[0]

    for box in results.boxes:
        if int(box.cls[0]) != 0 or float(box.conf[0]) < 0.5:
            continue

        px1, py1, px2, py2 = map(int, box.xyxy[0])

        person_crop = frame[py1:py2, px1:px2]

        if person_crop.size == 0:
            continue

        # -----------------------------
        # CLIP EMBEDDING
        # -----------------------------
        emb = get_embedding(person_crop)

        # -----------------------------
        # VECTOR SEARCH
        # -----------------------------
        D, I = index.search(emb, k=4)  # check all labels

        scores = D[0]
        predicted_labels = [labels[i] for i in I[0]]

        # -----------------------------
        # DECISION LOGIC
        # -----------------------------
        has_helmet = False
        has_coverall = False

        for label, score in zip(predicted_labels, scores):

            if "helmet" in label and "wearing" in label and score > 0.25:
                has_helmet = True

            if "coverall" in label and "wearing" in label and score > 0.25:
                has_coverall = True

        # -----------------------------
        # DRAW RESULT
        # -----------------------------
        if has_helmet and has_coverall:
            color = (0, 255, 0)
            status = "SAFE"
        else:
            color = (0, 0, 255)
            status = "UNSAFE"

        cv2.rectangle(output, (px1, py1), (px2, py2), color, 2)
        cv2.putText(output,
                    f"H:{has_helmet} C:{has_coverall} {status}",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.imwrite(os.path.join(output_folder, file), output)
    print(f"✅ {file}")

print("🎉 DONE")