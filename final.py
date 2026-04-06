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
ppe_model = YOLO(r"C:\Users\iqsha\Downloads\YOLO_CNN\oldrig.pt")

# Load CLIP and get embedding projections
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Get the projection dimension
text_config = clip_model.config.text_config
projection_dim = clip_model.config.projection_dim

# -----------------------------
# PATHS
# -----------------------------
input_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\new_data"
output_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\output"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# LABELS
# -----------------------------
labels = [
    "a person wearing a helmet",
    "a person without a helmet",
    "a person wearing a coverall",
    "a person without a coverall"
]

# -----------------------------
# TEXT EMBEDDINGS (SAFE VERSION)
# -----------------------------
def encode_text(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        text_embeds = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0, :]
        text_embeds = clip_model.text_projection(text_embeds)   # ✅ project to shared space

    feats = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")

text_embeddings = encode_text(labels)
print("TEXT SHAPE:", text_embeddings.shape)  # MUST be (4, 512)

# -----------------------------
# FAISS
# -----------------------------
dim = text_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(text_embeddings)

# -----------------------------
# IMAGE EMBEDDING (SAFE VERSION)
# -----------------------------
def get_embedding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = clip_processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        image_embeds = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state[:, 0, :]
        image_embeds = clip_model.visual_projection(image_embeds)   # ✅ project to shared space

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

    # YOLO detections
    person_results = person_model(frame)[0]
    ppe_results = ppe_model(frame)[0]

    person_boxes = []
    helmet_boxes = []
    coverall_boxes = []

    # PERSON
    for box in person_results.boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
            person_boxes.append(tuple(map(int, box.xyxy[0])))

    # PPE
    for box in ppe_results.boxes:
        if float(box.conf[0]) < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = ppe_model.names[int(box.cls[0])]

        if name == "helmet":
            helmet_boxes.append((x1, y1, x2, y2))
        elif name == "coverall":
            coverall_boxes.append((x1, y1, x2, y2))

    # -----------------------------
    # PER PERSON
    # -----------------------------
    for (px1, py1, px2, py2) in person_boxes:

        has_helmet = False
        has_coverall = False

        # YOLO matching
        for h in helmet_boxes:
            cx = (h[0] + h[2]) // 2
            cy = (h[1] + h[3]) // 2
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                has_helmet = True

        for c in coverall_boxes:
            cx = (c[0] + c[2]) // 2
            cy = (c[1] + c[3]) // 2
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                has_coverall = True

        # -----------------------------
        # CLIP FALLBACK
        # -----------------------------
        crop = frame[py1:py2, px1:px2]

        if crop.size > 0:
            emb = get_embedding(crop)
            print("IMAGE:", emb.shape)  # MUST be (1, 512)

            D, I = index.search(emb, k=1)
            label = labels[I[0][0]]

            # Assist only (do not override YOLO)
            if not has_helmet and "helmet" in label:
                has_helmet = True

            if not has_coverall and "coverall" in label:
                has_coverall = True

        # -----------------------------
        # FINAL
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