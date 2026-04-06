import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PATHS
# -----------------------------
person_model_path = "yolov8m.pt"
ppe_model_path = r"C:\Users\iqsha\Downloads\YOLO_CNN\oldrig.pt"

input_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\new_data"
output_folder = r"C:\Users\iqsha\Downloads\YOLO_CNN\output2"

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
helmet_texts = [
    "a person wearing a safety helmet on head",
    "a person not wearing a helmet"
]

coverall_texts = [
    "a worker wearing a protective coverall suit",
    "a worker without safety clothing"
]

def encode_texts(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

helmet_features = encode_texts(helmet_texts)
coverall_features = encode_texts(coverall_texts)

# -----------------------------
# CLIP IMAGE EMBEDDING
# -----------------------------
def get_clip_embedding(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = clip_processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

# -----------------------------
# HELPER: CENTER CHECK
# -----------------------------
def center_in_box(inner, outer):
    x1, y1, x2, y2 = inner
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    ox1, oy1, ox2, oy2 = outer
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

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

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
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
    # PER PERSON ANALYSIS
    # -----------------------------
    for person in person_boxes:
        px1, py1, px2, py2 = person

        has_helmet = False
        has_coverall = False

        # -----------------------------
        # YOLO MATCHING FIRST
        # -----------------------------
        for h in helmet_boxes:
            if center_in_box(h, person):
                has_helmet = True

        for c in coverall_boxes:
            if center_in_box(c, person):
                has_coverall = True

        # -----------------------------
        # CLIP FALLBACK (ONLY IF NEEDED)
        # -----------------------------
        person_crop = frame[py1:py2, px1:px2]

        if person_crop.size == 0:
            continue

        # 🔥 Helmet fallback (HEAD ONLY)
        if not has_helmet:
            head_crop = frame[py1:int(py1 + 0.4*(py2-py1)), px1:px2]

            if head_crop.size > 0:
                emb = get_clip_embedding(head_crop)
                sim = cosine_similarity(emb, helmet_features)
                if sim.argmax() == 0:  # wearing helmet
                    has_helmet = True

        # 🔥 Coverall fallback (FULL BODY)
        if not has_coverall:
            emb = get_clip_embedding(person_crop)
            sim = cosine_similarity(emb, coverall_features)
            if sim.argmax() == 0:
                has_coverall = True

        # -----------------------------
        # FINAL DECISION
        # -----------------------------
        if has_helmet and has_coverall:
            color = (0, 255, 0)
            status = "SAFE"
        else:
            color = (0, 0, 255)
            status = "UNSAFE"

        # DRAW
        cv2.rectangle(output, (px1, py1), (px2, py2), color, 2)
        cv2.putText(output,
                    f"H:{has_helmet} C:{has_coverall} {status}",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # -----------------------------
    # SAVE
    # -----------------------------
    cv2.imwrite(os.path.join(output_folder, file), output)
    print(f"✅ {file}")

print("🎉 DONE")