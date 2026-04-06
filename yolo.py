from ultralytics import YOLO
import cv2
import os

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

PERSON_CLASS_ID = 0

# -----------------------------
# IOU FUNCTION
# -----------------------------
def compute_iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    boxA_area = (ax2 - ax1) * (ay2 - ay1)
    boxB_area = (bx2 - bx1) * (by2 - by1)

    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# -----------------------------
# REMOVE DUPLICATES (NMS-lite)
# -----------------------------
def filter_duplicates(boxes, iou_threshold=0.5):
    filtered = []
    for box in boxes:
        keep = True
        for f in filtered:
            if compute_iou(box, f) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(box)
    return filtered

# -----------------------------
# HELMET MATCHING (HEAD REGION)
# -----------------------------
def is_helmet_on_person(person, helmet):
    px1, py1, px2, py2 = person
    hx1, hy1, hx2, hy2 = helmet

    hx_center = (hx1 + hx2) // 2
    hy_center = (hy1 + hy2) // 2

    # Expanded head region (top 50%)
    head_y_limit = py1 + 0.5 * (py2 - py1)

    return (
        px1 <= hx_center <= px2 and
        py1 <= hy_center <= head_y_limit
    )

# -----------------------------
# COVERALL MATCHING (BODY)
# -----------------------------
def is_coverall_on_person(person, coverall):
    px1, py1, px2, py2 = person
    cx1, cy1, cx2, cy2 = coverall

    cx_center = (cx1 + cx2) // 2
    cy_center = (cy1 + cy2) // 2

    return (
        px1 <= cx_center <= px2 and
        py1 <= cy_center <= py2
    )

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for file in os.listdir(input_folder):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"❌ Skipping {file}")
        continue

    # -----------------------------
    # RUN MODELS
    # -----------------------------
    person_results = person_model(frame)[0]
    ppe_results = ppe_model(frame)[0]

    output = frame.copy()

    person_boxes = []
    helmet_boxes = []
    coverall_boxes = []

    # -----------------------------
    # PERSON DETECTION
    # -----------------------------
    for box in person_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == PERSON_CLASS_ID and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

    # -----------------------------
    # PPE DETECTION
    # -----------------------------
    for box in ppe_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = ppe_model.names[cls_id]

        if class_name == "helmet":
            helmet_boxes.append((x1, y1, x2, y2))
        elif class_name == "coverall":
            coverall_boxes.append((x1, y1, x2, y2))

    # -----------------------------
    # REMOVE DUPLICATES
    # -----------------------------
    helmet_boxes = filter_duplicates(helmet_boxes)
    coverall_boxes = filter_duplicates(coverall_boxes)

    # -----------------------------
    # ASSOCIATE PPE WITH PERSON
    # -----------------------------
    for person in person_boxes:
        px1, py1, px2, py2 = person

        has_helmet = False
        has_coverall = False

        for helmet in helmet_boxes:
            if is_helmet_on_person(person, helmet):
                has_helmet = True

        for coverall in coverall_boxes:
            if is_coverall_on_person(person, coverall):
                has_coverall = True

        # -----------------------------
        # FINAL DECISION
        # -----------------------------
        if has_helmet and has_coverall:
            color = (0, 255, 0)  # GREEN
            status = "SAFE"
        else:
            color = (0, 0, 255)  # RED
            status = "UNSAFE"

        # Draw person box
        cv2.rectangle(output, (px1, py1), (px2, py2), color, 2)

        label = f"H:{has_helmet} C:{has_coverall} {status}"

        cv2.putText(output, label,
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # -----------------------------
    # OPTIONAL: LIGHT PPE DRAWING
    # -----------------------------
    for (x1, y1, x2, y2) in helmet_boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)

    for (x1, y1, x2, y2) in coverall_boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 165, 255), 1)

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    save_path = os.path.join(output_folder, file)
    cv2.imwrite(save_path, output)

    print(f"✅ Processed {file}")

print("🎉 ALL DONE!")