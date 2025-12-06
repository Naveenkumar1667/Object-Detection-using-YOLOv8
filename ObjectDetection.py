from ultralytics import YOLO
import cv2
import os

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Output folders
img_folder = "unique_detections/images"
txt_folder = "unique_detections/labels"
os.makedirs(img_folder, exist_ok=True)
os.makedirs(txt_folder, exist_ok=True)

# Track saved object classes
saved_classes = set()

# Start webcam
cap = cv2.VideoCapture(0)
frame_id = 0

print("📸 Detecting..the objects. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(source=frame, conf=0.3, verbose=False)
    result = results[0]
    boxes = result.boxes

    if boxes is not None and boxes.conf is not None:
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        names = model.names

        for i, conf in enumerate(confs):
            label = names[int(class_ids[i])]
            if conf > 0.8 and label not in saved_classes:
                # Save annotated image
                img_path = os.path.join(img_folder, f"{label}_{frame_id:04d}.jpg")
                annotated = result.plot()
                cv2.imwrite(img_path, annotated)

                # Save label log
                txt_path = os.path.join(txt_folder, f"{label}_{frame_id:04d}.txt")
                x1, y1, x2, y2 = map(int, xyxy[i])
                with open(txt_path, "w") as f:
                    f.write(f"{label} {conf:.2f} [{x1},{y1},{x2},{y2}]")

                saved_classes.add(label)
                print(f"✅ Saved once: {label} → {img_path}")

    # Show live frame
    cv2.imshow("YOLOv8n Live Detection", result.plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
print("🛑 Detection stopped.")
