from ultralytics import YOLO
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, "Book_Fair_Table.png")

model = YOLO("yolov8m.pt")

img = cv2.imread(img_path)
results = model(img, conf=0.05)

# Folder to save cropped books
crop_dir = os.path.join(BASE_DIR, "cropped_books")
os.makedirs(crop_dir, exist_ok=True)

book_id = 0

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])

        if model.names[cls] == "book":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🔥 Crop the detected book
            book_crop = img[y1:y2, x1:x2]

            if book_crop.size == 0:
                continue

            # Save cropped book
            crop_path = os.path.join(crop_dir, f"book_{book_id}.jpg")
            cv2.imwrite(crop_path, book_crop)

            book_id += 1

print(f"Saved {book_id} book crops")
