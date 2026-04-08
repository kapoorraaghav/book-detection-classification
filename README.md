# Book Sale Classifier

A computer vision pipeline that takes a photo of a book sale table and automatically detects, crops, and identifies books from the scene.

---

## Overview

Point it at any book sale or library table photo and it will locate every book in the frame, crop each one out, and save them individually — ready to be passed into the classification stage.

**Current status**
- Book detection — done (YOLOv8)
- ML-based title extraction and category classification — in progress

---

## How it works

```
Book sale photo
      │
      ▼
  YOLOv8m (pretrained)
  detects all "book" objects
      │
      ▼
  Bounding boxes extracted
  + individual crops saved
      │
      ▼
  cropped_books/
  ├── book_0.jpg
  ├── book_1.jpg
  └── ...
      │
      ▼
  [ML classification — coming soon]
  Title · Author · Category
```

---

## Project structure

```
book-sale-classifier/
├── vision.py              # YOLO detection + crop pipeline
├── cropped_books/         # output directory (auto-created)
├── Book_Fair_Table.png    # example input image
└── README.md
```

---

## Getting started

### Prerequisites

```bash
pip install ultralytics opencv-python
```

YOLOv8 weights (`yolov8m.pt`) are downloaded automatically by Ultralytics on first run.

### Run detection

Place your image in the project root and update `img_path` in `vision.py` if needed, then:

```bash
python vision.py
```

Cropped book images will be saved to `cropped_books/`.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf` | `0.05` | YOLO confidence threshold — lower = more detections, more false positives |
| `img_path` | `Book_Fair_Table.png` | Input image path |
| `crop_dir` | `cropped_books/` | Output directory for crops |

---

## Roadmap

- [x] Book detection with YOLOv8
- [x] Automatic crop extraction per detection
- [ ] OCR-based title and author extraction (EasyOCR)
- [ ] LLM-based category classification (local, via Ollama)
- [ ] Hybrid pipeline — OCR fast path + vision LLM fallback
- [ ] Structured JSON output per book
- [ ] Web UI for upload and results

---

## Tech stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [OpenCV](https://opencv.org/) — image loading and crop extraction
- EasyOCR — *(coming soon)*
- Ollama + minicpm-v / mistral — *(coming soon)*

---

## Example output

Given a cluttered book fair table, the pipeline identifies individual books and saves clean crops:

```
Saved 14 book crops
```

---

## Contributing

Pull requests welcome. If you're working on the ML classification stage, see the roadmap above for what's next.

---

## License

MIT# book-detection-classification
