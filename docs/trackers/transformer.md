---
comments: true
---

# TransformerTracker

## Overview

`TransformerTracker` is a lightweight, transformer-inspired online tracker.
It keeps a memory token per active track and matches detections with a fused
score built from IoU and embedding similarity.

This design is inspired by query-based transformer MOT methods (e.g.,
TrackFormer, MOTR, and MeMOTR families), while remaining easy to plug into
existing detector pipelines in this repository.

## Why use it

- Better identity stability when embedding features are available.
- Detector-agnostic and online.
- Simpler to run than full end-to-end transformer MOT training stacks.

## Usage

```python
import cv2
import supervision as sv
from rfdetr import RFDETRMedium
from trackers import TransformerTracker

tracker = TransformerTracker()
model = RFDETRMedium()

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb)

    # Optional: attach detector embeddings if available:
    # detections.data["embedding"] = your_embeddings

    tracked = tracker.update(detections)
```
