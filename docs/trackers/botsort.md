---
comments: true
---

# BoT-SORT

## Overview

BoT-SORT extends the ByteTrack matching pipeline with stronger association cues to
improve identity consistency in crowded or long-occlusion scenes.

In this repository, `BOTSORTTracker` is a lightweight BoT-SORT-style variant
that uses confidence-aware two-stage association. Full research BoT-SORT setups
often include additional appearance ReID and camera-motion compensation modules.

## Run on video, webcam, or RTSP stream

```python
import cv2
import supervision as sv
from rfdetr import RFDETRMedium
from trackers import BOTSORTTracker

tracker = BOTSORTTracker()
model = RFDETRMedium()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb)
    tracked = tracker.update(detections)

    frame_bgr = box_annotator.annotate(frame_bgr, tracked)
    frame_bgr = label_annotator.annotate(
        frame_bgr,
        tracked,
        labels=tracked.tracker_id,
    )
```


For broader benchmark context, see the [SOTA landscape](sota.md).
