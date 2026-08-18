"""
train.py

Trains a YOLOv8 object-detection model on the CoachCV Clean & Jerk dataset
(7 classes: Barbell, Setting, Cleaning, Hold, Pressing, Complete, Release).

Designed to run in Google Colab with a free GPU. See the accompanying
setup guide for exact steps (uploading your data to Drive, running this
in a Colab notebook, downloading the resulting weights).

Usage (in a Colab cell, after installing ultralytics and mounting Drive):
    !python train.py
"""

from ultralytics import YOLO

# 'yolov8n.pt' = the "nano" variant -- smallest and fastest of the YOLOv8
# family. Starting from these pretrained (COCO) weights instead of random
# initialization is "transfer learning" -- the model already knows general
# shapes/edges/objects, so it converges much faster on our 7 custom classes
# than starting from scratch would.
model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,       # stop early if val loss hasn't improved in 20 epochs
    name="coachcv_run1",
)

# After this finishes, your trained weights will be saved to:
#   runs/detect/coachcv_run1/weights/best.pt   <- the one you want
#   runs/detect/coachcv_run1/weights/last.pt
print("Training complete. Download best.pt from runs/detect/coachcv_run1/weights/")
