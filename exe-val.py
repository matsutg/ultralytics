from ultralytics import YOLO
import torch

torch.cuda.set_device(0) # Set to your desired GPU number


# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("C:/Users/kikuchilab/Documents/Project/yolov8/ultralytics/runs/segment/epoch100/weights/best.pt")

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map50-95(M) of each category