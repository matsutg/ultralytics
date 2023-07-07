from ultralytics import YOLO
import torch

torch.cuda.set_device(0) # Set to your desired GPU number

# Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model
model = YOLO("C:/Users/kikuchilab/Documents/Project/yolov8/ultralytics/runs/segment/epoch100/weights/best.pt")  # load a pretrained model (recommended for training)

# Predict with the model
results = model("C:/Users/kikuchilab/Documents/Project/yolov8/datasets/nist-seg/images/test/", save=True, device=0, save_txt=True, save_conf=False,)  # predict on an image


"""
masks = results[0].masks  # Masks object
masks.xy  # x, y segments (pixels), List[segment] * N
masks.xyn  # x, y segments (normalized), List[segment] * N
masks.data  # raw masks tensor, (N, H, W) or masks.masks 
'C:/Users/kikuchilab/Documents/Project/yolov8/ultralytics/runs/segment/predict3/crops/k/2H6A97955.jpg'
"""


"""

results = model("C:/Users/KikuchiLab/Desktop/データ移行/移行データ/移行データ/人力文字データ群/nakajima_data/videos/2H6A9795.MOV", save=True, device=0, save_txt=True, save_conf=True, 
                save_crop=True, visualize=False, imgsz=1920)  # predict on an image
"""