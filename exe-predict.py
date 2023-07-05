from ultralytics import YOLO
import torch

torch.cuda.set_device(0) # Set to your desired GPU number

model = YOLO("C:/Users/KikuchiLab/Documents/project/yolov8/ultralytics/runs/segment/epoch100/weights/best.pt")  # load a pretrained model (recommended for training)

results = model("C:/Users/KikuchiLab/Desktop/データ移行/移行データ/移行データ/人力文字データ群/nakajima_data/videos/2H6A9795.MOV", save=True, device=0, save_txt=True, save_conf=True, 
                save_crop=True, visualize=False, imgsz=1920)  # predict on an image