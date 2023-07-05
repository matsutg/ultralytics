from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter

torch.cuda.set_device(0) # Set to your desired GPU number

hyp = dict()

hyp['lr0']= 0.01 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
hyp['lrf']= 0.01 # final learning rate (lr0 * lrf)
hyp['momentum']= 0.937 # SGD momentum/Adam beta1
hyp['weight_decay']= 0.0005 # optimizer weight decay 5e-4
hyp['warmup_epochs']= 3.0 # warmup epochs (fractions ok)
hyp['warmup_momentum']= 0.8 # warmup initial momentum
hyp['warmup_bias_lr']= 0.1 # warmup initial bias lr
hyp['box']= 7.5 # box loss gain
hyp['cls']= 0.5 # cls loss gain (scale with pixels)
hyp['dfl']= 1.5 # dfl loss gain
hyp['pose']= 12.0 # pose loss gain
hyp['kobj']= 1.0 # keypoint obj loss gain
hyp['label_smoothing']= 0.0 # label smoothing (fraction)
hyp['nbs']= 64 # nominal batch size
hyp['hsv_h']= 0.7 # image HSV-Hue augmentation (fraction)
hyp['hsv_s']= 0.7 # image HSV-Saturation augmentation (fraction)
hyp['hsv_v']= 0.7 # image HSV-Value augmentation (fraction)
hyp['degrees']= 0.5 # image rotation (+/- deg)
hyp['translate']= 0.1 # image translation (+/- fraction)
hyp['scale']= 0.0 # image scale (+/- gain)
hyp['shear']= 0.0 # image shear (+/- deg)
hyp['perspective']= 0.001 # image perspective (+/- fraction), range 0-0.001
hyp['flipud']= 0.0 # image flip up-down (probability)
hyp['fliplr']= 0.5 # image flip left-right (probability)
hyp['mosaic']= 1.0 # image mosaic (probability)
hyp['mixup']= 0.5 # image mixup (probability)
hyp['copy_paste']= 0.0 # segment copy-paste (probability)

# Load a model
model = YOLO('yolov8n-seg.yaml')  # build a new model from scratch
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Use the model(Windowsでのトレーニングでは、num_workersが0以外の値に設定された場合、プロセスを作成するspawnメソッドを使用するため、問題を引き起こす可能性がある)
model.train(data="C:/Users/KikuchiLab/Documents/project/yolov8/nist_seg.yaml", 
            hyp=hyp, epochs=1, project="epoch1", workers=0, imgsz=128, device=0)  # train the model