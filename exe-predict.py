from ultralytics import YOLO
import torch

torch.cuda.set_device(0) # Set to your desired GPU number

# Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model
model = YOLO(r"C:\Users\KikuchiLab\Documents/Project/yolov8/ultralytics/runs/segment/train-epoch100/weights/best.pt")  # load a pretrained model (recommended for training)

# Predict with the model
results = model(r"C:\Users\KikuchiLab\Documents\Project\yolov8\ultralytics\runs\segment\predict3-nakajima\crops\k\2H6A9795.jpg", save=True, 
                device=0, save_txt=False, save_conf=True, show_labels=True, show_conf=True, boxes=False, retina_masks=False, 
                visualize=True)  # predict on an image

"""
    model.predict argment
    source          'ultralytics/assets'	画像またはビデオのソース ディレクトリ
    conf    	    0.25                	検出のためのオブジェクト信頼度しきい値
    iou	            0.7	                    NMS の交差オーバーユニオン (IoU) しきい値
    half	        False	                半精度 (FP16) を使用する
    device	        None	                実行するデバイス、つまり cuda device=0/1/2/3 または device=cpu
    show	        False	                可能であれば結果を表示する
    save	        False	                結果を含む画像を保存する
    save_txt	    False	                結果を .txt ファイルとして保存
    save_conf	    False	                結果を信頼度スコアとともに保存する
    save_crop	    False	                切り取った画像を結果とともに保存する
    hide_labels	    False	                ラベルを非表示にする
    hide_conf	    False	                信頼度スコアを非表示にする
    max_det	        300	                    画像あたりの最大検出数
    vid_stride	    False	                ビデオのフレームレートのストライド
    line_width	    None	                境界ボックスの線の幅。None の場合、画像サイズに合わせて拡大縮小されます。
    visualize	    False	                モデルの特徴を視覚化する
    augment     	False	                画像拡張を予測ソースに適用する
    agnostic_nms	False	                クラスに依存しない NMS
    retina_masks	False	                高解像度のセグメンテーション マスクを使用する
    classes	        None                   	結果をクラスでフィルタリングします。つまり、class=0 または class=[0,2,3]
    boxes	        True                	セグメンテーション予測でボックスを表示する
"""


masks = results[0].masks  # Masks object
masks.xy  # x, y segments (pixels), List[segment] * N
masks.xyn  # x, y segments (normalized), List[segment] * N
masks.data  # raw masks tensor, (N, H, W) or masks.masks 
"""
'C:/Users/kikuchilab/Documents/Project/yolov8/ultralytics/runs/segment/predict3/crops/k/2H6A97955.jpg'
"""


"""

results = model("C:/Users/KikuchiLab/Desktop/データ移行/移行データ/移行データ/人力文字データ群/nakajima_data/videos/2H6A9795.MOV", save=True, device=0, save_txt=True, save_conf=True, save_crop=True, visualize=False, imgsz=1920)  # predict on an image
"""