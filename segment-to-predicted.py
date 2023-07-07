"""
指定した読み込みファイルと同じディレクトリーに保存
セグメント情報を追加
"""
import cv2
from os.path import dirname, basename, join, listdir
import numpy as np

# ラベル読み込み
def load_label(labels_list, i):
    labels_path = join(labels_path, labels_list[i])
    f = open(labels_path, 'r') # テキストファイル読み出し
    datalist = f.readlines() # 行単位でリストとして保存する
    data = np.empty(0) 
    for data in datalist:
        l = data.split() # 空白で分割してリスト化
        seg_data=l[1:-1] # 1番目のデータから-2番目のデータまでを取得
        seg_data=map(int, seg_data) # リストに含まれるデータをint型に変換
        data = np.concatenate(seg_data) # リストを結合(axis=0)
    f.close() # 開いていたテキストファイルを閉じる
    return data # 正規化されているセグメント情報を返す

# セグメント情報追加・保存
def add_seg(mov_path):
    dname = dirname(mov_path) # パスの親ディレクトリー取得
    fname = basename(mov_path) # パスのファイル名取得
    save_name = "add-seg-"+fname # 保存ファイル名
    save_path = join(dname, save_name) # 保存パス作成

    cap_file = cv2.VideoCapture(mov_path)
    print(cap_file.isOpened()) # True
    print("fps:", cap_file.get(cv2.CAP_PROP_FPS)) # fps取得
    print("# 総フレーム数:", cap_file.get(cv2.CAP_PROP_FRAME_COUNT)) # 総フレーム数

    #動画サイズ取得
    width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH)) # 幅取得
    height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 高さ取得

    fps = cap_file.get(cv2.CAP_PROP_FPS) #フレームレート取得
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #フォーマット指定
    writer = cv2.VideoWriter(save_path, fmt, fps, (width, height))

    labels_path = join(dname, "labels")
    labels_list = listdir(labels_path)

    i = 1 # フレーム数カウント
    # 動画終了まで処理
    while True :
        print("Frame: "+ str(i)) # フレーム数表示
        s = labels_list[i-1].stem
        mov_name = "2H6A9795_"
        idx = s.find(mov_name)
        r = s[idx+len(mov_name):]
        
        ret, img = cap_file.read() # 判別，numpyデータ
        if i != int(r):
            #動画書き込み
            writer.write(img)
            i += 1
            continue
        if ret == False: #動画が終われば処理終了
            break

        # ラベル読み込み
        seg_data = load_label(labels_list, i)
        if len(seg_data) >=2:
            for data in seg_data:
                for j, point in enumerate(data[::4]):
                    # 線の始点と終点の座標
                    points = [(point[j]*width, point[j+1]*height), (point[j+2]*width, point[j+3]*height)]
                    cv2.line(img, points[0], points[1], (255, 0, 0), 2) # (B, G, R)
        else:
            for k, data in enumerate(seg_data[::4]):
                # 線の始点と終点の座標
                points = [(data[k]*width, data[k+1]*height), (data[k+2]*width, data[k+3]*height)]
                cv2.line(img, points[0], points[1], (255, 0, 0), 2) # (B, G, R)


        #動画書き込み
        writer.write(img)

        i += 1

    cap_file.release()


def main():
    mov_path = 'runs/segment/predict3/2H6A9795.mp4' # 読み込みファイル
    label_path = "runs/segment/predict3/labels"
    # add_seg(mov_path)
    load_label('runs/segment/predict3', )


if __name__ == '__main__':
    main()
