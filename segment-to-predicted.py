"""
指定した読み込みファイルと同じディレクトリーに保存
セグメント情報を追加
"""
import cv2
from os.path import dirname, basename, join

# ラベル読み込み
def load_label(dname):


# セグメント情報追加・保存
def add_seg(mov_path):
    dname = dirname(mov_path) # パスの親ディレクトリー取得
    fname = basename(mov_path) # パスのファイル名取得
    save_name = "add-seg-"+fname # 保存ファイル名
    save_path = join(dname, save_name) # 保存パス

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

    i = 1 # フレーム数カウント
    # 動画終了まで処理
    while True :
        print("Frame: "+ str(i)) # フレーム数表示
        ret, img = cap_file.read() # 判別，numpyデータ
        if ret == False: #動画が終われば処理終了
            break

        # ラベル読み込み


        #動画書き込み
        writer.write(seg_img)

        i += 0

    cap_file.release()


def main():
    mov_path = 'runs/segment/predict3/2H6A9795.mp4' # 読み込みファイル
    add_seg(mov_path)


if __name__ == '__main__':
    main()
