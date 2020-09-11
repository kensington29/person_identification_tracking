#Import the neccesary libraries
import numpy as np
import argparse
import cv2 

import time
import concurrent.futures
from multiprocessing import Pool

# 物体検出処理のクラス
class object_detectier_mobilenetssd:
    # コンストラクタ
    def __init__(self):
        # DNNモデルのパス
        PROTOTXT_PATH = './MobileNetSSD_deploy.prototxt'
        WEIGHTS_PATH = './MobileNetSSD_deploy.caffemodel'

        # オブジェクトを識別する信頼度の閾値
        self.CONFIDENCE = 0.5

        # Labels of オブジェクトの
        self.classNames = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

        # モデルの読み込み
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

    # 物体検出処理
    def object_detection_mobilenetssd(self, frame):
        # 読み込んだ画像をblob(Binary Large Object)に変換
        # 
        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
        #Size of resized frame
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        self.net.setInput(blob)
        #Prediction of network
        detections = self.net.forward()

        # Factor for scale to original size of frame
        heightFactor = frame.shape[0]/300.0  
        widthFactor = frame.shape[1]/300.0 

        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Scale object detection to frame
            detections[0, 0, i, 3] = int(widthFactor * xLeftBottom) 
            detections[0, 0, i, 4] = int(heightFactor * yLeftBottom)
            detections[0, 0, i, 5]  = int(widthFactor * xRightTop)
            detections[0, 0, i, 6]  = int(heightFactor * yRightTop)

        return detections


# 画像処理のクラス
class image_processor:
    # コンストラクタ
    def __init__(self):
        pass


    def image_process(self):
        # カメラ準備
        cap = cv2.VideoCapture(0) 
        # 初期画像読み込み
        ret, frame = cap.read() 
        # カメラ設定
        cap.set(3,640) # set Width
        cap.set(4,480) # set Height

        test_label = "image processing"
        cap_status = cap.isOpened()
        print(cap_status)


        while True:
            # 画像を取得し、後の処理のために変換
            ret, frame = cap.read()
            # 画像を左右、上下反転
            frame = cv2.flip(frame, -1)

            # 画像を読み込めなかった場合は終了
            if not ret:
                break

            # 物体検出処理をインスタンス化  これを忘れるとエラーになる
            od = object_detectier_mobilenetssd()
            detections = od.object_detection_mobilenetssd(frame)

            # 検出した物体の内、信頼度が閾値を超えるものを物体として確定する
            # For get the class and location of object detected, 
            # There is a fix index for class, location and confidence
            # value in @detections array .
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2] #Confidence of prediction 
                if confidence > od.CONFIDENCE: # Filter prediction 
                    class_id = int(detections[0, 0, i, 1]) # Class label

                    xLeftBottom = int(detections[0, 0, i, 3])
                    yLeftBottom = int(detections[0, 0, i, 4])
                    xRightTop   = int(detections[0, 0, i, 5])
                    yRightTop   = int(detections[0, 0, i, 6])
                    # 検出した物体の位置に矩形を描画
                    # Draw location of object  
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                (0, 255, 0))

                    # 検出した物体のラベルと信頼度を描画
                    # Draw label and confidence of prediction in frame
                    if class_id in od.classNames:
                        label = od.classNames[class_id] + ": " + str(confidence)
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        yLeftBottom = max(yLeftBottom, labelSize[1])
                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                            (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                            (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                        print(label) #print class and confidence


            cv2.imshow("frame", frame)

            print(test_label)
            # time.sleep(2)

            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cap.release()
        cv2.destroyAllWindows()

def test():
    i=0
    while True:
        print("test"+str(i))

        # time.sleep(1)
        print("awake")

        if i < 9:
            i += 1
        else:
            i = 0


def main():
    # executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    # 画像処理をインスタンス化  これをしないと動かない
    ip = image_processor()
    # ip.image_process()
    # executor.submit(test,[1])
    # executor.submit(ip.image_process,[1])
    # executor.submit(timer_process)

    # with Pool(4) as pool:
    #     _ = [1]
    #     pool.map(ip.image_process,_)
    #     pool.map(test,a)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ip.image_process) for _ in range(1)]
        futures = [executor.submit(test) for _ in range(1)]

if __name__ == '__main__':
    main()
