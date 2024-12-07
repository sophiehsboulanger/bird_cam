import os.path
from ultralytics import YOLO
import cv2
import math
from datetime import datetime
import argparse

def run_birdcam(DEBUG):
    if DEBUG:
        cap = cv2.VideoCapture('test_video.mp4')
    else:
        # start webcam
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    # model
    model = YOLO("yolo-Weights/yolov8n.pt", verbose=False)


    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier"
                  ]


    while True:
        if DEBUG:
            print("running")
        success, img = cap.read()
        #results = model(img, stream=True)
        results = model.predict(img, classes=[14], verbose=False)
        its_a_bird = False
        # coordinates
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                its_a_bird = True
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # class name
                #cls = int(box.cls[0])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                #cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        # check if there are results (birds)
        if its_a_bird:
            now = datetime.now()
            print("A bird was detected at {}. Confidence: {}".format(now.strftime("%H:%M:%S"), confidence ))
            # check if folder for today's date exists, if not create one
            date = now.strftime("%d%m%Y")
            folder_dir = 'saved_images/' + date
            if not os.path.isdir(folder_dir):
                os.makedirs(folder_dir)
                print("New folder created for today's date")
            file_name = folder_dir + "/" + now.strftime("%d%m%Y_%H%M%S.jpg")
            cv2.imwrite(file_name, img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for detecting birds.")
    parser.add_argument('--DEBUG', action='store_true', help='Enable debugging mode')
    args = parser.parse_args()
    run_birdcam(args.DEBUG)