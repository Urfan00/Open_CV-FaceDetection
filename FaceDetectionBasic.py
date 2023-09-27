import cv2
import mediapipe as mp
import time


cam = cv2.VideoCapture(0)

pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:
    success, img = cam.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bBoxc = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bBox = int(bBoxc.xmin * iw), int(bBoxc.ymin * ih), \
                    int(bBoxc.width * iw), int(bBoxc.height * ih)
            cv2.rectangle(img, bBox, (0, 0, 255), 2)
            cv2.putText(img, f"{int(detection.score[0]*100)}%", (bBox[0], bBox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)

