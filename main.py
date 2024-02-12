import threading


import cv2
import mediapipe as mp
import time
import modules.handDetection as hd
import modules.gestureRecognize as gr
from api.socketio import startServer, sendAxys

cap = cv2.VideoCapture(0)

cTime = 0
pTime = 0

mouseClick = False
calibration = False

openPalm = False
openPalmStart = False

socket = startServer()

def main():
    while True:
        sucess, img = cap.read()

        relativeMov = hd.handDetection(img, calibration)

        if relativeMov[0] or relativeMov[1]:
            threading.Thread(target=sendAxys, kwargs={"x": relativeMov[0], "y": relativeMov[1]}).start()

        # recognization_result = gr.gestureDetection(img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
