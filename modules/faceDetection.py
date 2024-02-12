import mediapipe as mp
import numpy as np
import cv2
import utils.getLandmarkPixelPosition as getLandmarkPixelPosition
from utils.midpoint import midpoint

mpFace = mp.solutions.face_mesh
face = mpFace.FaceMesh()

mpDraw = mp.solutions.drawing_utils
mouseClick = False

def gazeRatio(img, eye, area):

    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.polylines(mask, [eye], True, 255, 2)
    cv2.fillPoly(mask, [eye], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    eye = eye[area[0][1]: area[1][1], area[0][0]: area[1][0]]
    lwidth, lheight = eye.shape

    left_side_threshold = eye[0: lheight, 0: np.int32(lwidth)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = eye[0: lheight, np.int32(lwidth): width]
    right_side_white = cv2.countNonZero(right_side_threshold)


    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    # cv2.imshow("eye", eye)
    # cv2.imshow("left_side_threshold", left_side_threshold)
    # cv2.imshow("right_side_threshold", right_side_threshold)
    # print(left_side_white, right_side_white, gaze_ratio)

    return left_side_white, right_side_white
def faceDetection(img):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)

    h, w, c = img.shape

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            height, width, _ = img.shape


            right_eye = np.array([
                        [faceLms.landmark[33].x * w, faceLms.landmark[33].y * h],
                        [faceLms.landmark[7].x * w, faceLms.landmark[7].y * h],
                        [faceLms.landmark[163].x * w, faceLms.landmark[163].y * h],
                        [faceLms.landmark[144].x * w, faceLms.landmark[144].y * h],
                        [faceLms.landmark[145].x * w, faceLms.landmark[145].y * h],
                        [faceLms.landmark[153].x * w, faceLms.landmark[153].y * h],
                        [faceLms.landmark[154].x * w, faceLms.landmark[154].y * h],
                        [faceLms.landmark[155].x * w, faceLms.landmark[155].y * h],
                        [faceLms.landmark[133].x * w, faceLms.landmark[133].y * h],
                        [faceLms.landmark[173].x * w, faceLms.landmark[173].y * h],
                        [faceLms.landmark[157].x * w, faceLms.landmark[157].y * h],
                        [faceLms.landmark[158].x * w, faceLms.landmark[158].y * h],
                        [faceLms.landmark[159].x * w, faceLms.landmark[159].y * h],
                        [faceLms.landmark[160].x * w, faceLms.landmark[160].y * h],
                        [faceLms.landmark[161].x * w, faceLms.landmark[161].y * h],
                        [faceLms.landmark[246].x * w, faceLms.landmark[246].y * h]], np.int32)
            left_eye = np.array([
                        (faceLms.landmark[263].x * w, faceLms.landmark[263].y * h),
                        (faceLms.landmark[249].x * w, faceLms.landmark[249].y * h),
                        (faceLms.landmark[390].x * w, faceLms.landmark[390].y * h),
                        (faceLms.landmark[373].x * w, faceLms.landmark[373].y * h),
                        (faceLms.landmark[374].x * w, faceLms.landmark[374].y * h),
                        (faceLms.landmark[380].x * w, faceLms.landmark[380].y * h),
                        (faceLms.landmark[381].x * w, faceLms.landmark[381].y * h),
                        (faceLms.landmark[382].x * w, faceLms.landmark[382].y * h),
                        (faceLms.landmark[362].x * w, faceLms.landmark[362].y * h),
                        (faceLms.landmark[398].x * w, faceLms.landmark[398].y * h),
                        (faceLms.landmark[384].x * w, faceLms.landmark[384].y * h),
                        (faceLms.landmark[385].x * w, faceLms.landmark[385].y * h),
                        (faceLms.landmark[386].x * w, faceLms.landmark[386].y * h),
                        (faceLms.landmark[387].x * w, faceLms.landmark[387].y * h),
                        (faceLms.landmark[388].x * w, faceLms.landmark[388].y * h),
                        (faceLms.landmark[466].x * w, faceLms.landmark[466].y * h)],np.int32)

            minl_x = np.min(left_eye[:, 0])
            maxl_x = np.max(left_eye[:, 0])
            minl_y = np.min(left_eye[:, 1])
            maxl_y = np.max(left_eye[:, 1])

            minr_x = np.min(right_eye[:, 0])
            maxr_x = np.max(right_eye[:, 0])
            minr_y = np.min(right_eye[:, 1])
            maxr_y = np.max(right_eye[:, 1])

            left_eye_ratio1, left_eye_ratio2 = gazeRatio(img, left_eye, [[minl_x,minl_y], [maxl_x,maxl_y]])
            right_eye_ratio1, right_eye_ratio2 = gazeRatio(img, right_eye, [[minr_x,minr_y], [maxr_x,maxr_y]])

            print("a:", left_eye_ratio1, left_eye_ratio2, (left_eye_ratio1 / left_eye_ratio2))
            print("b:", right_eye_ratio1, right_eye_ratio2, (right_eye_ratio1 / right_eye_ratio2))

            # cv2.imshow('left', left_eye_ratio)
            # cv2.imshow('right', right_eye_ratio)
            # rightGazeRatio = gazeRatio(img, right_eye, [[minr_x,minr_y], [maxr_x,maxr_y]])
            #
            # TotalgazeRatio = np.int32((left_eye_ratio + right_eye_ratio) / 2)
            # print(left_eye_ratio, right_eye_ratio, TotalgazeRatio)


            cv2.polylines(img, [right_eye], True, 255, 2)
            cv2.polylines(img, [left_eye], True, 255, 2)



            # mpDraw.draw_landmarks(img, faceLms, mpFace.FACEMESH_LEFT_EYE, is_drawing_landmarks=False)