import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from api.magicMirror import fire_and_forget
import utils.detectclick as detectclick
import cv2
import numpy as np
import pyautogui
import asyncio


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = './hand_landmarker.task'
landmarks = False
move = False
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global landmarks
    landmarks = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence= 0.8,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result)

detector = vision.HandLandmarker.create_from_options(options)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image

timestamp = 0
tap = False


minX = False
maxX = False
minY = False
maxY = False


points = []


def checkReferencial(newRef, scale):
    global minX, maxX, minY, maxY, points

    height, width, _ = scale
    points.append(newRef)

    minX = min(point.x for point in points)
    maxX = max(point.x for point in points)
    minY = min(point.y for point in points)
    maxY = max(point.y for point in points)

    reference_area = (minX, minY), (maxX, maxY)

    return reference_area

lastPosition = False
def handDetection(img, calibration):
    global timestamp, tap, lastPosition
    global minX, maxX, minY, maxY


    timestamp = timestamp + 1

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detector.detect_async(mp_image, timestamp)

    cursorX = 0
    cursorY = 0

    if landmarks:
        annoted = draw_landmarks_on_image(mp_image.numpy_view(), landmarks)

        if landmarks.hand_landmarks != []:

            height, width, _ = annoted.shape

            p4 = landmarks.hand_landmarks[0][4]
            p8 = landmarks.hand_landmarks[0][8]


            cv2.circle(annoted, [int(p4.x * width), int(p4.y * height)], 2, 255, 2)
            cv2.circle(annoted, [int(p8.x * width), int(p8.y * height)], 2, 255, 2)

            cv2.line(annoted, [int(p4.x * width), int(p4.y * height)],
                 [int(p8.x * width), int(p8.y * height)], (255, 0, 255), 3)

            p3x, p3y = ((p4.x * width) + (p8.x * width)) // 2, ((p4.y * height)+ (p8.y * height)) // 2

            reference = checkReferencial(p4, annoted.shape)

            #current Position
            if not lastPosition:
                lastPosition = p4

            #BorderZone
            space_percentage = 0.05

            space_x = (reference[1][0] - reference[0][0]) * space_percentage
            space_y = (reference[1][1] - reference[0][1]) * space_percentage
            # Ajuste as coordenadas do segundo retângulo com o espaço calculado
            second_minX = reference[0][0] + space_x
            second_minY = reference[0][1] + space_y
            second_maxX = reference[1][0] - space_x
            second_maxY = reference[1][1] - space_y

            cv2.rectangle(annoted, [int(reference[0][0] * width), int(reference[0][1] * height)], [int(reference[1][0] * width), int(reference[1][1]* height)], 255 , 1)

            cv2.rectangle(annoted, [int(second_minX * width), int(second_minY* height)],
                          [int(second_maxX * width), int(second_maxY* height)], 255, 1)


            areaWidth = int((reference[1][0] - reference[0][0]) * width)
            areaHeight = int((reference[1][1] - reference[0][1]) * height)

            # cv2.line(annoted, [int(reference[1][0] * width), int(p4.y * height)], [int(reference[0][0] * width), int(p4.y * height)], (255,100,100), 2)
            cv2.line(annoted, [int(p4.x * width), int(p4.y * height)], [int(reference[0][0] * width), int(p4.y * height)], 255, 2)

            cv2.line(annoted, [int(p4.x * width), int(reference[1][1] * height), ], [int(p4.x * width), int(reference[0][1] * height)], (255,100,100), 2)
            cv2.line(annoted, [int(p4.x * width), int(p4.y * height)], [int(p4.x * width), int(reference[0][1] * height), ], 255, 2)

            areaX = (reference[1][0] - reference[0][0]) * width
            areaY = (reference[1][1] - reference[0][1]) * height

            relPositionX = reference[1][0] - p4.x
            relPositionY = reference[1][1] - p4.y

            # Assuming you have the coordinates
            x1, y1 = int(p4.x * width), int(p4.y * height)
            x2, y2 = int(reference[0][0] * width), int(p4.y * height)

            # Calculate the distance
            line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Assuming you have the coordinates
            x1, y1 = int(p4.x * width), int(p4.y * height)
            x2, y2 = int(p4.x * width), int(reference[0][1] * height),

            # Calculate the distance
            line_length2 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if areaWidth:
                cursorX = ((line_length / areaWidth) * 1920)
                cursorY = ((line_length2 / areaHeight) * 1080)

                pyautogui.FAILSAFE = False


                # if -1 > (lastPosition.x * width) - (p4.x * width) < 1 or -1 > (lastPosition.y * height) - (p4.y * height) < 1:
                    # pyautogui.moveTo(cursorX, cursorY, _pause=False)

            lastPosition = p4

            clicked = detectclick.detectclick((p4, p8), (height, width))
            # if clicked:
            # pyautogui.click(cursorX, cursorY, _pause=False)
        else:
            minX, maxX, minY, maxY = False, False, False, False
            lastPosition = False
        cv2.imshow('annoted', annoted)

    else:
        print('noLandmark')

    return [cursorX, cursorY]