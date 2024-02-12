import math


def detectclick(pointPosition, resolution):

    length = int(math.hypot((pointPosition[1].x * resolution[0]) - (pointPosition[0].x * resolution[0]), (pointPosition[1].y * resolution[1]) - pointPosition[0].y *  resolution[1]))
    # print(length)
    if length < 80 :
        return True

    return False
