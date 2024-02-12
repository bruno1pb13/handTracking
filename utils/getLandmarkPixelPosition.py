def get(position, resolution):
    h, w, c = resolution

    return [int(position.x * w), int(position.y * h)]