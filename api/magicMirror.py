import threading
import requests
import json

def fire_and_forget(cursorX, cursorY):
    threading.Thread(target=sendPosition, args=(cursorX, cursorY)).start()


def sendPosition(cursorX, cursorY):
    print(cursorX, cursorY)

    url = "http://localhost:8080/cursor/updateCursor"

    payload = json.dumps({
        'x': cursorX,
        'y': cursorY
    })

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        requests.post(url, headers=headers, data=payload)  # Use `.json()` for direct payload
    except:
        pass

