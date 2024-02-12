import threading
import socketio
from aiohttp import web

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()

sio.attach(app)

@sio.event
def connect(sid, data):
    print(sid, 'Conectou')
    return "connect", 123


@sio.event
async def message(sid, data):
    await sio.emit('message', {'data': 'foobar'})
    pass

def sendAxys(x, y):
    sio.emit('axis_data', {'data': {'x': x, 'y': y}})  # Use unique event name
    return 'ok'


def startServer():

    threading.Thread(target=web.run_app, kwargs={"app" : app, "port": 3010}).start()


