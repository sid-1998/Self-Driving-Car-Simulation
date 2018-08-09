

from flask import Flask
import socketio
import eventlet
import eventlet.wsgi
from io import BytesIO
import argparse
import numpy as np
from PIL import Image
import Dataset_generation
from keras.models import load_model
import base64



parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str,
                   help='Enter IP address for socket', default='0.0.0.0')
parser.add_argument('--min_speed', type=int,
                   help='Enter minimum speed of car', default=15)
parser.add_argument('--max_speed', type=int,
                   help='Enter maximum spped of car', default=25)
parser.add_argument('--path', type=str,
                   help='Enter path to saved model file',
                    default='./model.h5')

args = parser.parse_args()
path = args.path
ip = args.ip
Max_speed = args.max_speed
Min_speed = args.min_speed
speed_limit = Max_speed

model = load_model(path)

sio = socketio.Server()
app = Flask(__name__)

def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True
    )

@sio.on('connect')
def connect(sid, environ):
    print('connect', sid)
    send_control(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data['steering_angle'])
        throttle = float(data['throttle'])
        speed = float(data['speed'])

        image = Image.open(BytesIO(base64.b64decode(data['image'])))

        try:
            image = np.asarray(image)
            image = Dataset_generation.process(image)
            image = np.array((image))
            img = image.reshape((1,66,200,3))
            steering_angle = float(model.predict(img))

            global speed_limit
            if speed > speed_limit:
                speed_limit = Min_speed
            else:
                speed_limit = Max_speed

            throttle = 1.0 - ((steering_angle)**2) - ((speed/speed_limit)**2)

            print(steering_angle, throttle, speed)
            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)

app = socketio.Middleware(sio, app)
eventlet.wsgi.server(eventlet.listen((ip, 4567)), app)
