import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
import utils

# Cài đặt cấu hình server, để phần mềm xe tự lái xe gọi vào đây lấy lệnh điều khiển xe
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# Tốc độ tối thiểu và tối đa của xe
MAX_SPEED = 35
MIN_SPEED = 10

# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED


# Đăng ký lệnh
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Tốc độ hiện tại của ô tô
        speed = float(data["speed"])
        # Ảnh từ camera giữa
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            # Tiền xử lý ảnh, cắt, reshape
            image = np.asarray(image)
            image = utils.preprocess(image)
            image = np.array([image])

            steering_angle = float(model.predict(image, batch_size=1))

            # Tốc độ ta để trong khoảng từ 10 đến 25
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # giảm tốc độ
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            # Gửi lại dữ liệu về góc lái, tốc độ cho phần mềm để ô tô tự lái
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("Đã kết nối ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Đường dẫn dến file model.h5'
    )

    args = parser.parse_args()

    # Load model mà ta đã train được từ bước trước
    model = load_model(args.model)

    # Bật socket
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)