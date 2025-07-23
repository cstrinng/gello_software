import threading
import time
import pickle

import numpy as np
import cv2
import zmq

from gello.zmq_core.camera_node import ZMQServerCamera, ZMQClientCamera
from gello.cameras.camera import DummyCamera

# 서버 실행 함수
def start_server(port: int):
    camera = DummyCamera()
    server = ZMQServerCamera(camera, port=port)
    server.serve()


def main():
    # 5000, 5001 포트에 각각 서버 스레드 실행
    t1 = threading.Thread(target=start_server, args=(5000,), daemon=True)
    t2 = threading.Thread(target=start_server, args=(5001,), daemon=True)
    t1.start()
    t2.start()

    # 서버가 바인딩될 시간을 잠시 대기
    time.sleep(1)

    # 클라이언트 생성
    client1 = ZMQClientCamera(port=5000)
    client2 = ZMQClientCamera(port=5001)

    try:
        while True:
            # 이미지 요청 (480×640으로 리사이즈)
            color1, depth1 = client1.read((480, 640))
            color2, depth2 = client2.read((480, 640))

            # OpenCV 창에 출력
            cv2.imshow('Camera1 Color', color1)
            cv2.imshow('Camera1 Depth', depth1.astype(np.uint8))
            cv2.imshow('Camera2 Color', color2)
            cv2.imshow('Camera2 Depth', depth2.astype(np.uint8))

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
