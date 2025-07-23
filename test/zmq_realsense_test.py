# zmq_realsense_test.py
import threading
import time
import pickle

import numpy as np
import cv2
import zmq
import pyrealsense2 as rs

from gello.zmq_core.camera_node import ZMQServerCamera, ZMQClientCamera
from gello.cameras.camera import CameraDriver  # Protocol

class RealSenseDriver(CameraDriver):
    def __init__(self, serial: str, width=640, height=480, fps=30):
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # depth stream도 원하면 cfg.enable_stream(rs.stream.depth, ...)
        self.pipeline.start(cfg)

    def read(self, img_size=None):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
        if img_size is not None:
            img = cv2.resize(img, (img_size[1], img_size[0]))
        # 깊이는 일단 무시하거나 동일하게 처리
        depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        return img, depth

    def __str__(self):
        return f"RealSense({self.serial})"

def start_server(port, driver):
    server = ZMQServerCamera(driver, port=port)
    server.serve()

def main():
    # 1) RealSense 기기 찾기
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) < 2:
        print("두 대 이상의 RealSense가 연결되어 있지 않습니다.")
        return
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devs[:2]]

    # 2) 포트별 서버 스레드
    drivers = [RealSenseDriver(s) for s in serials]
    ports = [5000, 5001]
    threads = []
    for port, drv in zip(ports, drivers):
        t = threading.Thread(target=start_server, args=(port, drv), daemon=True)
        t.start()
        threads.append(t)

    time.sleep(1)  # 바인딩 대기

    # 3) 클라이언트 연결
    client1 = ZMQClientCamera(port=5000)
    client2 = ZMQClientCamera(port=5001)

    try:
        while True:
            c1, d1 = client1.read((480, 640))
            c2, d2 = client2.read((480, 640))

            cv2.imshow("RS Camera1 Color", c1)
            cv2.imshow("RS Camera2 Color", c2)
            # 깊이영상은 임시로 컬러윈도우에서 보기 어려우니 생략하거나 normalize 후 보여줄 수 있습니다.

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
