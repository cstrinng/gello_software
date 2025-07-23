from dataclasses import dataclass
from multiprocessing import Process

import tyro
import time

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    # hostname: str = "128.32.175.167"


# def launch_server(port: int, camera_id: int, args: Args):
#     camera = RealSenseCamera(camera_id)
#     server = ZMQServerCamera(camera, port=port, host=args.hostname)
#     print(f"Starting camera server on port {port}")
#     server.serve()

def launch_server(port: int, camera_id: int, args: Args):
    try:
        print(f"[{port}] 초기화 중 (device_id={camera_id}) …")
        camera = RealSenseCamera(camera_id)
        server = ZMQServerCamera(camera, port=port, host=args.hostname)
        print(f"Starting camera server on port {port}")
        server.serve()
    except Exception as e:
        print(f"[Error][port={port}, device={camera_id}] 카메라 초기화 실패:", e)
        return


def main(args):
    ids = get_device_ids()
    camera_port = 5000
    camera_servers = []
    for camera_id in ids:
        # start a python process for each camera
        print(f"Launching camera {camera_id} on port {camera_port}")
        time.sleep(1)
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, args))
        )
        camera_port += 1

    for server in camera_servers:
        server.start()
        time.sleep(0.5)


if __name__ == "__main__":
    main(tyro.cli(Args))
