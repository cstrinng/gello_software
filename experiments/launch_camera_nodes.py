import multiprocessing as mp

# --------------------------------------------------
# 반드시 __main__ 진입부로 들어가기 전에 spawn 설정
# (Linux의 기본은 fork여서 librealsense 내부 스레드와 충돌 발생)
mp.set_start_method('spawn', force=True)
# --------------------------------------------------

from dataclasses import dataclass
import tyro


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    start_port: int = 5000


def launch_server(port: int, camera_id: int, args: Args):
    # RealSense 카메라와 ZMQ 서버는 자식 프로세스 안에서만 import
    from gello.cameras.realsense_camera import RealSenseCamera
    from gello.zmq_core.camera_node    import ZMQServerCamera

    camera = RealSenseCamera(camera_id)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"▶ 카메라 {camera_id} 서버 시작 → tcp://{args.hostname}:{port}")
    server.serve()


def main(args: Args):
    # get_device_ids 역시 여기서만 호출
    from gello.cameras.realsense_camera import get_device_ids

    device_ids = get_device_ids()
    processes = []
    print("device_ids:", device_ids)

    for idx, cam_id in enumerate(device_ids):
        port = args.start_port + idx
        p = mp.Process(target=launch_server, args=(port, cam_id, args), daemon=True)
        p.start()
        processes.append(p)
        print(f"  • 프로세스 {p.pid} → 카메라 {cam_id} on port {port}")

    # 부모 프로세스가 살아 있어야 자식도 살아 있으므로
    # 터미널에서 Ctrl-C 로 종료할 때까지 여기서 대기
    for p in processes:
        p.join()


if __name__ == "__main__":
    main(tyro.cli(Args))
