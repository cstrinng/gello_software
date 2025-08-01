import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import threading
from queue import Queue, Empty

import numpy as np
import tyro
import h5py

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera
from gello.cameras.camera import CameraDriver  # Protocol

import pyrealsense2 as rs
import cv2

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = True
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    cut_frames: bool = False # Only needed when you want to save a fixed number of frames 
    frames: int = 84


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
        # depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        return img

    def __str__(self):
        return f"RealSense({self.serial})"

def start_server(port, driver):
    server = ZMQServerCamera(driver, port=port)
    server.serve()

class AsyncCamera:
    def __init__(self, client: ZMQClientCamera, target_size=(224,224), crop_center=None):
        self.client = client
        self.target_size = target_size
        h, w = 320, 320
        self.frame = None
        self.lock = threading.Lock()

        self.crop_center = crop_center

        if crop_center is not None:
            cy, cx = crop_center
            self.x1 = max(cx - w // 2, 0)
            self.x2 = min(cx + w // 2, 640)
            self.y1 = max(cy - h // 2, 0)
            self.y2 = min(cy + h // 2, 480)
        else:
            cy, cx = None, None
            self.x1 = None
            self.x2 = None
            self.y1 = None
            self.y2 = None

        # 데몬 쓰레드로 백그라운드에서 프레임 갱신
        t = threading.Thread(target=self._update_loop, daemon=True)
        t.start()

    def _update_loop(self):
        while True:
            raw = self.client.read()
            if self.crop_center is not None:
                crop = raw[self.y1:self.y2, self.x1:self.x2]
                img = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                # img = crop
            else:
                # 리사이즈는 쓰레드 안에서
                img = cv2.resize(raw, self.target_size, interpolation=cv2.INTER_LINEAR)

            # img = cv2.resize(raw, self.target_size, interpolation=cv2.INTER_LINEAR)
            with self.lock:
                self.frame = img

    def read(self):
        # 제어 루프에선 가장 최신 프레임만 꺼내 감
        with self.lock:
            return self.frame

def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) < 2:
            print("두 대 이상의 RealSense가 연결되어 있지 않습니다.")
            return
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devs[:2]]
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
        

        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
            
            # "base": client1,
            # "wrist": client2,

            "base": AsyncCamera(client1, crop_center=(320, 240)),
            # "base": AsyncCamera(client1),
            "wrist": AsyncCamera(client2),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    # [0, -90, 90, -90, -90, 0, 0]
                    [0, -90, -90, -90, 90, 180]
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
            agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))


    save_path = None
    buffer: List[Tuple[Dict[str, Any], np.ndarray]] = []
    recording: bool = False
    start_time = time.time()
    cnt = 0
    action = obs['joint_positions']
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = agent.act(obs)
        # joints = obs["joint_positions"]
        # cnt += 1
        # if cnt % 10 == 0:
        #     action = joints + np.random.normal(loc=0.0, scale=0.02, size=7)
        # print(action)

        dt = datetime.datetime.now()
        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / args.agent
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Recording to {save_path}")
                buffer.clear()
                recording = True

            elif state == "save":
                assert save_path is not None, "something went wrong"
                buffer.append((obs.copy(), action.copy()))  

            elif state == "normal":
                if recording:
                # End of recording: flush buffer to HDF5
                    if save_path and buffer:
                        # # Create HDF5 file
                        h5_file = save_path / "data.hdf5"
                        with h5py.File(h5_file, 'w') as f:
                            grp = f.create_group('data')
                            # keys from first obs
                            keys = list(buffer[0][0].keys())
                            # prepare arrays
                            for key in keys:
                                # stack obs values
                                data_arr = np.stack([b[0][key] for b in buffer], axis=0)
                                grp.create_dataset(key, data=data_arr)
                            # actions
                            acts = np.stack([b[1] for b in buffer], axis=0)
                            grp.create_dataset('actions', data=acts)
                                    # 2) 동일 폴더에 RGB 비디오로도 저장
                        import cv2
                        fps = 30  # 원하는 재생 속도
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                        # base 와 wrist 각 카메라별로 비디오 생성
                        for cam in ('base','wrist'):
                            #  프레임 모으기
                            frames = [b[0][f'{cam}_rgb'] for b in buffer]
                            h, w = frames[0].shape[:2]
                            video_path = save_path / f"{cam}_rgb.mp4"
                            vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                            for frame in frames:
                                # frame 이 이미 (H,W,3) uint8 BGR이라 가정
                                vw.write(frame)
                            vw.release()
                    # reset
                    recording = False
                    buffer.clear()
                    save_path = None
            else:
                raise ValueError(f"Invalid state {state}")
        print("start")
        t0 = time.perf_counter()
        obs = env.step(action)
        
        # t_now = time.perf_counter()
        # dt = t_now - t_prev
        # print(dt)
        # t_prev = t_now
        dt_hz = 30
        dt_target = 1 / dt_hz
        # 실제 소요 시간
        dt_actual = time.perf_counter() - t0
        # print(f"[Timing] 실제 소요: {dt_actual*1000:.1f} ms")

        # 남은 시간 계산
        sleep_time = dt_target - dt_actual
        if sleep_time > 0:
            time.sleep(sleep_time)
            print(f"[Timing] sleep: {sleep_time*1000:.1f} ms → 총 주기: {dt_target*1000:.1f} ms")
        else:
            # 작업이 목표 주기보다 오래 걸린 경우
            print(f"[Warning] 처리 시간이 {dt_actual:.4f}s로 목표 주기 {dt_target:.4f}s를 초과했습니다.")

        print("end")
        dt = time.perf_counter() - t0
        print(f"[Timing] env. () 소요: {dt*1000:.1f} ms")


if __name__ == "__main__":
    main(tyro.cli(Args))
