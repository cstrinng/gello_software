# from typing import Dict

# import numpy as np

# from gello.robots.robot import Robot

from typing import Dict
import numpy as np
from gym_custom.envs.real.ur.interface import URScriptInterface
from gello.robots.robot import Robot
import time

class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.5.101", no_gripper: bool = False):

        self.robot = URScriptInterface(host_ip=robot_ip)
        if not no_gripper:
            # URScriptInterface.comm 안에 이미 RobotiqGripper가 연결되어 있음
            self._use_gripper = True
        else:
            self._use_gripper = False
        # freedrive 모드는 기본 꺼진 상태로 시작

        self._free_drive = False
        self.robot.comm.end_freedrive_mode(wait=False)
        self._last_gripper_query_time = 0
        self._last_gripper_pos = 0.0
        

        # self._free_drive = False
        # self.robot.endFreedriveMode()
        # self._use_gripper = not no_gripper

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 7 if self._use_gripper else 6

    def _get_gripper_pos(self) -> float:
        now = time.time()
        if now - self._last_gripper_query_time > 0.5:
            try:
                raw = self.robot.get_gripper_position()
                self._last_gripper_query_time = now
                if raw is not None:
                    self._last_gripper_pos = float(raw[0]) / 255.0
            except Exception as e:
                print(f"[WARNING] Failed to get gripper position: {e}")
        return self._last_gripper_pos

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """

        joints = self.robot.get_joint_positions()
        if self._use_gripper:
            return np.append(joints, self._get_gripper_pos())
        return joints
        
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        # velocity = 0.5
        # acceleration = 0.5
        # dt = 1.0 / 500  # 2ms
        # lookahead_time = 0.2
        # gain = 100

        # robot_joints = joint_state[:6]
        # t_start = self.robot.initPeriod()
        # self.robot.servoJ(
        #     robot_joints, velocity, acceleration, dt, lookahead_time, gain
        # )
        # if self._use_gripper:
        #     gripper_pos = joint_state[-1] * 255
        #     self.gripper.move(gripper_pos, 255, 10)
        # self.robot.waitPeriod(t_start)

        q = joint_state[:6]
        hz = 15
        dt = 1.0 / hz
        self.robot.servoj(
            q = q.tolist(),
            t = dt, # t=0.04,
            lookahead_time=dt + 0.02,
            gain=300,
            wait=False,
        )
        if self._use_gripper:
            g_raw = float(joint_state[-1]) * 255.0
            # self.robot.get_gripper_position(g_raw, wait=False)
            self.robot.move_gripper_position(g_raw, wait=False)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.comm.freedrive_mode(wait=True)
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.comm.end_freedrive_mode(wait=True)

    def get_observations(self) -> Dict[str, np.ndarray]:
        print("get_observation")
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            # "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }


def main():
    robot_ip = "192.168.5.101"
    ur = URRobot(robot_ip, no_gripper=False)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
