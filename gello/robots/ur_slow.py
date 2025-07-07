from typing import Dict

import numpy as np

from gello.robots.robot import Robot
import urx

class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.5.101", no_gripper: bool = False):
        self.robot = urx.Robot(robot_ip)
        print(f"[URRobot] Connected to robot at {robot_ip}")

        [print("in ur robot") for _ in range(4)]

        if not no_gripper:
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")
            # gripper.activate()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        # self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        import time

        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = np.array(self.robot.getj(), dtype=np.float32)

        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]

        self.robot.movej(list(robot_joints), acc=acceleration, vel=velocity)  # ← 변경된 부분
        
        if self._use_gripper:
            # gripper 명령은 RTDE와 별도이므로 그대로 재사용
            gripper_pos = int(joint_state[-1] * 255)
            self.gripper.move(gripper_pos, 255, 10)

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
        self._free_drive = enable

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }


def main():
    # robot_ip = "192.168.1.11" # default
    robot_ip = '192.168.5.101' # LARR_left
    ur = URRobot(robot_ip, no_gripper=False)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
