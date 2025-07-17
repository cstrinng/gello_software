import time
from typing import Any, Dict, Optional

import numpy as np
import sys

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot
from ur3_forward_kinematics import forward_kinematics

class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints) # 여기로 들어가는거 맞는지 확인!!
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs       # robot_obs 딕셔너리에 joint_positions라는 키가 반드시 있어야 함
        # assert "joint_velocities" in robot_obs    # velocity no need for data
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        # observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = forward_kinematics(robot_obs["joint_positions"])  # Optional!! calculate ur3 forward kinematics
        observations["gripper_position"] = robot_obs["gripper_position"]

        print(observations, file=sys.stderr, flush=True)

        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()
