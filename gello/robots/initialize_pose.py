import numpy as np
from gym_custom.envs.real.ur.interface import URScriptInterface
from gello.robots.robot import Robot

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
        
        self._last_q = self.get_joint_state()[:6]


        self._init_qpos = np.deg2rad([-5, -122, 98, -69, -90, -85])

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Robot 추상 메서드 구현: joint_state 명령을 URScriptInterface에 보냅니다."""
        # 예: position servo(via servoj)
        # self.robot.servoj(
        #     q=joint_state[:6].tolist(),
        #     **self.servoj_args
        # )
        # if self._use_gripper:
        #     g = float(joint_state[6]) * self.gripper_scale
        #     self.robot.move_gripper_position(g, wait=False)

    def get_joint_state(self) -> np.ndarray:
        """현재 관절 상태를 반환합니다."""
        # URScriptInterface.get_joint_positions()가 list 반환 시 예시
        return np.array(self.robot.get_joint_positions())

    def get_observations(self) -> dict:
        """로봇 센서(ee pose 등) 관찰값을 한 번에 가져옵니다."""
        # 필요한 관찰값만 추려서 리턴. 예시:
        qpos = np.array(self.robot.get_joint_positions())
        tcp  = np.array(self.robot.comm.get_actual_tcp_pose())
        return {'qpos': qpos, 'tcp_pose': tcp}

    def num_dofs(self) -> int:
        """이 로봇의 자유도 수(DOF)를 반환합니다."""
        # gripper 포함 여부 등에 따라 달라질 수 있습니다.
        return 7 if self._use_gripper else 6

    def move_to_init(self, wait: bool = True):
        self.robot.movej(q=self._init_qpos.tolist(), wait=wait)

def main():
    ur = URRobot(robot_ip="192.168.5.101", no_gripper=False)
    ur.move_to_init(wait=True)

if __name__ == "__main__":
    main()
