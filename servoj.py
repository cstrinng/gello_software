import time
import numpy as np
from gym_custom.envs.real.ur.interface import URScriptInterface

def main():
    ur = URScriptInterface(host_ip="192.168.5.101")

    # 현재 joint 상태 읽기
    q_now = ur.get_joint_positions()
    print("Current joint positions:", q_now)

    # 타겟 위치 설정 (모든 joint에 약간의 offset)
    q_target = q_now + np.array([-0.1, 0.1, -0.05, 0.0, 0.0, 0.0])
    print("Target joint positions:", q_target)

    # 반복적으로 servoj 명령 보내기
    hz = 100  # 100Hz 주기
    dt = 1.0 / hz
    duration = 2.0  # 2초간 반복
    num_steps = int(duration * hz)

    # print("[INFO] Start servoj loop")
    # for i in range(num_steps):
    #     ur.servoj(
    #         q=q_target,
    #         t=dt,
    #         lookahead_time=0.2,
    #         gain=300,
    #         wait=False
    #     )
    #     time.sleep(dt)
    # print("[INFO] Done servoj loop")


    ur.servoj(
        q=q_target,
        t=dt,
        lookahead_time=0.2,
        gain=300,
        wait=False
    )

    # 마지막 위치 확인
    q_final = ur.get_joint_positions()
    print("Final joint positions:", q_final)

if __name__ == "__main__":
    main()
