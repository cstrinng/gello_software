from gym_custom.envs.real.ur.interface import URScriptInterface
import numpy as np
import time

# UR3 IP 설정
ur = URScriptInterface(host_ip="192.168.5.101")

# 현재 상태 출력
print("==> Initial joint positions:")
print(ur.get_joint_positions())
print("==> Initial gripper position:")
print(ur.get_gripper_position())

# === 1. Joint 이동 (movej)
print("\n[Step 1] Moving joint to test position 1")
test_joint_pos1 = np.array([0.0, -1.5, 1.5, -1.5, 1.5, 0.0])
ur.movej(q=test_joint_pos1, wait=True)
print("Moved to test position 1")

# === 2. Gripper 닫기
print("\n[Step 2] Closing gripper")
ur.move_gripper_position(g=230, wait=True)
time.sleep(1.0)
print("Gripper position after close:", ur.get_gripper_position())

# === 3. Joint 이동 (movej)
print("\n[Step 3] Moving joint to test position 2")
test_joint_pos2 = np.array([0.0, -1.0, 1.0, -1.0, 1.0, 0.0])
ur.movej(q=test_joint_pos2, wait=True)
print("Moved to test position 2")

# === 4. Gripper 열기
print("\n[Step 4] Opening gripper")
ur.move_gripper_position(g=3, wait=True)
time.sleep(1.0)
print("Gripper position after open:", ur.get_gripper_position())

# === 5. 현재 상태 다시 출력
print("\n==> Final joint positions:")
print(ur.get_joint_positions())
print("==> Final gripper position:")
print(ur.get_gripper_position())

# 종료
ur.close()
