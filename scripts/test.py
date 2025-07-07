import rtde_control

robot_ip = "192.168.5.101"
try:
    ctrl = rtde_control.RTDEControlInterface(robot_ip)
    print("RTDE 연결 성공")
    ctrl.endFreedriveMode()
except Exception as e:
    print(f"RTDE 연결 실패: {e}")