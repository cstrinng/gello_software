# safeunlock.py
import time
from gym_custom.envs.real.ur.interface import URScriptInterface
import logging
logging.basicConfig(level=logging.DEBUG)

# 1) 로봇에 접속
ur = URScriptInterface(host_ip="192.168.5.102")
comm = ur.comm  # 내부에 UrScriptExt 인스턴스가 들어 있습니다
dash = comm.robotConnector.DashboardClient

# 2) 전원 켜기 & 브레이크 해제
dash.ur_power_on()
dash.ur_brake_release()
print("PowerOn:", comm.robotConnector.RobotModel.RobotStatus().PowerOn)
print("Safety stopped:", comm.robotConnector.RobotModel.SafetyStatus().StoppedDueToSafety)
time.sleep(1.0)

# 3) 프로텍티브 스톱 해제 & 팝업 닫기
dash.ur_unlock_protective_stop()
dash.ur_close_safety_popup()
time.sleep(1.0)

# 4) (선택) 전체 오류 상태 리셋 확인
ok = comm.reset_error()
print("Reset OK:", ok)