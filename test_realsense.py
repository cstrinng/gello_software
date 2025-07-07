import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 장치 연결 확인
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("❌ RealSense 장치가 연결되지 않았습니다.")
    exit()

print("✅ RealSense 장치가 감지되었습니다. 스트리밍을 시작합니다...")

# 2. 스트림 설정 및 시작
pipeline = rs.pipeline()
config = rs.config()

# RGB 및 Depth 스트림 활성화
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 파이프라인 시작
pipeline.start(config)

try:
    while True:
        # 프레임 수신
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 프레임 유효성 확인
        if not color_frame or not depth_frame:
            continue

        # numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Depth 시각화를 위한 컬러맵 적용
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # 두 이미지 좌우로 나란히 표시
        images = np.hstack((color_image, depth_colormap))

        # 출력
        cv2.imshow('RealSense Stream (Color | Depth)', images)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 스트림 종료
    pipeline.stop()
    cv2.destroyAllWindows()
