import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # RealSense 컨텍스트 생성
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        print(f"두 대의 RealSense 카메라가 연결되지 않았습니다. 연결된 카메라 수: {len(devices)}")
        return

    # 각 카메라별 파이프라인 설정
    pipelines = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        # 컬러 스트림(640×480 @30fps) 활성화
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append((serial, pipeline))
        print(f"카메라 {serial} 스트리밍 시작")

    try:
        while True:
            for serial, pipeline in pipelines:
                # 프레임 대기
                frameset = pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                if not color_frame:
                    continue

                # NumPy 배열로 변환
                img = np.asanyarray(color_frame.get_data())

                # 창에 출력
                cv2.imshow(f"Camera {serial}", img)

            # 'q' 키 입력 시 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 파이프라인 종료 및 윈도우 해제
        for _, pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
