from pathlib import Path

import cv2
import numpy as np
import openvino as ov

core = ov.Core()
# 테스트시 경로 수정 필요
model = core.read_model(
    "/home/test/Haidi/team-haidi-project/BoltCheck/model/Bolt/model.xml"
)

input_layer = model.input(0)
input_shape = input_layer.partial_shape
if input_shape.is_dynamic:
    input_shape[0] = 1  # Batch size
    input_shape[1] = 3  # Channels
    input_shape[2] = 736  # Height (모델 정보 참조)
    input_shape[3] = 992  # Width (모델 정보 참조)
    model.reshape({input_layer: input_shape})

# 모델 컴파일
compiled_model = core.compile_model(model=model, device_name="GPU")
# Get input dimensions
N, C, H, W = compiled_model.input(0).shape


def process_output(boxes_output, labels_output, frame_shape):
    boxes = []
    h, w = frame_shape[:2]

    for i in range(boxes_output.shape[1]):
        conf = boxes_output[0, i, 4]
        if conf > 0.07:
            # 1. 좌표 클리핑 (모델 입력 크기 내로 제한)
            x1 = max(0, min(boxes_output[0, i, 0], W - 1))
            y1 = max(0, min(boxes_output[0, i, 1], H - 1))
            x2 = max(0, min(boxes_output[0, i, 2], W - 1))
            y2 = max(0, min(boxes_output[0, i, 3], H - 1))

            # 2. 비율 계산 (원본 프레임에 맞게 스케일링)
            x_scale = w / W
            y_scale = h / H

            x1 = int(x1 * x_scale)
            x2 = int(x2 * x_scale)
            y1 = int(y1 * y_scale)
            y2 = int(y2 * y_scale)

            # 3. 최종 클리핑 (원본 프레임 경계 확인)
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

            label = int(labels_output[0, i])
            boxes.append((x1, y1, x2, y2, conf, label))

    return boxes


# 카메라 설정
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# 라벨 매핑 (model_info에서 확인)
LABEL_NAMES = {0: "True", 1: "False"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}  # True: 녹색  # False: 빨간색

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 전처리: 모델 입력에 맞게 크기 조정 및 차원 변경 (HWC → CHW)
        # resized = cv2.resize(frame, (W, H))
        resized = cv2.resize(frame, (992, 736))  # width, height 순서
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)

        # 모델 추론: 전처리된 이미지로 볼트 검출 수행
        # 결과 처리: 감지된 볼트 정보를 실제 좌표로 변환
        raw_output = compiled_model([input_tensor])
        boxes_output = raw_output["boxes"]  # or raw_output[compiled_model.output(0)]
        labels_output = raw_output["labels"]  # or raw_output[compiled_model.output(1)]

        boxes = process_output(boxes_output, labels_output, frame.shape)
        BoltsCount = 0

        # 녹색 박스 그리기
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box

            color = COLORS.get(class_id, (0, 255, 255))  # 기본값: cyan
            label_name = LABEL_NAMES.get(class_id, f"Class {class_id}")
            if label_name == "True":
                BoltsCount += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # 테스트용 박스 그리기 (이 코드가 작동하면 박스 그리기 로직은 정상)
        # cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)

        # 볼트 개수 표시
        cv2.putText(
            frame,
            f"Bolts: {BoltsCount}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if BoltsCount >= 2:
            cv2.circle(frame, (600, 330), 300, (255, 255, 0), 5, cv2.LINE_AA)

        cv2.imshow("Bolt Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
