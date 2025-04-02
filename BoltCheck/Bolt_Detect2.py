import cv2
import numpy as np
import openvino as ov
from pathlib import Path

# Initialize OpenVINO
core = ov.Core()

# 볼트 모델 로드
bolt_model = core.read_model("model/Bolt/model.xml")

input_layer = bolt_model.input(0)
input_shape = input_layer.partial_shape
if input_shape.is_dynamic:
    input_shape[0] = 1  # Batch size
    input_shape[1] = 3  # Channels
    input_shape[2] = 736  # Height (모델 정보 참조)
    input_shape[3] = 992  # Width (모델 정보 참조)
    bolt_model.reshape({input_layer: input_shape})

bolt_compiled = core.compile_model(model=bolt_model, device_name="GPU")
N_bolt, C_bolt, H_bolt, W_bolt = bolt_compiled.input(0).shape

# 크랙 모델 로드 (새로운 모델 추가)
crack_model = core.read_model("model/Crack/model.xml")  # 경로 수정 필요

input_layer2 = crack_model.input(0)
input_shape2 = input_layer2.partial_shape
if input_shape2.is_dynamic:
    input_shape2[0] = 1  # Batch size
    input_shape2[1] = 3  # Channels
    input_shape2[2] = 736  # Height (모델 정보 참조)
    input_shape2[3] = 992  # Width (모델 정보 참조)
    crack_model.reshape({input_layer2: input_shape2})

crack_compiled = core.compile_model(model=crack_model, device_name="GPU")
N_crack, C_crack, H_crack, W_crack = crack_compiled.input(0).shape

def process_output(boxes_output, labels_output, frame_shape, model_input_size):
    boxes = []
    h, w = frame_shape[:2]
    W_model, H_model = model_input_size  # 모델 입력 크기 (width, height)

    for i in range(boxes_output.shape[1]):
        conf = boxes_output[0, i, 4]
        if conf > 0.4:
            # 좌표 클리핑 및 스케일링
            x1 = max(0, min(boxes_output[0, i, 0], W_model-1))
            y1 = max(0, min(boxes_output[0, i, 1], H_model-1))
            x2 = max(0, min(boxes_output[0, i, 2], W_model-1))
            y2 = max(0, min(boxes_output[0, i, 3], H_model-1))
            
            x1 = int(x1 * (w / W_model))
            x2 = int(x2 * (w / W_model))
            y1 = int(y1 * (h / H_model))
            y2 = int(y2 * (h / H_model))
            
            x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
            y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
            
            label = int(labels_output[0, i])
            boxes.append((x1, y1, x2, y2, conf, label))
    
    return boxes

# 카메라 설정
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 라벨 및 색상 설정
LABEL_NAMES = {
    "bolt": {0: "Bolt_OK", 1: "Bolt_NG"},
    "crack": {0: "CASE", 1: "Crack"}
}

COLORS = {
    "bolt": {0: (0, 255, 0), 1: (0, 0, 255)},  # 볼트: 녹색/빨강
    "crack": {0: (255, 255, 0), 1: (255, 0, 255)}  # 크랙: 노랑/분홍
}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 공통 전처리
        resized = cv2.resize(frame, (992, 736))
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)

        # 볼트 검출
        bolt_output = bolt_compiled([input_tensor])
        bolt_boxes = process_output(bolt_output["boxes"], bolt_output["labels"], frame.shape, (W_bolt, H_bolt))

        # 크랙 검출
        crack_output = crack_compiled([input_tensor])
        crack_boxes = process_output(crack_output["boxes"], crack_output["labels"], frame.shape, (W_crack, H_crack))

        # 결과 시각화
        bolt_count = crack_count = 0
        
        # 볼트 박스 그리기
        for box in bolt_boxes:
            x1, y1, x2, y2, conf, label = box
            color = COLORS["bolt"].get(label, (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{LABEL_NAMES['bolt'][label]} {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label == 0: bolt_count += 1

        # 크랙 박스 그리기
        for box in crack_boxes:
            x1, y1, x2, y2, conf, label = box
            color = COLORS["crack"].get(label, (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{LABEL_NAMES['crack'][label]} {conf:.2f}", 
                       (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label == 0: crack_count += 1

        # 카운트 표시
        cv2.putText(frame, f"Bolts: {bolt_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Crack: {crack_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Multi Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()