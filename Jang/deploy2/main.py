import cv2
import numpy as np
import openvino as ov
from pathlib import Path
import os
from datetime import datetime
import time

core = ov.Core()
# 테스트시 경로 수정 필요
model = core.read_model("./model/model.xml", weights="./model/model.bin")

input_layer = model.input(0)
input_shape = input_layer.partial_shape
if input_shape.is_dynamic:
    input_shape[0] = 1  # Batch size
    input_shape[1] = 3  # Channels
    input_shape[2] = 416  # Height (모델 정보 참조)
    input_shape[3] = 416  # Width (모델 정보 참조)
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
        if conf > 0.6:
            # 1. 좌표 클리핑 (모델 입력 크기 내로 제한)
            x1 = max(0, min(boxes_output[0, i, 0], W-1))
            y1 = max(0, min(boxes_output[0, i, 1], H-1))
            x2 = max(0, min(boxes_output[0, i, 2], W-1))
            y2 = max(0, min(boxes_output[0, i, 3], H-1))
            
            # 2. 비율 계산 (원본 프레임에 맞게 스케일링)
            x_scale = w / W
            y_scale = h / H
            
            x1 = int(x1 * x_scale)
            x2 = int(x2 * x_scale)
            y1 = int(y1 * y_scale)
            y2 = int(y2 * y_scale)
            
            # 3. 최종 클리핑 (원본 프레임 경계 확인)
            x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
            y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
            
            label = int(labels_output[0, i])
            boxes.append((x1, y1, x2, y2, conf, label))
    
    return boxes


# 카메라 설정
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)  # 프레임 속도를 15 FPS로 제한

# 라벨 매핑 (model_info에서 확인)
LABEL_NAMES = {0: "Bolt"}
COLORS = {0: (0, 255, 0)}
# CRACK 검출 타이머 초기화
crack_detected_time = None
crack_missed_time = None
image_saved = False  # 이미지 저장 여부 플래그

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the ROI coordinates
        x1, y1, x2, y2 = 560, 390, 720, 540
        # Crop the frame to the ROI
        roi = frame[y1:y2, x1:x2]

        # Preprocess the ROI: Resize and change dimensions (HWC → CHW)
        resized = cv2.resize(roi, (416, 416))  # width, height order
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)

        # Perform inference on the cropped ROI
        raw_output = compiled_model([input_tensor])
        boxes_output = raw_output["boxes"]  # or raw_output[compiled_model.output(0)]
        labels_output = raw_output["labels"]  # or raw_output[compiled_model.output(1)]

        # Process the output and scale back to the ROI coordinates
        boxes = process_output(boxes_output, labels_output, roi.shape)

        # Draw the green rectangle for the ROI on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw detection results within the ROI
        crack_detected = False  # CRACK 검출 여부 플래그
        for box in boxes:
            bx1, by1, bx2, by2, conf, class_id = box

            # Adjust coordinates to the original frame
            bx1 += x1
            bx2 += x1
            by1 += y1
            by2 += y1

            color = COLORS.get(class_id, (0, 255, 255))  # Default: cyan
            label_name = LABEL_NAMES.get(class_id, f"Class {class_id}")
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
            cv2.putText(frame, f"{label_name}: {conf:.2f}", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # CRACK 클래스 검출 여부 확인
            if label_name == "Bolt":
                crack_detected = True

        # CRACK 검출 시 타이머 시작
        if crack_detected:
            if crack_detected_time is None:
                crack_detected_time = time.time()
            crack_missed_time = None
        else:
            if crack_detected_time is None:
                crack_missed_time = time.time()
            elif time.time() - crack_detected_time >= 0.5:
                crack_detected_time = None
                image_saved = False  # CRACK이 검출되지 않으면 플래그 초기화

        # CRACK 검출 후 1초 경과 시 이미지 저장
        if crack_detected_time and (time.time() - crack_detected_time >= 0.5):
            if not image_saved:  # 이미지가 저장되지 않은 경우에만 저장
                # Create 'crack' folder if it doesn't exist
                if not os.path.exists("Bolt"):
                    os.makedirs("Bolt")

                # Create a folder for today's date
                today_date = datetime.now().strftime("%Y-%m-%d")
                date_folder = os.path.join("Bolt", today_date)
                if not os.path.exists(date_folder):
                    os.makedirs(date_folder)

                # Generate the image filename with timestamp
                timestamp = datetime.now().strftime("%H-%M-%S")
                image_filename = f"{timestamp}.jpg"
                image_path = os.path.join(date_folder, image_filename)

                # Save the cropped ROI image
                cv2.imwrite(image_path, roi)
                print(f"Image saved: {image_path}")
                image_saved = True  # 이미지 저장 완료 플래그 설정

        # Display the frame with detections
        cv2.imshow("Bolt Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()