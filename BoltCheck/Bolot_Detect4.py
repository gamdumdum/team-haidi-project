# 볼트 및 크랙 검출 (듀얼 카메라 버전)
# 2025.04.03 목요일

import cv2
import numpy as np
import openvino as ov
from pathlib import Path
import os
import datetime as dt
from module import processOutput as pO
from module import drawBoltBox as dB 
from module import drawCrackBox as dC

# Initialize OpenVINO
core = ov.Core()

# 볼트 모델 로드 [ 경로 확인 ]
bolt_model = core.read_model("./model/Bolt/model.xml", weights="./model/Bolt/model.bin")

input_layer = bolt_model.input(0)
input_shape = input_layer.partial_shape
if input_shape.is_dynamic:
    input_shape[0] = 1  # Batch size
    input_shape[1] = 3  # Channels
    input_shape[2] = 416  # Height (모델 정보 참조)
    input_shape[3] = 416  # Width (모델 정보 참조)
    bolt_model.reshape({input_layer: input_shape})

bolt_compiled = core.compile_model(model=bolt_model, device_name="GPU.1") # 외장 그래픽 카드
N_bolt, C_bolt, H_bolt, W_bolt = bolt_compiled.input(0).shape

# 크랙 모델 로드 [ 경로 확인 ]
crack_model = core.read_model("./model/Crack/model_test.xml", weights="./model/Crack/model_test.bin") 

input_layer2 = crack_model.input(0)
input_shape2 = input_layer2.partial_shape
if input_shape2.is_dynamic:
    input_shape2[0] = 1  # Batch size
    input_shape2[1] = 3  # Channels
    input_shape2[2] = 416  # Height (모델 정보 참조)
    input_shape2[3] = 416  # Width (모델 정보 참조)
    crack_model.reshape({input_layer2: input_shape2})

crack_compiled = core.compile_model(model=crack_model, device_name="GPU.1") # 외장 그래픽 카드
N_crack, C_crack, H_crack, W_crack = crack_compiled.input(0).shape

# 프로세스 출력
P_output = pO.processOutput()
# 볼트박스 그리기
D_bolt = dB.drawBoltBox()
# 크랙박스 그리기
D_crack = dC.drawCrackBox()

# 결함 
Defect_Visual = DefectVisual.DefectVisualizer()
Defect_Chart = dV.RealTimeDefectVisualizer()

# 카메라 설정 (2개 카메라)
cap1 = cv2.VideoCapture(4)  # 첫 번째 카메라
cap2 = cv2.VideoCapture(10)  # 두 번째 카메라 (디바이스 번호는 시스템에 맞게 조정)

# 카메라 속성 설정
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 라벨 및 색상 설정
LABEL_NAMES = {
    "bolt": {1: "Bolt_OK", 0: "Bolt_NG"},
    "bolt": {0: "Bolt_OK", 1: "Bolt_NG"},
    "crack": {0: "Crack"}
}

COLORS = {
    "bolt": {1: (0, 255, 0), 0: (0, 0, 255)},   # 볼트: 빨강/녹색
    "bolt": {0: (0, 255, 0), 1: (0, 0, 255)},   # 볼트: 녹색/빨강
    "crack": {0: (255, 255, 0)}                 # 크랙: 파랑/분홍
}

try:
    frame_count = 0
    while True:
        # 두 카메라에서 프레임 읽기
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 and not ret2:
            break

        # Define the ROI coordinates
        #rx1, ry1, rx2, ry2 = 560, 390, 720, 540
        rx1, ry1, rx2, ry2 = 400, 200, 800, 600

        # Crop the frame to the ROI
        roi1 = frame1[ry1:ry2, rx1:rx2] # 1번 카메라
        roi2 = frame2[ry1:ry2, rx1:rx2] # 2번 카메라

        # 공통 전처리 1번 카메라
        resized1 = cv2.resize(roi1, (416, 416)) 
        resized2 = cv2.resize(roi1, (416, 416)) 
        input_tensor1 = np.expand_dims(resized1.transpose(2, 0, 1), 0).astype(np.float32)
        input_tensor2 = np.expand_dims(resized2.transpose(2, 0, 1), 0).astype(np.float32)

        # 공통 전처리 2번 카메라
        resized3 = cv2.resize(roi2, (416, 416)) 
        resized4 = cv2.resize(roi2, (416, 416)) 
        input_tensor3 = np.expand_dims(resized3.transpose(2, 0, 1), 0).astype(np.float32)
        input_tensor4 = np.expand_dims(resized4.transpose(2, 0, 1), 0).astype(np.float32)

        # Draw the green rectangle for the ROI on the original frame
        cv2.rectangle(frame1, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.rectangle(frame2, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        # 카메라 1 처리
        if ret1:

            # 볼트 검출
            bolt_output1 = bolt_compiled([input_tensor1])
            bolt_boxes1 = P_output.process_output(bolt_output1["boxes"], bolt_output1["labels"], roi1.shape, W_bolt, H_bolt)

            # 크랙 검출
            crack_output1 = crack_compiled([input_tensor2])
            crack_boxes1 = P_output.process_output(crack_output1["boxes"], crack_output1["labels"], roi1.shape, W_crack, H_crack)

            # 결과 시각화
            bolt_count1 = crack_count1 = 0
            
            # 볼트 박스 그리기
            bolt_count1 = D_bolt.draw_bolt(frame1, bolt_boxes1, rx1, ry1)

            # 크랙 박스 그리기
            crack_count1 = D_crack.draw_crack(frame1, crack_boxes1, rx1, ry1)

            # 카운트 표시
            cv2.putText(frame1, f"Cam1 Bolts: {bolt_count1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame1, f"Cam1 Crack: {crack_count1}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # 화면에 표시
            cv2.imshow("Camera 1 Detection", frame1)

        # 카메라 2 처리
        if ret2:

            # 볼트 검출
            bolt_output2 = bolt_compiled([input_tensor3])
            bolt_boxes2 = P_output.process_output(bolt_output2["boxes"], bolt_output2["labels"], roi2.shape, W_bolt, H_bolt)

            # 크랙 검출
            crack_output2 = crack_compiled([input_tensor4])
            crack_boxes2 = P_output.process_output(crack_output2["boxes"], crack_output2["labels"], roi2.shape, W_crack, H_crack)

            # 결과 시각화
            bolt_count2 = crack_count2 = 0
            
            # 볼트 박스 그리기
            bolt_count2 = D_bolt.draw_bolt(frame2, bolt_boxes2, rx1, ry1)

            # 크랙 박스 그리기
            crack_count2 = D_crack.draw_crack(frame2, crack_boxes2, rx1, ry1)

            # 카운트 표시
            cv2.putText(frame2, f"Cam2 Bolts: {bolt_count2}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame2, f"Cam2 Crack: {crack_count2}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # 화면에 표시
            cv2.imshow("Camera 2 Detection", frame2)

        # 주기적으로 차트 업데이트
        if frame_count % 20 == 0:  # 20프레임마다 업데이트
            Defect_Chart.update_chart()
        
        frame_count += 1
        if cv2.waitKey(1) == ord('q'):
            break
        

finally:
    cap1.release()
    cap2.release()

    Defect_Chart.close()
    cv2.destroyAllWindows()
    # 일자 데이터 시각화
    Defect_Visual.visualize_defect_counts_by_date("2025-04-01", "2025-04-04") 
    # 주별 데이터
    #DefectVisualizer.visualize_defect_counts_by_date(group_by='weekly')
    # 월별 데이터
    #DefectVisualizer.visualize_defect_counts_by_date(group_by='monthly')

