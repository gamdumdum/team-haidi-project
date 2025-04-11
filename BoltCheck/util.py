
# 볼트 나사 및 크랙 검출 프로그램 (util.py)
# 작성일 : 2025. 04. 08 화요일  작성자 : 윤태검

import cv2
import numpy as np
import openvino as ov
from pathlib import Path
import os
import datetime as dt
from module import processOutput as pO
from module import drawBoltBox as dB 
from module import drawCrackBox as dC
from module import DefectCounter as DefectCount
from module import RealTimeDefectVisualizer as dV
from module import RealTimeTotalVisualizer as dT
from module import DeftectVisualizer as DefectVisual
from module import Detect as dG
import time
import sqlite3
import pymcprotocol


# Initialize OpenVINO
core = ov.Core()

# 볼트 모델 로드 [ 경로 확인 ]
bolt_model = core.read_model("BoltCheck/model/Bolt/model.xml", weights="BoltCheck/model/Bolt/model.bin")

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
crack_model = core.read_model("BoltCheck/model/crack/model01.xml", weights="BoltCheck/model/crack/model01.bin") 

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
# 그래프 및 검출처리
Defect_Visual = DefectVisual.DefectVisualizer()
Defect_Total = dT.RealTimeTotalVisualizer()
Defect_Chart = dV.RealTimeDefectVisualizer()
Defect_Good = dG.DetectGood()


# 카메라 설정 (2개 카메라)
cap1 = cv2.VideoCapture(4)  # 첫 번째 카메라
cap2 = cv2.VideoCapture(10)  # 두 번째 카메라 (디바이스 번호는 시스템에 맞게 조정)

# 카메라 속성 설정
#   Size: Discrete 1280x720			Interval: Discrete 0.033s (30.000 fps)
#   Size: Discrete 848x480			Interval: Discrete 0.011s (90.000 fps)
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 라벨 및 색상 설정
LABEL_NAMES = {
    "bolt": {2: "Bolt_OK", 0: "Bolt_NG", 1: "Bolt_No"},
    "crack": {0: "Crack"}
}

COLORS = {
    "bolt": {2: (0, 255, 0), 0: (0, 0, 255), 1: (255, 0, 255)},     # 볼트: 빨강/녹색
    "crack": {0: (255, 255, 0)}                                    # 크랙: 파랑/분홍
}
