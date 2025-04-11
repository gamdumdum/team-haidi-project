
import cv2
import os
import openvino as ovs
from pathlib import Path
import datetime as dt
import sqlite3


# 라벨 및 색상 설정
LABEL_NAMES = {
    "bolt": {2: "Bolt_OK", 0: "Bolt_NG", 1: "Bolt_No"},
    "crack": {0: "Crack"}
}

COLORS = {
    "bolt": {2: (0, 255, 0), 0: (0, 0, 255), 1: (255, 0, 255)},     # 볼트: 빨강/녹색
    "crack": {0: (255, 255, 0)}                                    # 크랙: 파랑/분홍
}