import cv2
import openvino as ov

# 라벨 및 색상 설정
LABEL_NAMES = {
    "bolt": {0: "Bolt_OK", 1: "Bolt_NG"},
    "crack": {0: "Crack"}
}

COLORS = {
    "bolt": {0: (0, 255, 0), 1: (0, 0, 255)},   # 볼트: 녹색/빨강
    "crack": {0: (255, 255, 0)}                 # 크랙: 파랑/분홍
}

class drawBoltBox:
            
    # 볼트 박스 그리기
    @staticmethod
    def draw_bolt(frame1, bolt_boxes1, rx, ry) :
        bolt_count = 0
        for box in bolt_boxes1:
            x1, y1, x2, y2, conf, label = box

            # Adjust coordinates to the original frame
            x1 += rx
            x2 += rx
            y1 += ry
            y2 += ry

            color = COLORS["bolt"].get(label, (0, 255, 255))
            cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame1, f"{LABEL_NAMES['bolt'][label]} {conf:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label == 0: bolt_count += 1
        return bolt_count

