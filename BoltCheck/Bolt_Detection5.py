
# 볼트 나사 및 크랙 검출 프로그램 (듀얼 카메라 버전)
# 작성일 : 2025. 04. 07 월요일  작성자 : 윤태검

# 수정일 : 2025. 04. 10 목요일
# 수정 내용 : 캠화면 및 그래프 화면 위치 설정 및 크기 조절

from util import *

try:
    frame_count = 0
    # 창 생성 시 툴바 제거를 위한 설정
    # 창 생성 시 GUI 확장 기능 비활성화
    cv2.namedWindow("Camera 1 Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Camera 2 Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)

    # 창 위치와 크기 설정
    cv2.moveWindow("Camera 1 Detection", 10, 10)  # 첫 번째 창 위치 (x=100, y=100)
    cv2.resizeWindow("Camera 1 Detection", 1280, 720)  # 첫 번째 창 크기 (width=800, height=600)
    
    cv2.moveWindow("Camera 2 Detection", 1360, 10)   # 두 번째 창 위치 (x=950, y=100)
    cv2.resizeWindow("Camera 2 Detection", 1280, 720)  # 두 번째 창 크기 (width=800, height=600)

    # PLC 연결 설정 (IP 주소와 포트)
    # mc = pymcprotocol.Type3E()
    # mc.connect("192.168.3.40", 5010)  # PLC의 IP와 포트번호

    # 불량 검출 중복 방지
    take_BoltPic = 0
    take_CrackPic = 0
    take_Status = 0
    sign = 0    # 대기상태

    # mc.batchwrite_bitunits(headdevice="M100", values=[0])

    while True:
        # 두 카메라에서 프레임 읽기
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # M100~M101 비트 값 읽기
        # bit_data = mc.batchread_bitunits(headdevice="X12", readsize=1)
        # bit_data2 = mc.batchread_bitunits(headdevice="M100", readsize=1)
        
        #print("M100 상태:", bit_data2)
        # if bit_data[0] == 1 and bit_data2[0] == 0:
        #     take_BoltPic = 0
        #     take_CrackPic = 0
        #     sign = 1 # 신호 들어옴

        if not ret1 and not ret2:
            break

        # Define the ROI coordinates
        rx1, ry1, rx2, ry2 = 400, 200, 800, 600
        #rx1, ry1, rx2, ry2 = 300, 100, 600, 400

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
            if take_BoltPic == 0 and sign == 1:
                bolt_count1, take_BoltPic = D_bolt.draw_bolt(frame1, bolt_boxes1, rx1, ry1)

            # 크랙 박스 그리기
            if take_CrackPic == 0 and sign == 1:
                crack_count1, take_CrackPic = D_crack.draw_crack(frame1, crack_boxes1, rx1, ry1)

            # 카운트 표시
            cv2.putText(frame1, f"Cam1 Bolts: {bolt_count1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame1, f"Cam1 Crack: {crack_count1}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
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
            if take_BoltPic == 0 and sign == 1:
                bolt_count2, take_BoltPic = D_bolt.draw_bolt(frame2, bolt_boxes2, rx1, ry1)

            # 크랙 박스 그리기
            if take_CrackPic == 0 and sign == 1:
                crack_count2, take_CrackPic = D_crack.draw_crack(frame2, crack_boxes2, rx1, ry1)

            # 카운트 표시
            cv2.putText(frame2, f"Cam2 Bolts: {bolt_count2}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame2, f"Cam2 Crack: {crack_count2}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 화면에 표시
            cv2.imshow("Camera 2 Detection", frame2)

        # 양품 처리
        if take_BoltPic == 0 and take_CrackPic == 0:
            if sign == 1:
                if bolt_count1 == 1 and bolt_count2 == 1:
                    if crack_count1 < 1  and crack_count2 < 1:
                        print("양품 입니다.")
                        take_Status = 0
                        sign = Defect_Good.dtect_good(take_Status)
        else :
            if sign == 1 :
                if bolt_count1 == 0 or bolt_count2 == 0 :
                    if crack_count1 > 0 or crack_count2 > 0:
                        print("둘다 불량 입니다.")
                        take_Status = 1
                        cause = 'All'
                        sign = Defect_Good.dtect_good(take_Status, cause)
                    else :
                        print("나사 체결 불량 입니다.")
                        take_Status = 1
                        cause = 'bolt'
                        sign = Defect_Good.dtect_good(take_Status, cause)
                else :
                    if crack_count1 > 0 or crack_count2 > 0:
                        print("크랙 불량 입니다.")
                        take_Status = 1
                        cause = 'crack'
                        sign = Defect_Good.dtect_good(take_Status, cause)

        # if sign == 2 and bit_data2[0] == 0:
        #     mc.batchwrite_bitunits(headdevice="M101", values=[1])
        #     print("PLC로 양품 신호를 보냅니다.")
        #     mc.batchwrite_bitunits(headdevice="M100", values=[1])
        #     sign = 0 
        #     mc.batchwrite_bitunits(headdevice="M101", values=[0])
            
        # if sign == 3 and bit_data2[0] == 0:
        #     mc.batchwrite_bitunits(headdevice="M102", values=[1])
        #     print("PLC로 불량 신호를 보냅니다.")
        #     mc.batchwrite_bitunits(headdevice="M100", values=[1])
        #     sign = 0 
        #     mc.batchwrite_bitunits(headdevice="M102", values=[0])

        # 주기적으로 차트 업데이트 # 30프레임마다 업데이트
        if frame_count % 30 == 0:  
            Defect_Chart.update_chart()
            Defect_Total.update_chart()

        frame_count += 1
        
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap1.release()
    cap2.release()
    Defect_Chart.close()
    Defect_Total.close()
    cv2.destroyAllWindows()
    # 일자 데이터 시각화
<<<<<<< HEAD
    Defect_Visual.visualize_defect_counts_by_date("2025-04-01", "2025-04-11") 
=======
    Defect_Visual.visualize_defect_counts_by_date("2025-04-01", "2025-04-10") 
>>>>>>> c6dc1fbdebd71e09ed4712a6a4b33703f42443a8
    # 주별 데이터
    #DefectVisualizer.visualize_defect_counts_by_date(group_by='weekly')
    # 월별 데이터
    #DefectVisualizer.visualize_defect_counts_by_date(group_by='monthly')
    