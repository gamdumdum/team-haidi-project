# DrawCrackBox 그려주는 함수
# 작성일 : 2025. 04. 05     작성자 : 윤태검

# 수정일 : 2025. 04. 08 화요일 
# 수정 내용 : 크랙 결과 SQL 저장

from .moduleUtil import *

class drawCrackBox:

    # 크랙 박스 그리기
    @staticmethod
    def draw_crack(frame, crack_boxes, rx, ry) :

        # 시간 및 날짜
        dDay = dt.datetime.now().strftime("%Y-%m-%d")
        dTime = dt.datetime.now().strftime("%H:%M:%S")
        crack_count = 0
        take_CrackPic = 0

        for box in crack_boxes:
            x1, y1, x2, y2, conf, label = box

            # Adjust coordinates to the original frame
            x1 += rx
            x2 += rx
            y1 += ry
            y2 += ry

            color = COLORS["crack"].get(label, (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{LABEL_NAMES['crack'][label]} {conf:.2f}", 
                        (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label == 0: crack_count += 1

            # 박스의 가로, 세로 길이 계산
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # 대각선 길이 계산 (피타고라스 정리)
            diagonal_length = int((width**2 + height**2)**0.5)

            # 크랙 레벨 결정 (길이에 따라 분류)
            if diagonal_length < 30:
                crack_level = 1  # 작은 크랙
            elif 30 <= diagonal_length < 60:
                crack_level = 2  # 중간 크랙
            else:
                crack_level = 3  # 큰 크랙

            # 박스 크기 정보 표시 (디버깅용)
            cv2.putText(frame, f"W:{width} H:{height} L:{diagonal_length}", 
                        (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Level: {crack_level}", 
                        (x1, y1-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if conf > 0.7: # 신뢰도 0.8 이상
                # 'c' 키를 누르면 현재 프레임을 이미지로 저장
                #if cv2.waitKey(1)  == ord('c'):
                # 클래스별 폴더 경로 생성 
                save_dir = os.path.join(LABEL_NAMES['crack'][label], dDay)
                os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성 (exist_ok로 중복 방지)

                # 파일명 생성 (예: "crack_time.jpg")
                filename = os.path.join(
                    save_dir, 
                    f"{LABEL_NAMES['crack'][label]}_{dTime}.jpg"
                )
                # 이미지 저장
                cv2.imwrite(filename, frame) 
                print(f"Saved: {filename} (Class: {LABEL_NAMES['crack'][label]}, Confidence: {conf:.2f})")


                # SQLite3 데이터베이스에 저장
                try:
                    # 데이터베이스 연결
                    conn = sqlite3.connect('bolt.db3')
                    cursor = conn.cursor()
                    
                    # 테이블 생성 (없을 경우)
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS defectsDB (
                        ID INTEGER PRIMARY KEY AUTOINCREMENT,
                        NAME TEXT NOT NULL,
                        Defect TEXT NOT NULL,
                        DefectLevel TEXT NOT NULL,
                        DefectDate TEXT NOT NULL,
                        FilePath TEXT NOT NULL,
                        Confidence REAL NOT NULL,
                        DetectionTime TEXT NOT NULL
                    )
                    ''')
                    
                    # 데이터 삽입
                    cursor.execute('''
                    INSERT INTO defectsDB (NAME, Defect, DefectLevel, DefectDate, FilePath, Confidence, DetectionTime)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        LABEL_NAMES['crack'][label],    # NAME
                        LABEL_NAMES['crack'][label],    # DefectType (여기서는 NAME과 동일하게 설정)
                        crack_level,                    # 크랙 레벨
                        dDay,                           # DefectDate (YYYY-MM-DD)
                        filename,                       # FilePath
                        float(conf),                    # Confidence
                        dTime                           # DetectionTime (HH:MM:SS)
                    ))
                    
                    conn.commit()
                    print("Data 저장 완료.")
                    take_CrackPic = 1

                except sqlite3.Error as e:
                    print(f"Database error: {e}")
                finally:
                    if conn:
                        conn.close()

        return crack_count, take_CrackPic

                    