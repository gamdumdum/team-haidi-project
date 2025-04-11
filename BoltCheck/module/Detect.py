# 양품 처리 함수
# 작성일 : 2025. 04. 10 목요일    작성자 : 윤태검

from .moduleUtil import *

class DetectGood:
            
    # 볼트 박스 그리기
    @staticmethod
    def dtect_good(take_Status, cause = '없음') :

        # 시간 및 날짜
        dDay = dt.datetime.now().strftime("%Y-%m-%d")
        dTime = dt.datetime.now().strftime("%H:%M:%S")
        
        # 분류
        if take_Status == 0 :
            name = 'Good'
            sign = 2

        else :
            name = 'Bad'
            sign = 3

        # SQLite3 데이터베이스에 저장
        try:
            # 데이터베이스 연결
            conn = sqlite3.connect('bolt.db3')
            cursor = conn.cursor()
            
            # 테이블 생성 (없을 경우)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS goodsDB (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                NAME TEXT NOT NULL,
                Defect TEXT NOT NULL,
                DefectDate TEXT NOT NULL,
                DetectionTime TEXT NOT NULL
            )
            ''')
            
            # 데이터 삽입
            cursor.execute('''
            INSERT INTO goodsDB (NAME, Defect, DefectDate, DetectionTime)
            VALUES (?, ?, ?, ?)
            ''', (
                name,                           # 양/불 분류
                cause,                          # Defect Type (여기서는 NAME과 동일하게 설정)
                dDay,                           # DefectDate (YYYY-MM-DD)
                dTime                           # DetectionTime (HH:MM:SS)
            ))
            
            conn.commit()
            print("Data 저장 완료.")
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

        return sign

