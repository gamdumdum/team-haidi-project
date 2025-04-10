# 실시간 불량률 그래프 표시
# 작성일 : 2025. 04. 08 화요일  작성자 : 윤태검

# 수정일 : 2025. 04. 10 목요일
# 수정 내용 : 화면 및 그래프 화면 위치 설정 및 크기 조절

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Tkinter 백엔드 사용
import sqlite3
import datetime as dt

class RealTimeTotalVisualizer:
    def __init__(self):
        plt.ion()  # 대화형 모드 활성화
        self.fig, self.ax = plt.subplots(figsize=(10, 5))

        # 창 위치와 크기 설정
        mng = plt.get_current_fig_manager()
        
        # TkAgg 백엔드 사용 시 (가장 일반적인 방법)
        try:
            mng.window.geometry("1280x720+1360+800")  # width x height + xpos + ypos
        except:
            pass

        self.today = dt.datetime.now().date()
        # 창 제목 설정 (최신 Matplotlib 호환 방식)
        try:
            title = f"불량률 모니터 - {self.today.strftime('%Y-%m-%d')}"
            self.fig.canvas.manager.window.title(title)
        except AttributeError:
            try:
                self.fig.canvas.set_window_title('Real-time Total Monitor')
            except:
                self.fig.suptitle('Real-time Total Monitor')  # 최후의 수단
        
    def update_chart(self):
        try:
            conn = sqlite3.connect('bolt.db3')
            cursor = conn.cursor()

            query = '''
            SELECT NAME, COUNT(*) 
            FROM goodsDB 
            WHERE DefectDate = ? 
            GROUP BY NAME
            ORDER BY COUNT(*) DESC
            '''
            cursor.execute(query, (self.today.strftime("%Y-%m-%d"),))
            results = cursor.fetchall()
            
            self.ax.clear()
            
            if results:
                defect_types = [r[0] for r in results]
                counts = [r[1] for r in results]
                
                # 색상 리스트 정의 (결함 타입 개수에 맞게 조정 가능)
                colors = ['#1f77b4', '#ff7f0e', '#4CAF50']
                bars = self.ax.bar(defect_types, counts, color=colors)
                self.ax.set_title('Real-time Total Count')
                self.ax.set_xlabel('Type')
                self.ax.set_ylabel('Count')
                
                for bar in bars:
                    height = bar.get_height()
                    self.ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}',
                                ha='center', va='bottom')
            
            plt.draw()
            plt.pause(0.01)  # 짧은 일시 정지로 업데이트 허용
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def close(self):
        plt.ioff()
        plt.close()