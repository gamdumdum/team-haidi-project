# 실시간 결함 결과 그래프 표시
# 작성일 : 2025. 04. 08 화요일  작성자 : 윤태검

# 수정일 : 2025. 04. 10 목요일
# 수정 내용 : 화면 및 그래프 화면 위치 설정 및 크기 조절

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Tkinter 백엔드 사용
import sqlite3
import datetime as dt

class RealTimeDefectVisualizer:
    def __init__(self):
        plt.ion()  # 대화형 모드 활성화
        self.fig, self.ax = plt.subplots(figsize=(10, 5))

        # 창 위치와 크기 설정
        mng = plt.get_current_fig_manager()
        
        # TkAgg 백엔드 사용 시 (가장 일반적인 방법)
        try:
            mng.window.geometry("1280x720+10+800")  # width x height + xpos + ypos
        except:
            pass

        self.today = dt.datetime.now().date()
        # 창 제목 설정 (최신 Matplotlib 호환 방식)
        try:
            title = f"결함 모니터 - {self.today.strftime('%Y-%m-%d')}"
            self.fig.canvas.manager.window.title(title)
        except AttributeError:
            try:
                self.fig.canvas.set_window_title('Real-time Defect Monitor')
            except:
                self.fig.suptitle('Real-time Defect Monitor')  # 최후의 수단
        
    def update_chart(self):
        try:
            conn = sqlite3.connect('bolt.db3')
            cursor = conn.cursor()

            query = '''
            SELECT Defect, COUNT(*) 
            FROM defectsDB 
            WHERE DefectDate = ? 
            GROUP BY Defect
            ORDER BY COUNT(*) DESC
            '''
            cursor.execute(query, (self.today.strftime("%Y-%m-%d"),))
            results = cursor.fetchall()
            
            self.ax.clear()
            
            if results:
                defect_types = [r[0] for r in results]
                counts = [r[1] for r in results]
                
                # 색상 리스트 정의 (결함 타입 개수에 맞게 조정 가능)
                colors = ['#2ca02c', '#d62728', '#9467bd', '#9C27B0']
                bars = self.ax.bar(defect_types, counts, color=colors)
                self.ax.set_title('Real-time Defect Count')
                self.ax.set_xlabel('Defect Type')
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