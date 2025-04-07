import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Tkinter 백엔드 사용
import sqlite3
import datetime as dt

class RealTimeDefectVisualizer:
    def __init__(self):
        plt.ion()  # 대화형 모드 활성화
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
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
                
                bars = self.ax.bar(defect_types, counts)
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