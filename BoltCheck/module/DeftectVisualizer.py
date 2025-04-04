import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class DefectVisualizer:
    
    @staticmethod
    def visualize_defect_counts_by_date(start_date=None, end_date=None, group_by='daily'):
        """
        defects 테이블에서 결함 유형별 개수를 날짜별로 조회하여 시각화하는 함수
        
        Parameters:
        - start_date: 조회 시작일 (YYYY-MM-DD 형식)
        - end_date: 조회 종료일 (YYYY-MM-DD 형식)
        - group_by: 그룹화 기준 ('daily', 'weekly', 'monthly')
        """
        try:
            # 데이터베이스 연결
            conn = sqlite3.connect('bolt.db3')
            cursor = conn.cursor()
            
            # 기본 날짜 범위 설정 (전체 기간)
            if not start_date:
                cursor.execute("SELECT MIN(DefectDate) FROM defectsDB")
                start_date = cursor.fetchone()[0]
            if not end_date:
                cursor.execute("SELECT MAX(DefectDate) FROM defectsDB")
                end_date = cursor.fetchone()[0]
            
            # 날짜별 그룹화 쿼리 생성
            if group_by == 'daily':
                date_format = "DefectDate"
                date_display_format = "%Y-%m-%d"
            elif group_by == 'weekly':
                date_format = "strftime('%Y-%W', DefectDate)"
                date_display_format = "Week %W, %Y"
            elif group_by == 'monthly':
                date_format = "strftime('%Y-%m', DefectDate)"
                date_display_format = "%Y-%m"
            else:
                raise ValueError("group_by must be 'daily', 'weekly', or 'monthly'")
            
            query = f'''
            SELECT 
                {date_format} as DateGroup,
                Defect, 
                COUNT(*) as Count 
            FROM defectsDB
            WHERE DefectDate BETWEEN ? AND ?
            GROUP BY DateGroup, Defect
            ORDER BY DateGroup, Count DESC
            '''
            
            cursor.execute(query, (start_date, end_date))
            results = cursor.fetchall()
            
            if not results:
                print(f"No defect records found between {start_date} and {end_date}.")
                return
            
            # 데이터 정리
            date_groups = sorted(set(r[0] for r in results))
            defect_types = sorted(set(r[1] for r in results))
            
            # 날짜별 데이터 구조 생성
            data = {date: {defect: 0 for defect in defect_types} for date in date_groups}
            for date, defect, count in results:
                data[date][defect] = count
            
            # 색상 설정
            colors = {
                "Bolt_OK": (0, 255, 0),    # 녹색
                "Bolt_NG": (0, 0, 255),    # 빨강
                "Crack": (255, 255, 0)     # 노랑
            }
            bar_colors = [tuple(c/255 for c in colors.get(defect, (100, 100, 100))) 
                         for defect in defect_types]
            
            # 그래프 생성
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # 누적 막대 그래프 생성
            bottom = np.zeros(len(date_groups))
            for i, defect in enumerate(defect_types):
                counts = [data[date][defect] for date in date_groups]
                ax.bar(date_groups, counts, bottom=bottom, 
                       color=bar_colors[i], edgecolor='black',
                       label=defect)
                bottom += counts
            
            # 그래프 제목 및 라벨 설정
            title = f'Defect Count by {group_by.capitalize()} ({start_date} to {end_date})'
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            
            # x축 레이블 회전 및 정렬
            plt.xticks(rotation=45, ha='right')
            
            # 범례 추가
            ax.legend(title='Defect Types')
            
            # 총계 표시
            total = sum(sum(data[date].values()) for date in date_groups)
            ax.text(0.95, 0.95, f'Total: {total}', 
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if conn:
                conn.close()