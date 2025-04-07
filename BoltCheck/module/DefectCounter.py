import sqlite3


class DefectCounter:

    @staticmethod
    def print_defect_counts():
        """
        defects 테이블에서 각 결함 유형별 개수를 조회하여 출력하는 함수
        """
        try:
            # 데이터베이스 연결
            conn = sqlite3.connect("bolt.db3")
            cursor = conn.cursor()

            # 결함 유형별 개수 조회 쿼리
            cursor.execute(
                """
            SELECT Defect, COUNT(*) as Count 
            FROM defects 
            GROUP BY Defect
            ORDER BY Count DESC
            """
            )

            # 결과 가져오기
            defect_counts = cursor.fetchall()

            if not defect_counts:
                print("No defect records found in the database.")
                return

            # 결과 출력
            print("\n=== Defect Count Summary ===")
            print("{:<20} | {:<10}".format("Defect Type", "Count"))
            print("-" * 32)

            for defect_type, count in defect_counts:
                print("{:<20} | {:<10}".format(defect_type, count))

            # 총계 출력
            total = sum(count for _, count in defect_counts)
            print("-" * 32)
            print("{:<20} | {:<10}".format("TOTAL", total))

        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
