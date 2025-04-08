#plc 연결

import pymcprotocol

# PLC 연결 설정 (IP 주소와 포트)
mc = pymcprotocol.Type3E()
mc.connect("192.168.0.1", 5007)  # PLC의 IP와 포트번호

# PLC에서 데이터 읽기 (D100 레지스터 값 읽기)
data = mc.batchread_wordunits(headdevice="D100", readsize=10)
print("D100~D109 값:", data)

# PLC에 데이터 쓰기 (D200에 값 1234 저장)
mc.batchwrite_wordunits(headdevice="D200", values=[1234])
print("D200에 값 1234 저장 완료!")

# M100~M109 비트 값 읽기
bit_data = mc.batchread_bitunits(headdevice="M100", readsize=10)
print("M100~M109 상태:", bit_data)

# M100 = 1 (ON), M101 = 0 (OFF)
mc.batchwrite_bitunits(headdevice="M100", values=[1, 0])

print("M100 ON, M101 OFF 설정 완료!")


mc.close()