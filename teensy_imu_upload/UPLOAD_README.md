# Arduino IDE 업로드용 (Teensy 4.0 + GY-9250)

이 폴더 전체를 복사하세요. 폴더명 teensy_imu_upload와 스케치명 teensy_imu_upload.ino를 동일하게 유지합니다. 원본은 imu/src/main.cpp이며 이 스케치는 해당 펌웨어의 복사본입니다. 원본을 수정했다면 이 복사본도 갱신해야 합니다.

## Arduino IDE가 되는 Windows 계정에서

1. Arduino IDE 2.x를 실행합니다.
2. Teensy 보드가 없다면 파일 > 기본 설정 > 추가 보드 매니저 URL에 아래 주소를 추가합니다.

   https://www.pjrc.com/teensy/package_teensy_index.json

3. 보드 매니저에서 Teensy를 검색해 설치합니다. 이미 설치했다면 생략합니다.
4. 이 폴더의 teensy_imu_upload.ino를 엽니다.
5. 도구 > 보드에서 Teensy 4.0, USB Type은 Serial을 선택합니다. 다른 옵션은 기본값을 유지합니다.
6. USB 데이터 케이블로 Teensy를 연결하고 해당 포트를 선택합니다.
7. 확인(컴파일) 후 업로드합니다. 자동 업로드가 안 되면 Teensy의 Program 버튼을 누릅니다.
8. 시리얼 모니터를 115200으로 열어 JSON 출력과 chip, mag_present 값을 확인합니다. 정상 GY-9250이면 MPU9250 및 true가 예상됩니다. 다른 값이면 배선/실물을 확인합니다.
9. 시리얼 모니터와 Arduino IDE를 닫고 로그아웃합니다. 계정 전환만 하면 프로그램이 남아 COM 포트를 점유할 수 있습니다.

별도 MPU9250 라이브러리는 필요 없으며 Teensy 보드 패키지의 Arduino/Wire를 사용합니다. Arduino IDE 1.8.x는 설치 방식이 다르므로 아래 공식 안내를 참고하세요.

공식 설치 안내: https://www.pjrc.com/teensy/td_download.html

## VS Code가 되는 계정으로 돌아온 뒤

Teensy USB를 다시 연결합니다. 업로드된 코드는 전원을 꺼도 유지됩니다. Python 실행에 Arduino IDE는 필요 없습니다.

```powershell
cd "C:\Users\현희섭\Downloads\한이음_2026\imu"
.\.venv\Scripts\Activate.ps1
python bridge.py --list-ports
python bridge.py --port COM5
```

COM5는 예시이며 출력된 실제 번호를 사용하세요. .venv가 없다면 먼저 python -m venv .venv, 활성화 후 python -m pip install -r requirements.txt를 실행합니다. 실행 후 로봇을 약 2초간 완전히 정지시키세요. bridge.py --demo가 실행 중이면 먼저 Ctrl+C로 종료하세요.

## 배선

전원을 빼고 연결합니다. 센서 VCC/NCS -> Teensy 3.3V, GND/AD0/FSYNC -> GND, SDA -> 18, SCL -> 19. EDA/ECL/INT는 미연결. 모듈의 3.3V 입력 지원을 확인하고 Teensy 신호핀에 5V를 넣지 마세요.

이 패키지는 업로드용 소스와 설명서뿐이며 Python 환경/기록/개인 설정은 포함하지 않습니다. 실제 Teensy 컴파일·업로드·실물 측정은 아직 검증하지 않았습니다.
