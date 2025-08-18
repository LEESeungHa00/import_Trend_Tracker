📊 수입량 증감 품목 알리미
월별 수입 실적 데이터를 기반으로 품목별 증감 추이를 분석하고 시각화하는 인터랙티브 웹 대시보드입니다.

복잡한 엑셀 데이터를 자동으로 분석하여, 비즈니스 의사결정에 필요한 핵심 인사이트를 신속하게 제공하는 것을 목표로 합니다.

<br>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Badge"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas Badge"/>
<img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy Badge"/>
<img src="https://img.shields.io/badge/Google%20Sheets-34A853?style=for-the-badge&logo=google-sheets&logoColor=white" alt="Google Sheets Badge"/>
<img src="https://img.shields.io/badge/Altair-F28E2B?style=for-the-badge&logoColor=white" alt="Altair Badge"/>
</p>

✨ 주요 기능
이 대시보드는 사용자가 수입 데이터를 다각도로 분석할 수 있도록 크게 4가지 핵심 기능을 제공합니다.

1. 수입 현황 대시보드
가장 핵심적인 기능으로, 다양한 기준에 따른 수입량 변화를 비교 분석합니다.

5가지 비교 분석: 전년 대비, 전월 대비, 전년 동월 대비 등 5가지 기준의 분석을 제공합니다.

인터랙티브 컨트롤: 드롭다운 메뉴를 통해 기준 시점을 자유롭게 변경할 수 있습니다.

나비 차트 시각화: 이전 시점과 기준 시점의 수입량을 한눈에 비교할 수 있는 '나비 차트'를 제공합니다.

TOP 5 데이터: 각 기준별로 증감량(KG)이 가장 큰 상/하위 5개 품목을 테이블로 보여줍니다.

2. 기간별 수입량 분석
선택한 특정 품목들의 장기적인 수입량 추이를 시각적으로 분석합니다.

기간 단위 선택: 월별, 분기별, 반기별 중 원하는 집계 단위를 선택할 수 있습니다.

다중 품목 비교: 최대 5개 품목을 동시에 선택하여 시계열 추이를 비교할 수 있습니다.

인터랙티브 차트: 그래프에 마우스를 올리면 해당 시점의 정확한 수입량(kg)을 확인할 수 있습니다.

3. 시계열 추세 분석
단순 비교를 넘어, 선택한 기간 동안 '꾸준히' 증가하거나 감소하는 추세를 보이는 품목을 식별합니다.

두 가지 분석 모드: 연도별 장기 추세와 월별 단기 추세 분석을 제공합니다.

기간 범위 슬라이더: 시작과 끝 시점을 자유롭게 설정하여 특정 구간의 트렌드를 분석할 수 있습니다.

개별 품목 추이 확인: 분석된 TOP 10 품목 리스트에서 품목을 선택하면 실제 추이를 그래프로 확인할 수 있습니다.

4. 데이터 추가
새로운 수입 실적 데이터를 원본 데이터베이스에 안전하게 업로드합니다.

파일 형식 지원: xlsx (엑셀) 및 csv 파일을 지원합니다.

데이터 유효성 검사: 업로드된 파일의 컬럼이 표준 형식과 일치하는지 자동으로 검사하여 오류를 방지합니다.

대용량 처리: 10,000행 단위의 배치(batch) 처리 방식으로 대용량 파일도 안정적으로 업로드합니다.

<br>

🚀 시작하기
사전 준비
Python 3.8 이상 설치

Google Cloud Platform에서 서비스 계정을 생성하고, Google Sheets API 및 Google Drive API를 활성화합니다.

생성된 서비스 계정의 JSON 키 파일을 다운로드하여 Streamlit의 Secrets 관리 기능을 통해 안전하게 등록합니다.

설치 및 실행
필요한 라이브러리 설치:

pip install streamlit pandas gspread google-auth altair

Google Sheets 공유:
데이터베이스로 사용할 Google Sheets 파일을 생성하고, 발급받은 서비스 계정의 이메일 주소(client_email)로 편집자 권한을 부여하여 공유합니다.

애플리케이션 실행:
터미널에서 아래 명령어를 실행합니다.

streamlit run your_app_name.py
