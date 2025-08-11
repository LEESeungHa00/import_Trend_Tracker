import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
import io

# ---------------------------------
# 페이지 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="수입량 증감 품목 알리미",
    page_icon="📊",
    layout="wide",
)

GOOGLE_SHEET_NAME = "수입실적_DB"  # 구글 시트 파일 이름
WORKSHEET_NAME = "월별통합" # 데이터를 저장할 시트 이름

# 구글 시트 인증 및 클라이언트 객체 생성 함수
def get_google_sheet_client():
    """Streamlit의 Secrets를 사용하여 구글 시트 API에 연결하고 클라이언트 객체를 반환합니다."""
    try:
        # st.secrets에서 직접 서비스 계정 정보를 읽어옵니다.
        creds_dict = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"🚨 구글 시트 인증 중 오류가 발생했습니다: {e}")
        st.error("Streamlit Cloud의 Secrets 설정이 올바른지 확인해주세요.")
        return None

# ---------------------------------
# 데이터 로딩 및 전처리
# ---------------------------------
@st.cache_data(ttl=600) # 10분마다 데이터 캐시 갱신
def load_data():
    """구글 시트에서 데이터를 로드하고 전처리합니다."""
    client = get_google_sheet_client()
    if client is None:
        st.warning("구글 시트 연동에 실패하여 샘플 데이터로 앱을 실행합니다. 실제 데이터를 보려면 설정을 완료해주세요.")
        return create_sample_data()

    try:
        sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
        data = sheet.get_all_values()
        
        if not data or len(data) < 2:
            st.info("데이터베이스에 아직 데이터가 없습니다. '데이터 추가' 탭에서 파일을 업로드해주세요.")
            return pd.DataFrame()

        df = pd.DataFrame(data[1:], columns=data[0])
        
        if not df.empty:
            df = preprocess_dataframe(df)
        return df

    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"'{GOOGLE_SHEET_NAME}' 구글 시트를 찾을 수 없습니다. 파일 이름을 확인하거나 새로 생성해주세요.")
        return create_sample_data()
    except gspread.exceptions.WorksheetNotFound:
         st.error(f"'{WORKSHEET_NAME}' 워크시트를 찾을 수 없습니다. 시트 이름을 확인하거나 새로 생성해주세요.")
         return create_sample_data()
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return create_sample_data()

def preprocess_dataframe(df):
    """데이터프레임 전처리 (날짜 변환, 숫자 변환 등)"""
    df['총 중량(KG)'] = pd.to_numeric(df['총 중량(KG)'].astype(str).str.replace(',', ''), errors='coerce')
    df.dropna(subset=['총 중량(KG)'], inplace=True)
    df['총 중량(KG)'] = df['총 중량(KG)'].astype(float)

    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df.dropna(subset=['날짜'], inplace=True)

    df['연도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    df['분기'] = df['날짜'].dt.quarter
    df['반기'] = (df['날짜'].dt.month - 1) // 6 + 1
    
    return df

def create_sample_data():
    """앱 시연을 위한 샘플 데이터 생성"""
    items = ['소고기(냉장)', '바지락(활)', '김치', '과자', '맥주', '새우(냉동)', '오렌지', '바나나', '커피원두', '치즈']
    daterange = pd.date_range(start='2021-01-01', end='2025-07-31', freq='M')
    data = []
    for date in daterange:
        for item in items:
            base_weight = 10000 + items.index(item) * 5000
            monthly_factor = 1 + np.sin(date.month * np.pi / 6) * 0.2
            trend_factor = 1 + (date.year - 2021) * 0.1 + date.month / 12 * 0.1
            random_factor = np.random.uniform(0.8, 1.2)
            weight = base_weight * monthly_factor * trend_factor * random_factor
            data.append([date, item, weight])
    
    df = pd.DataFrame(data, columns=['날짜', '대표품목별', '총 중량(KG)'])
    df = preprocess_dataframe(df)
    return df

# ---------------------------------
# 사이드바 메뉴
# ---------------------------------
st.sidebar.title("메뉴")
menu = st.sidebar.radio(
    "원하는 기능을 선택하세요.",
    ("수입 현황 대시보드", "기간별 수입량 분석", "데이터 추가")
)

# ---------------------------------
# 메인 데이터 로드
# ---------------------------------
df = load_data()

if df.empty and menu != "데이터 추가":
    st.warning("데이터가 없습니다. '데이터 추가' 탭으로 이동하여 엑셀 파일을 업로드해주세요.")
    st.stop()

# ---------------------------------
# 탭 1: 수입 현황 대시보드
# ---------------------------------
if menu == "수입 현황 대시보드":
    st.title("📊 수입 현황 대시보드")
    st.markdown("---")

    latest_date = df['날짜'].max()
    latest_year = latest_date.year
    latest_month = latest_date.month

    st.header(f"🥇 {latest_year}년 누적 수입량 TOP 5 품목")
    
    top5_this_year = df[df['연도'] == latest_year].groupby('대표품목별')['총 중량(KG)'].sum().nlargest(5)
    
    cols = st.columns(5)
    for i, (item, weight) in enumerate(top5_this_year.items()):
        with cols[i]:
            st.metric(label=f"{i+1}. {item}", value=f"{weight:,.0f} kg")

    st.markdown("---")
    st.header(f"📈 {latest_year}년 {latest_month}월 수입량 증감 분석")
    
    current_month_start = datetime(latest_year, latest_month, 1)
    prev_month_date = current_month_start - pd.DateOffset(months=1)
    prev_year_date = current_month_start - pd.DateOffset(years=1)

    # --- 오류 수정 부분 ---
    # 파이썬 datetime 객체를 Pandas Timestamp로 변환 후 to_period() 호출
    current_period = pd.Timestamp(current_month_start).to_period('M')
    prev_month_period = pd.Timestamp(prev_month_date).to_period('M')
    prev_year_period = pd.Timestamp(prev_year_date).to_period('M')

    # 기간별 데이터 필터링
    current_data = df[df['날짜'].dt.to_period('M') == current_period]
    prev_month_data = df[df['날짜'].dt.to_period('M') == prev_month_period]
    prev_year_data = df[df['날짜'].dt.to_period('M') == prev_year_period]
    # --- 오류 수정 완료 ---

    current_agg = current_data.groupby('대표품목별')['총 중량(KG)'].sum()
    prev_month_agg = prev_month_data.groupby('대표품목별')['총 중량(KG)'].sum()
    prev_year_agg = prev_year_data.groupby('대표품목별')['총 중량(KG)'].sum()

    analysis_df = pd.DataFrame(current_agg).rename(columns={'총 중량(KG)': '현재월_중량'})
    analysis_df = analysis_df.join(prev_month_agg.rename('전월_중량'), how='outer')
    analysis_df = analysis_df.join(prev_year_agg.rename('전년동월_중량'), how='outer')
    analysis_df.fillna(0, inplace=True)

    analysis_df['전월대비_증감량'] = analysis_df['현재월_중량'] - analysis_df['전월_중량']
    analysis_df['전년동월대비_증감량'] = analysis_df['현재월_중량'] - analysis_df['전년동월_중량']
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🆚 전월 대비 (vs " + f"{prev_month_date.year}년 {prev_month_date.month}월)")
        st.write("🔼 **수입량 증가 TOP 5**")
        top5_increase_mom = analysis_df.nlargest(5, '전월대비_증감량')
        st.dataframe(top5_increase_mom[['현재월_중량', '전월_중량', '전월대비_증감량']].style.format("{:,.0f}"))
        st.write("🔽 **수입량 감소 TOP 5**")
        top5_decrease_mom = analysis_df.nsmallest(5, '전월대비_증감량')
        st.dataframe(top5_decrease_mom[['현재월_중량', '전월_중량', '전월대비_증감량']].style.format("{:,.0f}"))

    with col2:
        st.subheader("🆚 전년 동월 대비 (vs " + f"{prev_year_date.year}년 {prev_year_date.month}월)")
        st.write("🔼 **수입량 증가 TOP 5**")
        top5_increase_yoy = analysis_df.nlargest(5, '전년동월대비_증감량')
        st.dataframe(top5_increase_yoy[['현재월_중량', '전년동월_중량', '전년동월대비_증감량']].style.format("{:,.0f}"))
        st.write("🔽 **수입량 감소 TOP 5**")
        top5_decrease_yoy = analysis_df.nsmallest(5, '전년동월대비_증감량')
        st.dataframe(top5_decrease_yoy[['현재월_중량', '전년동월_중량', '전년동월대비_증감량']].style.format("{:,.0f}"))

# ---------------------------------
# 탭 2: 기간별 수입량 분석
# ---------------------------------
elif menu == "기간별 수입량 분석":
    st.title("📆 기간별 수입량 변화 분석")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        period_type = st.radio("분석 기간 단위를 선택하세요.", ('월별', '분기별', '반기별'), horizontal=True, key="period_type")
    
    with col2:
        if period_type == '월별':
            selected_period = st.selectbox("분석할 월을 선택하세요.", range(1, 13), format_func=lambda x: f"{x}월", key="month_select")
            period_col = '월'
        elif period_type == '분기별':
            selected_period = st.selectbox("분석할 분기를 선택하세요.", range(1, 5), format_func=lambda x: f"{x}분기", key="quarter_select")
            period_col = '분기'
        else:
            selected_period = st.selectbox("분석할 반기를 선택하세요.", range(1, 3), format_func=lambda x: f"{x}반기", key="half_select")
            period_col = '반기'
    
    period_df = df[df[period_col] == selected_period]
    pivot_df = period_df.pivot_table(index='대표품목별', columns='연도', values='총 중량(KG)', aggfunc='sum').fillna(0)
    pivot_df['변화폭(표준편차)'] = pivot_df.std(axis=1)
    pivot_df.sort_values('변화폭(표준편차)', ascending=False, inplace=True)
    
    st.markdown("---")
    st.header(f"📊 {selected_period}{'월' if period_type=='월별' else ''} 연도별 수입량 변화폭 TOP 10 품목")
    st.info("연도별 수입량의 표준편차가 큰 순서대로 정렬됩니다. 변화가 클수록 순위가 높습니다.")
    
    display_cols = [col for col in pivot_df.columns if col != '변화폭(표준편차)']
    st.dataframe(pivot_df[display_cols].head(10).style.format("{:,.0f}"))

    st.markdown("---")
    st.header("📈 품목별 연도별 수입량 추이 비교")
    
    top_items = pivot_df.index.tolist()
    default_selection = top_items[:3] if len(top_items) >= 3 else top_items
    selected_items = st.multiselect(
        "비교할 품목을 선택하세요 (최대 5개).", 
        top_items, 
        default=default_selection,
        max_selections=5
    )

    if selected_items:
        chart_type = st.radio("차트 종류 선택", ('선 그래프', '막대 그래프'), horizontal=True, key="chart_type")
        chart_data = pivot_df.loc[selected_items, display_cols]
        
        if chart_type == '선 그래프':
            st.line_chart(chart_data.T)
        else:
            st.bar_chart(chart_data.T)
        
        with st.expander("데이터 상세 보기"):
            st.subheader("수입량 원본 데이터 (단위: KG)")
            st.dataframe(chart_data.style.format("{:,.0f}"))
            st.subheader("전년 대비 증감률 (%)")
            growth_rate_df = chart_data.pct_change(axis='columns') * 100
            st.dataframe(growth_rate_df.style.format("{:.2f}%").highlight_null(null_color='transparent').format(na_rep="-"))
    else:
        st.info("차트를 보려면 위에서 품목을 하나 이상 선택해주세요.")

# ---------------------------------
# 탭 3: 데이터 추가
# ---------------------------------
elif menu == "데이터 추가":
    st.title("📤 데이터 추가")
    st.markdown("---")
    st.info("월별 수입량 데이터가 포함된 엑셀(xlsx) 또는 CSV 파일을 업로드해주세요.")
    
    uploaded_file = st.file_uploader("파일 선택", type=['xlsx', 'csv'])
    password = st.text_input("업로드 비밀번호를 입력하세요.", type="password")

    if st.button("데이터베이스에 추가"):
        if uploaded_file is not None and password == "1004":
            with st.spinner('파일을 읽고 처리하는 중입니다...'):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        new_df = pd.read_csv(uploaded_file)
                    else:
                        new_df = pd.read_excel(uploaded_file)
                    
                    try:
                        date_str = ''.join(filter(str.isdigit, uploaded_file.name))[:8]
                        file_date = pd.to_datetime(date_str, format='%Y%m%d')
                        new_df['날짜'] = file_date
                    except:
                        st.error("파일명에서 날짜를 인식할 수 없습니다. 'YYYYMMDD' 형식이 포함되어야 합니다.")
                        st.stop()
                        
                    required_cols = ['대표품목별', '총 중량(KG)', '날짜']
                    if not all(col in new_df.columns for col in required_cols):
                        st.error(f"업로드한 파일에 필수 컬럼({', '.join(required_cols)})이 없습니다.")
                        st.stop()

                    new_df_processed = preprocess_dataframe(new_df[required_cols])
                    
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
                    st.stop()

            with st.spinner('데이터베이스에 연결하여 데이터를 추가하는 중입니다...'):
                try:
                    client = get_google_sheet_client()
                    if client:
                        if df.empty:
                            combined_df = new_df_processed
                        else:
                            df_filtered = df[df['날짜'].dt.date != file_date.date()]
                            combined_df = pd.concat([df_filtered, new_df_processed], ignore_index=True)
                        
                        combined_df.sort_values(by='날짜', inplace=True)
                        combined_df['날짜'] = combined_df['날짜'].dt.strftime('%Y-%m-%d')

                        sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
                        sheet.clear()
                        set_with_dataframe(sheet, combined_df)
                        
                        st.success(f"✅ 데이터베이스에 성공적으로 {len(new_df_processed)}개의 행이 추가/업데이트되었습니다!")
                        st.info("캐시된 데이터가 갱신되려면 잠시 기다리거나 페이지를 새로고침해주세요.")
                        st.cache_data.clear()
                    else:
                        st.error("🚨 구글 시트 연결에 실패하여 데이터를 추가할 수 없습니다.")

                except Exception as e:
                    st.error(f"데이터베이스 업데이트 중 오류가 발생했습니다: {e}")

        elif not uploaded_file:
            st.warning("⚠️ 파일을 먼저 업로드해주세요.")
        else:
            st.error("🚨 비밀번호가 틀렸습니다.")

