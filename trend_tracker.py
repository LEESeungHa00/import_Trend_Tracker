import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import time

# ---------------------------------
# 페이지 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="수입량 증감 품목 알리미",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------
# 상수 정의
# ---------------------------------
PRIMARY_WEIGHT_COL = '적합 중량(KG)'
DESIRED_HEADER = [
    'NO', 'Year', 'Month', '제품구분별', '제조국(원산지)별', '수출국별',
    '수입용도별', '대표품목별', '총 중량(KG)', '총 금액($)', '적합 중량(KG)',
    '적합 금액($)', '부적합 중량(KG)', '부적합 금액($)'
]
GOOGLE_SHEET_NAME = "수입실적_데이터베이스"  # 본인의 구글 시트 이름으로 변경
WORKSHEET_NAME = "월별통합"

# ---------------------------------
# 구글 시트 연동 설정
# ---------------------------------
def get_google_sheet_client():
    """Streamlit의 Secrets를 사용하여 구글 시트 API에 연결하고 클라이언트 객체를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"🚨 구글 시트 인증 중 오류가 발생했습니다: {e}")
        return None

# ---------------------------------
# 데이터 로딩 및 전처리 (안정화 버전)
# ---------------------------------
def preprocess_dataframe(df):
    """
    데이터프레임을 안정적으로 전처리합니다.
    - SettingWithCopyWarning 방지를 위해 복사본 사용
    - errors='coerce'를 활용하여 타입 변환 에러를 NaN으로 처리
    - '날짜' 관련 컬럼을 안전하게 생성
    """
    df_copy = df.copy()

    numeric_cols = [
        '총 중량(KG)', '총 금액($)', '적합 중량(KG)', '적합 금액($)',
        '부적합 중량(KG)', '부적합 금액($)'
    ]
    for col in numeric_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(',', ''), 
                errors='coerce'
            ).fillna(0)

    # 'Year', 'Month' 컬럼이 존재할 경우에만 처리
    if 'Year' in df_copy.columns and 'Month' in df_copy.columns:
        df_copy['Year'] = pd.to_numeric(df_copy['Year'], errors='coerce')
        df_copy['Month'] = pd.to_numeric(df_copy['Month'], errors='coerce')

        # '날짜' 컬럼 생성 (잘못된 값이 있어도 NaT로 처리되어 에러 방지)
        df_copy['날짜'] = pd.to_datetime(
            df_copy['Year'].astype('Int64').astype(str) + '-' + df_copy['Month'].astype('Int64').astype(str) + '-01',
            errors='coerce'
        )

        # 날짜가 유효한 경우에만 파생 컬럼 생성
        valid_dates = df_copy['날짜'].notna()
        df_copy.loc[valid_dates, '연도'] = df_copy.loc[valid_dates, '날짜'].dt.year
        df_copy.loc[valid_dates, '월'] = df_copy.loc[valid_dates, '날짜'].dt.month
        df_copy.loc[valid_dates, '분기'] = df_copy.loc[valid_dates, '날짜'].dt.quarter
        df_copy.loc[valid_dates, '반기'] = (df_copy.loc[valid_dates, '날짜'].dt.month - 1) // 6 + 1

    return df_copy


@st.cache_data(ttl=600)
def load_data():
    """구글 시트에서 데이터를 로드하고 전처리합니다. 헤더 유효성 검사를 강화합니다."""
    client = get_google_sheet_client()
    if client is None:
        st.warning("구글 시트 연동에 실패하여 샘플 데이터로 앱을 실행합니다.")
        return create_sample_data()
    try:
        sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
        all_data = sheet.get_all_values()

        if not all_data:
            st.info("시트가 비어있습니다.")
            return pd.DataFrame()

        header = all_data[0]
        data = all_data[1:]

        # 헤더 유효성 검사
        desired_set = set(DESIRED_HEADER)
        header_set = set(header)
        if desired_set != header_set:
            missing = desired_set - header_set
            extra = header_set - desired_set
            error_message = "🚨 구글 시트의 컬럼 구성이 올바르지 않습니다.\n"
            if missing:
                error_message += f"\n**- 누락된 컬럼:** `{', '.join(missing)}`"
            if extra:
                error_message += f"\n**- 불필요한 컬럼:** `{', '.join(extra)}`"
            st.error(error_message)
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=header)
        df.dropna(how='all', inplace=True)

        if not df.empty:
            df = preprocess_dataframe(df)
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"🚨 구글 시트 파일을 찾을 수 없습니다. 이름이 '{GOOGLE_SHEET_NAME}'인지, 서비스 계정에 공유되었는지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 로딩 중 예상치 못한 오류 발생: {e}")
        return create_sample_data()

def create_sample_data():
    """앱 시연을 위한 샘플 데이터 생성"""
    items = ['소고기(냉장)', '바지락(활)', '김치', '과자', '맥주', '새우(냉동)', '오렌지', '바나나', '커피원두', '치즈']
    daterange = pd.date_range(start='2021-01-01', end='2025-07-31', freq='M')
    data = []
    no_counter = 1
    for date in daterange:
        for item in items:
            weight = (10000 + items.index(item) * 5000) * np.random.uniform(0.8, 1.2)
            price = weight * np.random.uniform(5, 10)
            data.append([
                no_counter, date.year, date.month, '가공품', '미국', '미국', '판매용',
                item, weight, price, weight*0.95, price*0.95, weight*0.05, price*0.05
            ])
            no_counter += 1
    
    df = pd.DataFrame(data, columns=DESIRED_HEADER)
    df = preprocess_dataframe(df)
    return df

# ---------------------------------
# 대용량 데이터 업로드 함수
# ---------------------------------
def update_sheet_in_batches(worksheet, dataframe, batch_size=5000):
    """데이터프레임을 작은 배치로 나누어 구글 시트에 업로드합니다."""
    worksheet.clear()
    worksheet.update([dataframe.columns.values.tolist()] + dataframe.fillna('').values.tolist())
    st.success("✅ 데이터베이스 업로드 완료!")


# ---------------------------------
# ---- 메인 애플리케이션 로직 ----
# ---------------------------------

# 사이드바
st.sidebar.title("메뉴")
menu = st.sidebar.radio(
    "원하는 기능을 선택하세요.",
    ("수입 현황 대시보드", "기간별 수입량 분석", "데이터 추가")
)
if st.sidebar.button("🔄 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

# 데이터 로딩
df = load_data()

# 데이터가 없을 경우 메시지 표시
if df.empty and menu != "데이터 추가":
    st.warning("데이터가 없습니다. '데이터 추가' 탭으로 이동하여 엑셀 파일을 업로드해주세요.")
    st.stop()

# 1. 수입 현황 대시보드
if menu == "수입 현황 대시보드":
    st.title(f"📊 수입 현황 대시보드 (기준: {PRIMARY_WEIGHT_COL})")
    st.markdown("---")

    analysis_df_raw = df.dropna(subset=['날짜', PRIMARY_WEIGHT_COL])
    if analysis_df_raw.empty:
        st.warning("분석할 유효한 데이터가 없습니다. 'Year', 'Month' 데이터가 올바른지 확인해주세요.")
        st.stop()

    latest_date = analysis_df_raw['날짜'].max()
    latest_year = int(latest_date.year)
    latest_month = int(latest_date.month)

    st.header(f"🏆 {latest_year}년 누적 수입량 TOP 5 품목")
    top5_this_year = analysis_df_raw[analysis_df_raw['연도'] == latest_year].groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum().nlargest(5)
    cols = st.columns(len(top5_this_year) if len(top5_this_year) > 0 else 1)
    for i, (item, weight) in enumerate(top5_this_year.items()):
        with cols[i]:
            st.metric(label=f"{i+1}. {item}", value=f"{weight:,.0f} kg")

    st.markdown("---")
    st.header(f"📈 {latest_year}년 {latest_month}월 수입량 증감 분석")
    
    current_month_start = datetime(latest_year, latest_month, 1)
    prev_month_date = current_month_start - pd.DateOffset(months=1)
    prev_year_date = current_month_start - pd.DateOffset(years=1)

    current_period = pd.Timestamp(current_month_start).to_period('M')
    prev_month_period = pd.Timestamp(prev_month_date).to_period('M')
    prev_year_period = pd.Timestamp(prev_year_date).to_period('M')

    current_data = analysis_df_raw[analysis_df_raw['날짜'].dt.to_period('M') == current_period]
    prev_month_data = analysis_df_raw[analysis_df_raw['날짜'].dt.to_period('M') == prev_month_period]
    prev_year_data = analysis_df_raw[analysis_df_raw['날짜'].dt.to_period('M') == prev_year_period]

    current_agg = current_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    prev_month_agg = prev_month_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    prev_year_agg = prev_year_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()

    agg_df = pd.DataFrame(current_agg).rename(columns={PRIMARY_WEIGHT_COL: '현재월_중량'})
    agg_df = agg_df.join(prev_month_agg.rename('전월_중량'), how='outer')
    agg_df = agg_df.join(prev_year_agg.rename('전년동월_중량'), how='outer')
    agg_df.fillna(0, inplace=True)

    agg_df['전월대비_증감량'] = agg_df['현재월_중량'] - agg_df['전월_중량']
    agg_df['전년동월대비_증감량'] = agg_df['현재월_중량'] - agg_df['전년동월_중량']
    agg_df['전월대비_증감률'] = (agg_df['전월대비_증감량'] / agg_df['전월_중량'].replace(0, np.nan))
    agg_df['전년동월대비_증감률'] = (agg_df['전년동월대비_증감량'] / agg_df['전년동월_중량'].replace(0, np.nan))
    
    col1, col2 = st.columns(2)
    
    formatter = {
        '현재월_중량': '{:,.0f}', '전월_중량': '{:,.0f}', '전년동월_중량': '{:,.0f}',
        '전월대비_증감량': '{:+,.0f}', '전년동월대비_증감량': '{:+,.0f}',
        '전월대비_증감률': '{:+.2%}', '전년동월대비_증감률': '{:+.2%}'
    }
    
    with col1:
        st.subheader(f"🆚 전월 대비 (vs {prev_month_date.month}월)")
        st.markdown('<p style="color:red; font-weight:bold;">🔼 수입량 증가 TOP 5</p>', unsafe_allow_html=True)
        st.dataframe(agg_df.nlargest(5, '전월대비_증감량')[['현재월_중량', '전월_중량', '전월대비_증감량', '전월대비_증감률']].style.format(formatter, na_rep="-"))
        
        st.markdown('<p style="color:blue; font-weight:bold;">🔽 수입량 감소 TOP 5</p>', unsafe_allow_html=True)
        st.dataframe(agg_df.nsmallest(5, '전월대비_증감량')[['현재월_중량', '전월_중량', '전월대비_증감량', '전월대비_증감률']].style.format(formatter, na_rep="-"))

    with col2:
        st.subheader(f"🆚 전년 동월 대비 (vs {prev_year_date.year}년)")
        st.markdown('<p style="color:red; font-weight:bold;">🔼 수입량 증가 TOP 5</p>', unsafe_allow_html=True)
        st.dataframe(agg_df.nlargest(5, '전년동월대비_증감량')[['현재월_중량', '전년동월_중량', '전년동월대비_증감량', '전년동월대비_증감률']].style.format(formatter, na_rep="-"))
        
        st.markdown('<p style="color:blue; font-weight:bold;">🔽 수입량 감소 TOP 5</p>', unsafe_allow_html=True)
        st.dataframe(agg_df.nsmallest(5, '전년동월대비_증감량')[['현재월_중량', '전년동월_중량', '전년동월대비_증감량', '전년동월대비_증감률']].style.format(formatter, na_rep="-"))

# 2. 기간별 수입량 분석
elif menu == "기간별 수입량 분석":
    st.title(f"📆 기간별 수입량 변화 분석 (기준: {PRIMARY_WEIGHT_COL})")
    st.markdown("---")

    analysis_df = df.dropna(subset=['날짜', PRIMARY_WEIGHT_COL, '분기', '반기'])
    if analysis_df.empty:
        st.warning("분석할 유효한 데이터가 없습니다.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        period_type = st.radio("분석 기간 단위", ('월별', '분기별', '반기별'), horizontal=True)
    with col2:
        if period_type == '월별':
            selected_period = st.selectbox("월 선택", range(1, 13), format_func=lambda x: f"{x}월")
            period_col = '월'
        elif period_type == '분기별':
            selected_period = st.selectbox("분기 선택", range(1, 5), format_func=lambda x: f"{x}분기")
            period_col = '분기'
        else:
            selected_period = st.selectbox("반기 선택", options=[1, 2], format_func=lambda x: '상반기' if x == 1 else '하반기')
            period_col = '반기'
    
    period_df = analysis_df[analysis_df[period_col] == selected_period]
    pivot_df = period_df.pivot_table(index='대표품목별', columns='연도', values=PRIMARY_WEIGHT_COL, aggfunc='sum').fillna(0)
    pivot_df['변화폭(표준편차)'] = pivot_df.std(axis=1)
    pivot_df.sort_values('변화폭(표준편차)', ascending=False, inplace=True)
    
    st.header("📈 품목별 연도별 수입량 추이 비교")
    
    if 'selected_items_memory' not in st.session_state:
        st.session_state.selected_items_memory = []

    top_items = pivot_df.index.tolist()
    valid_selection = [item for item in st.session_state.selected_items_memory if item in top_items]
    if not valid_selection and top_items:
        valid_selection = top_items[:min(5, len(top_items))]
    st.session_state.selected_items_memory = valid_selection

    selected_items = st.multiselect(
        "품목 선택 (최대 5개)",
        options=top_items,
        placeholder="수입량을 확인할 품목을 선택해주세요",
        default=st.session_state.selected_items_memory,
        max_selections=5
    )
    st.session_state.selected_items_memory = selected_items

    if selected_items:
        chart_type = st.radio("차트 종류", ('선 그래프', '막대 그래프'), horizontal=True)
        chart_data = pivot_df.loc[selected_items].drop(columns=['변화폭(표준편차)'])
        
        if chart_type == '선 그래프': st.line_chart(chart_data.T)
        else: st.bar_chart(chart_data.T)
        
        with st.expander("데이터 상세 보기"):
            st.subheader("연도별 수입량 (KG)")
            st.dataframe(chart_data.style.format("{:,.0f}"))
            
            st.subheader("전년 대비 증감률 (%)")
            growth_rate_df = chart_data.pct_change(axis='columns')
            st.dataframe(growth_rate_df.style.format("{:+.2%}", na_rep="-"))

# 3. 데이터 추가
elif menu == "데이터 추가":
    st.title("📤 데이터 추가")
    st.info(f"다음 컬럼을 포함한 엑셀/CSV 파일을 업로드해주세요:\n`{', '.join(DESIRED_HEADER)}`")
    
    uploaded_file = st.file_uploader("파일 선택", type=['xlsx', 'csv'])
    password = st.text_input("업로드 비밀번호", type="password")

    if st.button("데이터베이스에 추가"):
        if uploaded_file and password == "1004": # 비밀번호는 실제 환경에 맞게 변경하세요.
            try:
                st.info("파일을 읽고 처리하는 중입니다...")
                if uploaded_file.name.endswith('.csv'):
                    new_df = pd.read_csv(uploaded_file, dtype=str)
                else:
                    new_df = pd.read_excel(uploaded_file, dtype=str)
                
                # 업로드 파일 헤더 검사
                desired_set = set(DESIRED_HEADER)
                new_df_set = set(new_df.columns)
                if desired_set != new_df_set:
                    missing = desired_set - new_df_set
                    extra = new_df_set - desired_set
                    error_message = "🚨 업로드한 파일의 컬럼 구성이 올바르지 않습니다.\n"
                    if missing:
                        error_message += f"\n**- 누락된 컬럼:** `{', '.join(missing)}`"
                    if extra:
                        error_message += f"\n**- 불필요한 컬럼:** `{', '.join(extra)}`"
                    st.error(error_message)
                    st.stop()

                new_df_processed = preprocess_dataframe(new_df)
                
                client = get_google_sheet_client()
                if client:
                    sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
                    
                    # 기존 데이터와 새 데이터 병합 (중복 월 제거)
                    unique_periods = new_df_processed.dropna(subset=['연도', '월'])[['연도', '월']].drop_duplicates()
                    df_filtered = df.copy()
                    if not df_filtered.empty and not unique_periods.empty:
                        for _, row in unique_periods.iterrows():
                            df_filtered = df_filtered[~((df_filtered['연도'] == row['연도']) & (df_filtered['월'] == row['월']))]
                    
                    combined_df = pd.concat([df_filtered, new_df_processed], ignore_index=True)
                    combined_df.sort_values(by=['Year', 'Month', 'NO'], inplace=True, na_position='last')
                    
                    # 시트에 쓸 데이터는 최종적으로 DESIRED_HEADER 순서에 맞춤
                    df_to_write = combined_df.reindex(columns=DESIRED_HEADER)

                    update_sheet_in_batches(sheet, df_to_write)
                    st.cache_data.clear()
                else:
                    st.error("🚨 구글 시트 연결에 실패했습니다.")

            except Exception as e:
                st.error(f"데이터 처리/업로드 중 오류 발생: {e}")
        else:
            if not uploaded_file:
                st.warning("⚠️ 파일을 먼저 업로드해주세요.")
            else:
                st.error("🚨 비밀번호가 틀렸습니다.")
