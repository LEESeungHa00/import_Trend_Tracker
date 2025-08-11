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
GOOGLE_SHEET_NAME = "수입실적_DB"  
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

    if 'Year' in df_copy.columns and 'Month' in df_copy.columns:
        df_copy['Year'] = pd.to_numeric(df_copy['Year'], errors='coerce')
        df_copy['Month'] = pd.to_numeric(df_copy['Month'], errors='coerce')

        df_copy['날짜'] = pd.to_datetime(
            df_copy['Year'].astype('Int64').astype(str) + '-' + df_copy['Month'].astype('Int64').astype(str) + '-01',
            errors='coerce'
        )

        valid_dates = df_copy['날짜'].notna()
        df_copy.loc[valid_dates, '연도'] = df_copy.loc[valid_dates, '날짜'].dt.year
        df_copy.loc[valid_dates, '월'] = df_copy.loc[valid_dates, '날짜'].dt.month
        df_copy.loc[valid_dates, '분기'] = df_copy.loc[valid_dates, '날짜'].dt.quarter
        df_copy.loc[valid_dates, '반기'] = (df_copy.loc[valid_dates, '날짜'].dt.month - 1) // 6 + 1

    return df_copy


@st.cache_data(ttl=600)
def load_data():
    client = get_google_sheet_client()
    if client is None:
        st.warning("구글 시트 연동에 실패하여 샘플 데이터로 앱을 실행합니다.")
        return create_sample_data()
    try:
        sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
        all_data = sheet.get_all_values()

        if not all_data or len(all_data) < 2:
            st.info("시트에 헤더 또는 데이터가 없습니다.")
            return pd.DataFrame()

        header = all_data[0]
        data = all_data[1:]
        desired_set = set(DESIRED_HEADER)
        header_set = set(header)
        if desired_set != header_set:
            missing = desired_set - header_set
            extra = header_set - desired_set
            error_message = "🚨 구글 시트의 컬럼 구성이 올바르지 않습니다.\n"
            if missing: error_message += f"\n**- 누락된 컬럼:** `{', '.join(missing)}`"
            if extra: error_message += f"\n**- 불필요한 컬럼:** `{', '.join(extra)}`"
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

def update_sheet_in_batches(worksheet, dataframe):
    worksheet.clear()
    worksheet.update([dataframe.columns.values.tolist()] + dataframe.fillna('').values.tolist(), value_input_option='USER_ENTERED')
    st.success("✅ 데이터베이스 업로드 완료!")


# ---------------------------------
# ---- 메인 애플리케이션 로직 ----
# ---------------------------------

st.sidebar.title("메뉴")
menu = st.sidebar.radio(
    "원하는 기능을 선택하세요.",
    ("수입 현황 대시보드", "기간별 수입량 분석", "데이터 추가")
)
if st.sidebar.button("🔄 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

df = load_data()

if df.empty and menu != "데이터 추가":
    st.warning("데이터가 없습니다. '데이터 추가' 탭으로 이동하여 데이터를 업로드해주세요.")
    st.stop()

# ----------------------------------------------------------------
# ★★★ 여기가 수정된 부분입니다 (수입 현황 대시보드) ★★★
# ----------------------------------------------------------------
if menu == "수입 현황 대시보드":
    st.title(f"📊 수입 현황 대시보드 (기준: {PRIMARY_WEIGHT_COL})")
    st.markdown("---")

    analysis_df_raw = df.dropna(subset=['날짜', PRIMARY_WEIGHT_COL, '연도', '분기', '반기'])
    if analysis_df_raw.empty:
        st.warning("분석할 유효한 데이터가 없습니다. 'Year', 'Month' 데이터가 올바른지 확인해주세요.")
        st.stop()
    
    # 공통으로 사용할 변수들
    available_years = sorted(analysis_df_raw['연도'].unique().astype(int), reverse=True)
    available_months = sorted(analysis_df_raw['월'].unique().astype(int))
    latest_date = analysis_df_raw['날짜'].max()
    
    def display_comparison_table(df_agg, base_name):
        """결과 데이터프레임을 받아 수직 정렬된 테이블을 표시하는 함수"""
        formatter = {f'기준{base_name}_중량': '{:,.0f}', f'이전{base_name}_중량': '{:,.0f}',
                     '증감량': '{:+,.0f}', '증감률': '{:+.2%}'}
        st.markdown('<p style="color:red; font-weight:bold;">🔼 수입량 증가 TOP 5</p>', unsafe_allow_html=True)
        st.dataframe(df_agg.nlargest(5, '증감량').style.format(formatter, na_rep="-"))
        st.markdown('<p style="color:blue; font-weight:bold;">🔽 수입량 감소 TOP 5</p>', unsafe_allow_html=True)
        st.dataframe(df_agg.nsmallest(5, '증감량').style.format(formatter, na_rep="-"))
        st.markdown("---")

    # --- 1. 전월 대비 분석 ---
    title_col, year_col, month_col = st.columns([0.5, 0.25, 0.25])
    with title_col:
        st.subheader("🆚 전월 대비")
    with year_col:
        mom_year = st.selectbox("연도", available_years, key="mom_year",
                                index=available_years.index(latest_date.year))
    with month_col:
        mom_month = st.selectbox("월", available_months, key="mom_month",
                                 index=available_months.index(latest_date.month))

    current_date = datetime(mom_year, mom_month, 1)
    prev_month_date = current_date - pd.DateOffset(months=1)
    current_data = analysis_df_raw[(analysis_df_raw['연도'] == mom_year) & (analysis_df_raw['월'] == mom_month)]
    prev_data = analysis_df_raw[(analysis_df_raw['연도'] == prev_month_date.year) & (analysis_df_raw['월'] == prev_month_date.month)]
    current_agg = current_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    prev_agg = prev_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    mom_df = pd.DataFrame(current_agg).rename(columns={PRIMARY_WEIGHT_COL: '기준월_중량'})
    mom_df = mom_df.join(prev_agg.rename('이전월_중량'), how='outer').fillna(0)
    mom_df['증감량'] = mom_df['기준월_중량'] - mom_df['이전월_중량']
    mom_df['증감률'] = mom_df['증감량'] / mom_df['이전월_중량'].replace(0, np.nan)
    display_comparison_table(mom_df, '월')


    # --- 2. 전년 동월 대비 분석 ---
    title_col_yoy, year_col_yoy, month_col_yoy = st.columns([0.5, 0.25, 0.25])
    with title_col_yoy:
        st.subheader("🆚 전년 동월 대비")
    with year_col_yoy:
        yoy_year = st.selectbox("연도", available_years, key="yoy_year",
                                index=available_years.index(latest_date.year))
    with month_col_yoy:
        yoy_month = st.selectbox("월", available_months, key="yoy_month",
                                 index=available_months.index(latest_date.month))

    current_data_yoy = analysis_df_raw[(analysis_df_raw['연도'] == yoy_year) & (analysis_df_raw['월'] == yoy_month)]
    prev_year_data = analysis_df_raw[(analysis_df_raw['연도'] == yoy_year - 1) & (analysis_df_raw['월'] == yoy_month)]
    current_agg_yoy = current_data_yoy.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    prev_year_agg = prev_year_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    yoy_df = pd.DataFrame(current_agg_yoy).rename(columns={PRIMARY_WEIGHT_COL: '기준월_중량'})
    yoy_df = yoy_df.join(prev_year_agg.rename('이전월_중량'), how='outer').fillna(0)
    yoy_df['증감량'] = yoy_df['기준월_중량'] - yoy_df['이전월_중량']
    yoy_df['증감률'] = yoy_df['증감량'] / yoy_df['이전월_중량'].replace(0, np.nan)
    display_comparison_table(yoy_df, '월')


    # --- 3. 분기별 비교 분석 ---
    title_col_q, year_col_q, quarter_col_q = st.columns([0.5, 0.25, 0.25])
    default_quarter = (latest_date.month - 1) // 3 + 1
    with title_col_q:
        st.subheader("🆚 전년 동분기 대비")
    with year_col_q:
        q_year = st.selectbox("연도", available_years, key="q_year",
                              index=available_years.index(latest_date.year))
    with quarter_col_q:
        q_quarter = st.selectbox("분기", [1, 2, 3, 4], key="q_quarter",
                                 index=int(default_quarter - 1))

    current_q_data = analysis_df_raw[(analysis_df_raw['연도'] == q_year) & (analysis_df_raw['분기'] == q_quarter)]
    prev_q_data = analysis_df_raw[(analysis_df_raw['연도'] == q_year - 1) & (analysis_df_raw['분기'] == q_quarter)]
    current_q_agg = current_q_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    prev_q_agg = prev_q_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    qoq_df = pd.DataFrame(current_q_agg).rename(columns={PRIMARY_WEIGHT_COL: '기준분기_중량'})
    qoq_df = qoq_df.join(prev_q_agg.rename('이전분기_중량'), how='outer').fillna(0)
    qoq_df['증감량'] = qoq_df['기준분기_중량'] - qoq_df['이전분기_중량']
    qoq_df['증감률'] = qoq_df['증감량'] / qoq_df['이전분기_중량'].replace(0, np.nan)
    display_comparison_table(qoq_df, '분기')


    # --- 4. 반기별 비교 분석 ---
    title_col_h, year_col_h, half_col_h = st.columns([0.5, 0.25, 0.25])
    default_half = (latest_date.month - 1) // 6 + 1
    half_display = lambda x: f"{'상반기' if x == 1 else '하반기'}"
    with title_col_h:
        st.subheader("🆚 전년 동반기 대비")
    with year_col_h:
        h_year = st.selectbox("연도", available_years, key="h_year",
                              index=available_years.index(latest_date.year))
    with half_col_h:
        h_half = st.selectbox("반기", [1, 2], key="h_half",
                              index=int(default_half - 1), format_func=half_display)

    current_h_data = analysis_df_raw[(analysis_df_raw['연도'] == h_year) & (analysis_df_raw['반기'] == h_half)]
    prev_h_data = analysis_df_raw[(analysis_df_raw['연도'] == h_year - 1) & (analysis_df_raw['반기'] == h_half)]
    current_h_agg = current_h_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    prev_h_agg = prev_h_data.groupby('대표품목별')[PRIMARY_WEIGHT_COL].sum()
    hoh_df = pd.DataFrame(current_h_agg).rename(columns={PRIMARY_WEIGHT_COL: '기준반기_중량'})
    hoh_df = hoh_df.join(prev_h_agg.rename('이전반기_중량'), how='outer').fillna(0)
    hoh_df['증감량'] = hoh_df['기준반기_중량'] - hoh_df['이전반기_중량']
    hoh_df['증감률'] = hoh_df['증감량'] / hoh_df['이전반기_중량'].replace(0, np.nan)
    display_comparison_table(hoh_df, '반기')


# ----------------------------------------------------------------
# ★★★ 이하 코드는 변경 없음 ★★★
# ----------------------------------------------------------------
elif menu == "기간별 수입량 분석":
    st.title(f"📆 기간별 수입량 추이 분석 (기준: {PRIMARY_WEIGHT_COL})")
    st.markdown("---")
    analysis_df = df.dropna(subset=['날짜', PRIMARY_WEIGHT_COL, '연도', '월', '분기', '반기'])
    if analysis_df.empty:
        st.warning("분석할 유효한 데이터가 없습니다.")
        st.stop()
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        period_type = st.radio("분석 기간 단위", ('월별', '분기별', '반기별'))
    all_items = sorted(analysis_df['대표품목별'].unique())
    with col2:
        if 'selected_items_memory' not in st.session_state:
            st.session_state.selected_items_memory = []
        selected_items = st.multiselect(
            "품목 선택 (최대 5개)",
            options=all_items,
            placeholder="수입량 추이를 확인할 품목을 선택해주세요",
            default=st.session_state.selected_items_memory,
            max_selections=5
        )
        st.session_state.selected_items_memory = selected_items
    if selected_items:
        filtered_df = analysis_df[analysis_df['대표품목별'].isin(selected_items)]
        agg_cols, title = [], ""
        if period_type == '월별':
            agg_cols, title = ['연도', '월'], "월별 수입량 추이"
        elif period_type == '분기별':
            agg_cols, title = ['연도', '분기'], "분기별 수입량 추이"
        elif period_type == '반기별':
            agg_cols, title = ['연도', '반기'], "반기별 수입량 추이"
        agg_df = filtered_df.groupby(agg_cols + ['대표품목별'])[PRIMARY_WEIGHT_COL].sum().unstack().fillna(0)
        if period_type == '월별':
            agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{int(x[1]):02d}")
        elif period_type == '분기별':
            agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{int(x[1])}분기")
        elif period_type == '반기별':
            agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{'상반기' if x[1] == 1 else '하반기'}")
        st.header(f"📈 {title}")
        chart_type = st.radio("차트 종류", ('선 그래프', '막대 그래프'), horizontal=True, key="chart_type_trends")
        if chart_type == '선 그래프':
            st.line_chart(agg_df)
        else:
            st.bar_chart(agg_df)
        with st.expander("데이터 상세 보기"):
            st.subheader("기간별 수입량 (KG)")
            st.dataframe(agg_df.style.format("{:,.0f}"))
            st.subheader("이전 기간 대비 증감률 (%)")
            growth_rate_df = agg_df.pct_change()
            st.dataframe(growth_rate_df.style.format("{:+.2%}", na_rep="-"))
    else:
        st.info("그래프를 보려면 먼저 품목을 선택해주세요.")

elif menu == "데이터 추가":
    st.title("📤 데이터 추가")
    st.info(f"다음 컬럼을 포함한 엑셀/CSV 파일을 업로드해주세요:\n`{', '.join(DESIRED_HEADER)}`")
    uploaded_file = st.file_uploader("파일 선택", type=['xlsx', 'csv'])
    password = st.text_input("업로드 비밀번호", type="password")
    if st.button("데이터베이스에 추가"):
        if uploaded_file and password == "1004":
            try:
                st.info("파일을 읽고 처리하는 중입니다...")
                if uploaded_file.name.endswith('.csv'):
                    new_df = pd.read_csv(uploaded_file, dtype=str)
                else:
                    new_df = pd.read_excel(uploaded_file, dtype=str)
                desired_set = set(DESIRED_HEADER)
                new_df_set = set(new_df.columns)
                if desired_set != new_df_set:
                    missing = desired_set - new_df_set
                    extra = new_df_set - desired_set
                    error_message = "🚨 업로드한 파일의 컬럼 구성이 올바르지 않습니다.\n"
                    if missing: error_message += f"\n**- 누락된 컬럼:** `{', '.join(missing)}`"
                    if extra: error_message += f"\n**- 불필요한 컬럼:** `{', '.join(extra)}`"
                    st.error(error_message)
                    st.stop()
                new_df_processed = preprocess_dataframe(new_df)
                client = get_google_sheet_client()
                if client:
                    sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
                    unique_periods = new_df_processed.dropna(subset=['연도', '월'])[['연도', '월']].drop_duplicates()
                    df_filtered = df.copy()
                    if not df_filtered.empty and not unique_periods.empty:
                        for _, row in unique_periods.iterrows():
                            year_val = row['연도']
                            month_val = row['월']
                            df_filtered = df_filtered[~((df_filtered['연도'] == year_val) & (df_filtered['월'] == month_val))]
                    combined_df = pd.concat([df_filtered, new_df_processed], ignore_index=True)
                    combined_df.sort_values(by=['Year', 'Month', 'NO'], inplace=True, na_position='last')
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
