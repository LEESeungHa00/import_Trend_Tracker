import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import time
import altair as alt
import plotly
import plotly.express as px # [수정] Plotly 라이브러리 추가

# ---------------------------------
# 페이지 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="수입 실적 분석 대시보드",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------
# 상수 정의
# ---------------------------------
# 분석 기준 컬럼 정의
PRIMARY_WEIGHT_COL = '적합 중량(KG)'
PRIMARY_AMOUNT_COL = '적합 금액($)'

# 구글 시트에서 불러올 헤더 정의
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
# 데이터 로딩 및 전처리
# ---------------------------------
def preprocess_dataframe(df):
    """데이터프레임의 숫자 컬럼을 정리하고 날짜 관련 파생 변수를 생성합니다."""
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

@st.cache_data(ttl=3600)
def load_data():
    """구글 시트에서 데이터를 로드하고, 실패 시 샘플 데이터를 생성합니다."""
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
    """분석용 샘플 데이터를 생성합니다."""
    items = ['소고기(냉장)', '바지락(활)', '김치', '과자', '맥주', '새우(냉동)', '오렌지', '바나나', '커피원두', '치즈']
    categories = {
        '소고기(냉장)': '축산물', '바지락(활)': '수산물', '김치': '가공식품', 
        '과자': '가공식품', '맥주': '가공식품', '새우(냉동)': '수산물', 
        '오렌지': '농산물', '바나나': '농산물', '커피원두': '농산물', '치즈': '축산물'
    }
    daterange = pd.date_range(start='2021-01-01', end='2025-07-31', freq='M')
    data = []
    no_counter = 1
    for date in daterange:
        for item in items:
            weight = (10000 + items.index(item) * 5000) * np.random.uniform(0.8, 1.2)
            price = weight * np.random.uniform(5, 10)
            data.append([
                no_counter, date.year, date.month, categories[item], '미국', '미국', '판매용',
                item, weight, price, weight*0.95, price*0.95, weight*0.05, price*0.05
            ])
            no_counter += 1
    df = pd.DataFrame(data, columns=DESIRED_HEADER)
    df = preprocess_dataframe(df)
    return df

def update_sheet_in_batches(worksheet, dataframe, batch_size=10000):
    """데이터프레임을 작은 배치로 나누어 구글 시트에 업로드합니다."""
    worksheet.clear()
    worksheet.append_row(dataframe.columns.values.tolist())
    
    data = dataframe.fillna('').values.tolist()
    total_rows = len(data)
    
    if total_rows == 0:
        st.success("✅ 업로드 완료! (업로드할 데이터 없음)")
        return

    progress_bar = st.progress(0, text="데이터 업로드를 시작합니다...")
    
    for i in range(0, total_rows, batch_size):
        batch = data[i:i+batch_size]
        worksheet.append_rows(batch, value_input_option='USER_ENTERED')
        
        progress_percentage = min((i + batch_size) / total_rows, 1.0)
        progress_text = f"{min(i + batch_size, total_rows)} / {total_rows} 행 업로드 중..."
        progress_bar.progress(progress_percentage, text=progress_text)
        
        time.sleep(1)
        
    progress_bar.progress(1.0, text="✅ 업로드 완료!")

# ---------------------------------
# ---- 메인 애플리케이션 로직 ----
# ---------------------------------

# 1. 사이드바 메뉴 및 분석 모드 선택
st.sidebar.title("메뉴")
analysis_mode = st.sidebar.radio(
    "분석 기준",
    ('중량 모드', '금액 모드'),
    horizontal=True
)
menu = st.sidebar.radio(
    "원하는 기능을 선택하세요.",
    ("수입 현황 대시보드", "시계열 추세 분석", "기간별 추이 분석", "데이터 추가")
)
if st.sidebar.button("🔄 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

# 2. 선택된 모드에 따라 동적 변수 설정
if analysis_mode == '중량 모드':
    primary_col = PRIMARY_WEIGHT_COL
    unit = '(KG)'
    value_name = '수입량'
    change_name = '증감량'
    format_str = '{:,.0f}'
    axis_format = '~s' # Altair 전용 포맷
    label_expr = """
    datum.value == 0 ? '0' : 
    (abs(datum.value) >= 1000000 ? format(abs(datum.value) / 1000000, ',.0f') + 'M' : 
    (abs(datum.value) >= 1000 ? format(abs(datum.value) / 1000, ',.0f') + 'K' : format(abs(datum.value), ',.0f')))
    """
else: # 금액 모드
    primary_col = PRIMARY_AMOUNT_COL
    unit = '($)'
    value_name = '수입액'
    change_name = '증감액'
    format_str = '${:,.0f}'
    axis_format = '$,.0s' # Altair 전용 포맷
    label_expr = """
    datum.value == 0 ? '$0' : 
    (abs(datum.value) >= 1000000 ? '$' + format(abs(datum.value) / 1000000, ',.0f') + 'M' : 
    (abs(datum.value) >= 1000 ? '$' + format(abs(datum.value) / 1000, ',.0f') + 'K' : '$' + format(abs(datum.value), ',.0f')))
    """

df = load_data()

if df.empty and menu != "데이터 추가":
    st.warning("데이터가 없습니다. '데이터 추가' 탭으로 이동하여 데이터를 업로드해주세요.")
    st.stop()

# ----- Altair 줌/팬 동작 정의 (다른 탭에서 사용) -----
zoom_on_drag = alt.selection_interval(
    bind='scales',
    on="[mousedown[!event.shiftKey], mouseup] > mousemove",
    empty='all'
)
pan_on_shift_drag = alt.selection_interval(
    bind='scales',
    on="[mousedown[event.shiftKey], mouseup] > mousemove",
    empty='all'
)
# -----------------------------------


# --- 대시보드 페이지 ---
if menu == "수입 현황 대시보드":
    st.title(f"📊 수입 현황 대시보드")
    st.info(f"(기준: {primary_col})")

    analysis_df_raw = df.dropna(subset=['날짜', primary_col, '연도', '분기', '반기'])
    if analysis_df_raw.empty:
        st.warning("분석할 유효한 데이터가 없습니다. 'Year', 'Month' 데이터가 올바른지 확인해주세요.")
        st.stop()
    
    available_years = sorted(analysis_df_raw['연도'].unique().astype(int), reverse=True)
    available_months = sorted(analysis_df_raw['월'].unique().astype(int))
    latest_date = analysis_df_raw['날짜'].max()

    def create_butterfly_chart_altair(df_agg, base_col, prev_col, base_label, prev_label):
        """증감 상위/하위 품목에 대한 나비 차트를 생성합니다."""
        change_col = f'{change_name}{unit}'
        top_items = df_agg.nlargest(5, change_col)
        bottom_items = df_agg.nsmallest(5, change_col)
        chart_data = pd.concat([top_items, bottom_items])
        
        if chart_data.empty:
            st.info("비교할 증감 내역이 있는 품목이 없습니다.")
            return

        chart_data = chart_data.reset_index()
        df_melted = chart_data.melt(
            id_vars='대표품목별', value_vars=[prev_col, base_col],
            var_name='시점_컬럼명', value_name=f'{value_name}{unit}'
        )
        df_melted['차트_값'] = df_melted.apply(
            lambda row: -row[f'{value_name}{unit}'] if row['시점_컬럼명'] == prev_col else row[f'{value_name}{unit}'],
            axis=1
        )
        df_melted['시점'] = df_melted['시점_컬럼명'].map({prev_col: prev_label, base_col: base_label})
        sort_order = chart_data.sort_values(change_col, ascending=False)['대표품목별'].tolist()
        
        final_chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X('차트_값:Q', title=f'{value_name} {unit}', axis=alt.Axis(labelExpr=label_expr)),
            y=alt.Y('대표품목별:N', sort=sort_order, title=None),
            color=alt.Color('시점:N',
                scale=alt.Scale(domain=[prev_label, base_label], range=['#5f8ad6', '#d65f5f']),
                legend=alt.Legend(title="시점 구분", orient='top')
            ),
            tooltip=[
                alt.Tooltip('대표품목별', title='품목'),
                alt.Tooltip('시점', title='기간'),
                alt.Tooltip(f'{value_name}{unit}', title=value_name, format=',.0f')
            ]
        ).properties(
            title=alt.TitleParams(text=f'{prev_label} vs {base_label} {value_name} 비교', anchor='middle')
        ).add_params( # Altair 줌/팬 적용
            zoom_on_drag,
            pan_on_shift_drag
        )
        
        st.altair_chart(final_chart, use_container_width=True)

    def display_comparison_tab(title, current_data, prev_data, base_label, prev_label):
        """비교 분석 탭의 UI와 로직을 표시하는 함수."""
        st.subheader(f"🆚 {title}")
        current_agg = current_data.groupby('대표품목별')[primary_col].sum()
        prev_agg = prev_data.groupby('대표품목별')[primary_col].sum()

        base_col_name = f'기준_{value_name}{unit}'
        prev_col_name = f'이전_{value_name}{unit}'
        change_col_name = f'{change_name}{unit}'
        rate_col_name = '증감률'

        df_agg = pd.DataFrame(current_agg).rename(columns={primary_col: base_col_name})
        df_agg = df_agg.join(prev_agg.rename(prev_col_name), how='outer').fillna(0)
        df_agg[change_col_name] = df_agg[base_col_name] - df_agg[prev_col_name]
        df_agg[rate_col_name] = df_agg[change_col_name] / df_agg[prev_col_name].replace(0, np.nan)
        
        with st.expander("📊 Before & After (증감 상위/하위 5개 품목)"):
            create_butterfly_chart_altair(df_agg, base_col_name, prev_col_name, base_label, prev_label)
        
        formatter = {
            base_col_name: format_str,
            prev_col_name: format_str,
            change_col_name: f'{{:+,.0f}}' if analysis_mode == '중량 모드' else f'${{:+,.0f}}',
            rate_col_name: '{:+.2%}'
        }
        
        st.markdown(f'<p style="color:red; font-weight:bold;">🔼 {value_name} 증가 TOP 5 ({change_name} 많은 순)</p>', unsafe_allow_html=True)
        st.dataframe(df_agg.nlargest(5, change_col_name).reset_index().style.format(formatter, na_rep="-"), hide_index=True, use_container_width=True)
        
        st.markdown(f'<p style="color:blue; font-weight:bold;">🔽 {value_name} 감소 TOP 5 ({change_name} 많은 순)</p>', unsafe_allow_html=True)
        st.dataframe(df_agg.nsmallest(5, change_col_name).reset_index().style.format(formatter, na_rep="-"), hide_index=True, use_container_width=True)

        st.markdown(f'<p style="color:green; font-weight:bold;">❇️ 신규 수입 품목 TOP 10 (이전 기간 0)</p>', unsafe_allow_html=True)
        
        new_items_df = df_agg[
            (df_agg[base_col_name] > 0) & (df_agg[prev_col_name] == 0)
        ]
        
        if new_items_df.empty:
            st.info("해당 기간에 신규로 수입된 품목이 없습니다.")
        else:
            new_items_top10 = new_items_df.sort_values(
                by=base_col_name, ascending=False
            ).head(10).reset_index()
            
            final_new_items_df = new_items_top10.rename(
                columns={'대표품목별': '품목명'}
            )[['품목명', base_col_name, prev_col_name]]
            
            st.dataframe(
                final_new_items_df.style.format(formatter, na_rep="-"), 
                hide_index=True,
                use_container_width=True
            )

    tab_yy, tab_mom, tab_yoy, tab_qoq, tab_hoh = st.tabs([
        "전년 대비", "전월 대비", "전년 동월 대비", "전년 동분기 대비", "전년 동반기 대비"
    ])

    with tab_yy:
        yy_year = st.selectbox("기준 연도", available_years, key="yy_year", index=0)
        current_yy_data = analysis_df_raw[analysis_df_raw['연도'] == yy_year]
        prev_yy_data = analysis_df_raw[analysis_df_raw['연도'] == yy_year - 1]
        display_comparison_tab(f"전년 대비 {value_name} 분석", current_yy_data, prev_yy_data, f'{yy_year}년', f'{yy_year-1}년')

    with tab_mom:
        mom_col1, mom_col2 = st.columns(2)
        with mom_col1:
            mom_year = st.selectbox("기준 연도", available_years, key="mom_year", index=0)
        with mom_col2:
            mom_month = st.selectbox("기준 월", available_months, key="mom_month", index=available_months.index(latest_date.month))
        current_date = datetime(mom_year, mom_month, 1)
        prev_month_date = current_date - pd.DateOffset(months=1)
        current_data = analysis_df_raw[(analysis_df_raw['연도'] == mom_year) & (analysis_df_raw['월'] == mom_month)]
        prev_data = analysis_df_raw[(analysis_df_raw['연도'] == prev_month_date.year) & (analysis_df_raw['월'] == prev_month_date.month)]
        display_comparison_tab(f"전월 대비 {value_name} 분석", current_data, prev_data, f'{mom_year}년 {mom_month}월', f'{prev_month_date.year}년 {prev_month_date.month}월')

    with tab_yoy:
        yoy_col1, yoy_col2 = st.columns(2)
        with yoy_col1:
            yoy_year = st.selectbox("기준 연도", available_years, key="yoy_year", index=0)
        with yoy_col2:
            yoy_month = st.selectbox("기준 월", available_months, key="yoy_month", index=available_months.index(latest_date.month))
        current_data_yoy = analysis_df_raw[(analysis_df_raw['연도'] == yoy_year) & (analysis_df_raw['월'] == yoy_month)]
        prev_year_data = analysis_df_raw[(analysis_df_raw['연도'] == yoy_year - 1) & (analysis_df_raw['월'] == yoy_month)]
        display_comparison_tab(f"전년 동월 대비 {value_name} 분석", current_data_yoy, prev_year_data, f'{yoy_year}년 {yoy_month}월', f'{yoy_year-1}년 {yoy_month}월')

    with tab_qoq:
        q_col1, q_col2 = st.columns(2)
        default_quarter = (latest_date.month - 1) // 3 + 1
        with q_col1:
            q_year = st.selectbox("기준 연도", available_years, key="q_year", index=0)
        with q_col2:
            q_quarter = st.selectbox("기준 분기", [1, 2, 3, 4], key="q_quarter", index=int(default_quarter - 1))
        current_q_data = analysis_df_raw[(analysis_df_raw['연도'] == q_year) & (analysis_df_raw['분기'] == q_quarter)]
        prev_q_data = analysis_df_raw[(analysis_df_raw['연도'] == q_year - 1) & (analysis_df_raw['분기'] == q_quarter)]
        display_comparison_tab(f"전년 동분기 대비 {value_name} 분석", current_q_data, prev_q_data, f'{q_year}년 {q_quarter}분기', f'{q_year-1}년 {q_quarter}분기')

    with tab_hoh:
        h_col1, h_col2 = st.columns(2)
        default_half = (latest_date.month - 1) // 6 + 1
        half_display = lambda x: f"{'상반기' if x == 1 else '하반기'}"
        with h_col1:
            h_year = st.selectbox("기준 연도", available_years, key="h_year", index=0)
        with h_col2:
            h_half = st.selectbox("기준 반기", [1, 2], key="h_half", index=int(default_half - 1), format_func=half_display)
        current_h_data = analysis_df_raw[(analysis_df_raw['연도'] == h_year) & (analysis_df_raw['반기'] == h_half)]
        prev_h_data = analysis_df_raw[(analysis_df_raw['연도'] == h_year - 1) & (analysis_df_raw['반기'] == h_half)]
        display_comparison_tab(f"전년 동반기 대비 {value_name} 분석", current_h_data, prev_h_data, f'{h_year}년 {half_display(h_half)}', f'{h_year-1}년 {half_display(h_half)}')

# --- 시계열 추세 분석 페이지 ---
elif menu == "시계열 추세 분석":
    st.title(f"📈 시계열 추세 분석 (기준: {primary_col})")
    st.info("선택한 기간 동안 꾸준한 증가 또는 감소 추세를 보이는 품목을 식별합니다.")
    
    trend_df = df.dropna(subset=['날짜', primary_col, '연도', '월'])
    if trend_df.empty:
        st.warning("분석할 유효한 데이터가 없습니다.")
        st.stop()

    # --- 연도별 - 장기 추세 분석 ---
    st.markdown("---")
    st.subheader("연도별 - 장기 추세 분석")

    yearly_agg = trend_df.groupby(['연도', '대표품목별'])[primary_col].sum().reset_index()
    available_years_trend = sorted(yearly_agg['연도'].unique().astype(int))

    if len(available_years_trend) >= 2:
        start_y, end_y = st.select_slider(
            '분석 기간 (년)',
            options=available_years_trend,
            value=(available_years_trend[0], available_years_trend[-1]),
            key='yearly_slider'
        )
        duration_years = end_y - start_y + 1
        st.caption(f"선택된 기간 : **{duration_years}년** ({start_y}년 ~ {end_y}년)")
        
        trend_type_years = st.radio("추세 선택", ("지속 증가 📈", "지속 감소 📉"), horizontal=True, key="trend_type_years")

        period_df_yearly = yearly_agg[(yearly_agg['연도'] >= start_y) & (yearly_agg['연도'] <= end_y)]
        results_yearly = []
        for item, group in period_df_yearly.groupby('대표품목별'):
            if len(group['연도'].unique()) == duration_years:
                group = group.sort_values('연도')
                diffs = group[primary_col].diff().dropna()
                if (trend_type_years == "지속 증가 📈" and (diffs > 0).all()) or \
                   (trend_type_years == "지속 감소 📉" and (diffs < 0).all()):
                    
                    start_val = group.iloc[0][primary_col]
                    end_val = group.iloc[-1][primary_col]
                    growth_rate = (end_val - start_val) / start_val if start_val > 0 else (np.inf if end_val > 0 else 0)
                    results_yearly.append({
                        '대표품목별': item,
                        f'{start_y}년_{value_name}{unit}': start_val, f'{end_y}년_{value_name}{unit}': end_val,
                        '기간내_증감률': growth_rate
                    })
        
        if results_yearly:
            result_df_yearly = pd.DataFrame(results_yearly)
            sort_col = '기간내_증감률'
            result_df_yearly = result_df_yearly.nlargest(10, sort_col) if trend_type_years == "지속 증가 📈" else result_df_yearly.nsmallest(10, sort_col)
            
            st.markdown(f"**선택 기간 동안 `{trend_type_years}` 품목 TOP 10**")
            st.dataframe(result_df_yearly.style.format({
                f'{start_y}년_{value_name}{unit}': format_str, f'{end_y}년_{value_name}{unit}': format_str,
                '기간내_증감률': '{:+.2%}'
            }, na_rep="-"), hide_index=True)

            if not result_df_yearly.empty:
                st.markdown("---")
                st.subheader("개별 품목 연도별 추이 그래프")
                selected_item_y = st.selectbox("그래프로 확인할 품목을 선택하세요", options=result_df_yearly['대표품목별'], key="selected_item_y")
                if selected_item_y:
                    item_trend_df_y = period_df_yearly[period_df_yearly['대표품목별'] == selected_item_y]
                    chart_y = alt.Chart(item_trend_df_y).mark_line(point=True).encode(
                        x=alt.X('연도:O', title='연도'),
                        y=alt.Y(f'{primary_col}:Q', title=f'{value_name} {unit}', axis=alt.Axis(format=axis_format)),
                        tooltip=['연도', alt.Tooltip(f'{primary_col}', title=value_name, format=',.0f')]
                    ).properties(title=f"'{selected_item_y}'의 {start_y}년 ~ {end_y}년 {value_name} 추이"
                    ).add_params( # Altair 줌/팬 적용
                        zoom_on_drag,
                        pan_on_shift_drag
                    )
                    st.altair_chart(chart_y, use_container_width=True)
    else:
        st.warning("연도별 추세를 분석하려면 최소 2년 이상의 데이터가 필요합니다.")

    # --- 월별 - 단기 추세 분석 ---
    st.markdown("---")
    st.subheader("월별 - 단기 추세 분석")
    monthly_periods = sorted(trend_df['날짜'].dt.to_period('M').unique().astype(str))
    if len(monthly_periods) >= 3:
        start_m, end_m = st.select_slider(
            '분석 기간 (월)',
            options=monthly_periods,
            value=(monthly_periods[0], monthly_periods[-1]),
            key='monthly_slider'
        )
        start_date = pd.to_datetime(start_m).to_pydatetime()
        end_date = pd.to_datetime(end_m).to_pydatetime()
        duration_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        st.caption(f"선택된 기간: **{duration_months}개월** ({start_m} ~ {end_m})")
        
        trend_type_months = st.radio("추세 선택", ("지속 증가 📈", "지속 감소 📉"), horizontal=True, key="trend_type_months")
        
        period_df_monthly = trend_df[(trend_df['날짜'] >= start_date) & (trend_df['날짜'] <= end_date)]
        results_monthly = []
        for item, group in period_df_monthly.groupby('대표품목별'):
            if len(group['날짜'].dt.to_period('M').unique()) == duration_months:
                monthly_agg = group.groupby(pd.Grouper(key='날짜', freq='M'))[primary_col].sum()
                diffs = monthly_agg.diff().dropna()
                if (trend_type_months == "지속 증가 📈" and (diffs > 0).all()) or \
                   (trend_type_months == "지속 감소 📉" and (diffs < 0).all()):
                    start_val = monthly_agg.iloc[0]
                    end_val = monthly_agg.iloc[-1]
                    growth_rate = (end_val - start_val) / start_val if start_val > 0 else (np.inf if end_val > 0 else 0)
                    results_monthly.append({
                        '대표품목별': item,
                        f'시작월_{value_name}{unit}': start_val, f'종료월_{value_name}{unit}': end_val,
                        '기간내_증감률': growth_rate
                    })
        
        if results_monthly:
            result_df_monthly = pd.DataFrame(results_monthly)
            sort_col = '기간내_증감률'
            result_df_monthly = result_df_monthly.nlargest(10, sort_col) if trend_type_months == "지속 증가 📈" else result_df_monthly.nsmallest(10, sort_col)
            
            st.markdown(f"**선택 기간 동안 `{trend_type_months}` 품목 TOP 10**")
            st.dataframe(result_df_monthly.style.format({
                f'시작월_{value_name}{unit}': format_str, f'종료월_{value_name}{unit}': format_str,
                '기간내_증감률': '{:+.2%}'
            }, na_rep="-"), hide_index=True)

            if not result_df_monthly.empty:
                st.markdown("---")
                st.subheader("개별 품목 월별 추이 그래프")
                selected_item_m = st.selectbox("그래프로 확인할 품목을 선택하세요", options=result_df_monthly['대표품목별'], key="selected_item_m")
                if selected_item_m:
                    item_trend_df_m = period_df_monthly[period_df_monthly['대표품목별'] == selected_item_m]
                    monthly_item_agg = item_trend_df_m.groupby(pd.Grouper(key='날짜', freq='M'))[primary_col].sum().reset_index()
                    monthly_item_agg['기간'] = monthly_item_agg['날짜'].dt.strftime('%Y-%m')
                    chart_m = alt.Chart(monthly_item_agg).mark_line(point=True).encode(
                        x=alt.X('기간:N', sort=None, title='월'),
                        y=alt.Y(f'{primary_col}:Q', title=f'{value_name} {unit}', axis=alt.Axis(format=axis_format)),
                        tooltip=['기간', alt.Tooltip(f'{primary_col}', title=value_name, format=',.0f')]
                    ).properties(title=f"'{selected_item_m}'의 {start_m} ~ {end_m} {value_name} 추이"
                    ).add_params( # Altair 줌/팬 적용
                        zoom_on_drag,
                        pan_on_shift_drag
                    )
                    st.altair_chart(chart_m, use_container_width=True)
    else:
        st.warning("월별 추세를 분석하려면 최소 3개월 이상의 데이터가 필요합니다.")

# --- [수정] 기간별 추이 분석 페이지 (Plotly로 교체) ---
elif menu == "기간별 추이 분석":
    st.title(f"📆 기간별 {value_name} 추이 분석 (기준: {primary_col})")
    st.markdown("---")
    analysis_df = df.dropna(subset=['날짜', primary_col, '연도', '월', '분기', '반기', '제품구분별', '대표품목별'])
    if analysis_df.empty:
        st.warning("분석할 유효한 데이터가 없습니다.")
        st.stop()
    
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        period_type = st.radio("분석 기간 단위", ('월별', '분기별', '반기별'))
    
    all_categories = sorted(analysis_df['제품구분별'].unique())
    all_items = sorted(analysis_df['대표품목별'].unique())

    with col2:
        st.markdown("##### 1. 제품구분별 선택 (최대 5개)")
        st.info("기본적으로 '카테고리' 그래프가 그려집니다. 2번에서 품목 선택 시 '필터'로 동작합니다.")
        selected_categories = st.multiselect(
            "제품구분별 선택",
            options=all_categories,
            placeholder="카테고리를 선택하세요 (최대 5개)",
            label_visibility="collapsed",
            max_selections=5,
            key='cat_select'
        )
        
        if selected_categories:
            filtered_items_df = analysis_df[analysis_df['제품구분별'].isin(selected_categories)]
            available_items = sorted(filtered_items_df['대표품목별'].unique())
            item_placeholder = "선택한 카테고리 내 개별 품목 (최대 5개)"
        else:
            available_items = all_items
            item_placeholder = "전체 개별 품목 (최대 5개)"

        st.markdown("##### 2. 대표품목별 선택 (최대 5개)")
        st.info("여기에 품목을 선택하면, 그래프는 '품목' 기준으로 그려집니다.")
        selected_items = st.multiselect(
            "대표품목별 선택",
            options=available_items,
            placeholder=f"{item_placeholder}",
            label_visibility="collapsed",
            max_selections=5,
            key='item_select'
        )

    agg_df = pd.DataFrame()
    
    if selected_items:
        graph_title = "대표품목별 추이"
        agg_by_col = '대표품목별'
        filtered_df = analysis_df[analysis_df['대표품목별'].isin(selected_items)]
        if selected_categories:
             filtered_df = filtered_df[filtered_df['제품구분별'].isin(selected_categories)]
    
    elif selected_categories:
        graph_title = "제품구분별 추이"
        agg_by_col = '제품구분별'
        filtered_df = analysis_df[analysis_df['제품구분별'].isin(selected_categories)]
    
    else:
        st.info("그래프를 보려면 '제품구분별' 또는 '대표품목별'을 선택해주세요.")
        filtered_df = pd.DataFrame()
        agg_by_col = None

    if not filtered_df.empty and agg_by_col:
        agg_cols, title_suffix = [], ""
        if period_type == '월별':
            agg_cols, title_suffix = ['연도', '월'], f"월별 {value_name} 추이"
        elif period_type == '분기별':
            agg_cols, title_suffix = ['연도', '분기'], f"분기별 {value_name} 추이"
        elif period_type == '반기별':
            agg_cols, title_suffix = ['연도', '반기'], f"반기별 {value_name} 추이"
        
        agg_df = filtered_df.groupby(agg_cols + [agg_by_col])[primary_col].sum().unstack(fill_value=0)
        
        if agg_df.empty:
            st.info("선택한 항목에 대한 데이터가 없습니다.")
        else:
            if period_type == '월별':
                agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{int(x[1]):02d}")
            elif period_type == '분기별':
                agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{int(x[1])}분기")
            elif period_type == '반기별':
                agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{'상반기' if x[1] == 1 else '하반기'}")
            
            st.header(f"📈 {graph_title} - {title_suffix}")
            
            df_melted = agg_df.reset_index().melt(id_vars='index', var_name=agg_by_col, value_name=f'{value_name}{unit}')
            df_melted.rename(columns={'index': '기간'}, inplace=True)
            
            chart_type = st.radio("차트 종류", ('선 그래프', '막대 그래프'), horizontal=True, key="chart_type_trends")
            
            # --- [수정] Altair -> Plotly로 교체 ---
            fig = None
            if chart_type == '선 그래프':
                fig = px.line(
                    df_melted, 
                    x='기간', 
                    y=f'{value_name}{unit}', 
                    color=agg_by_col,
                    markers=True, # 라인에 마커(점) 표시
                    labels={f'{value_name}{unit}': f'{value_name} {unit}', '기간': '기간', agg_by_col: '선택 항목'} # 범례 제목
                )
            else: # 막대 그래프
                fig = px.bar(
                    df_melted, 
                    x='기간', 
                    y=f'{value_name}{unit}', 
                    color=agg_by_col,
                    labels={f'{value_name}{unit}': f'{value_name} {unit}', '기간': '기간', agg_by_col: '선택 항목'}
                )
            
            # Plotly 차트 레이아웃 업데이트 (x축 정렬 유지)
            fig.update_layout(
                xaxis={'categoryorder':'total descending'} if (period_type == '월별' and len(agg_df) > 12) else {'categoryorder':'trace'},
                hovermode="x unified" # 마우스 오버 시 x축 기준 모든 데이터 표시
            )
            
            st.plotly_chart(fig, use_container_width=True)
            # --- 교체 완료 ---
                
            with st.expander("데이터 상세 보기"):
                st.subheader(f"기간별 {value_name} {unit}")
                st.dataframe(agg_df.style.format(format_str))
                st.subheader("이전 기간 대비 증감률 (%)")
                growth_rate_df = agg_df.pct_change()
                st.dataframe(growth_rate_df.style.format("{:+.2%}", na_rep="-"))
    elif not selected_categories and not selected_items:
        pass
    else:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")

# --- 데이터 추가 페이지 ---
elif menu == "데이터 추가":
    st.title("📤 데이터 추가")
    st.info(f"다음 컬럼을 포함한 엑셀/CSV 파일을 업로드해주세요:\n`{', '.join(DESIRED_HEADER)}`")
    uploaded_file = st.file_uploader("파일 선택", type=['xlsx', 'csv'])
    password = st.text_input("업로드 비밀번호", type="password")
    if st.button("데이터베이스에 추가"):
        if uploaded_file and password == "1004":
            try:
                st.info("파일을 읽고 처리하는 중입니다...")
                new_df = pd.read_csv(uploaded_file, dtype=str) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, dtype=str)
                
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
                        df_filtered['연도'] = pd.to_numeric(df_filtered['연도'], errors='coerce')
                        df_filtered['월'] = pd.to_numeric(df_filtered['월'], errors='coerce')
                        
                        merged = df_filtered.merge(unique_periods, on=['연도', '월'], how='left', indicator=True)
                        df_filtered = df_filtered[merged['_merge'] == 'left_only']

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
