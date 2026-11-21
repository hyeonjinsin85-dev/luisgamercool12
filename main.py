# main.py
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────
# 한글 폰트 (Streamlit CSS)
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# CSV 읽기 함수 (인코딩 대응)
# ──────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 시도, 실패 시 빈 DataFrame 반환"""
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ──────────────────────────────────────────────
# 열 이름 표준화 (KeyError 방지)
# ──────────────────────────────────────────────
def std_cols(df):
    """CSV 열 이름을 표준화된 이름으로 통일"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"

    df = df.rename(columns=rename)

    # 숫자 열 변환
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량이 없고 지상부+지하부 질량이 있으면 합산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # 시간 파싱
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    return df


def get_school_name(filename):
    """파일명에서 학교명 추출 (예: 송도고_환경데이터.csv → 송도고)"""
    if filename is None:
        return ""
    base = os.path.basename(filename)
    # 첫 번째 '_' 또는 '.' 이전까지를 학교명으로
    name = base.split("_")[0].split(".")[0]
    return name.strip()


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
st.sidebar.title("📂 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

# 생육 측정 항목 선택 (있는 열만 표시)
growth_cols_options = [
    "잎 수",
    "지상부 길이",
    "지하부 길이",
    "생중량",
    "지상부 질량",
    "지하부 질량",
]
selected_growth = st.sidebar.multiselect(
    "🌱 생육 측정 항목 선택",
    growth_cols_options,
    default=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
)

st.sidebar.markdown("---")
st.sidebar.info("CSV 파일을 올리면 자동으로 분석이 시작돼요.")

# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────
with st.spinner("데이터를 읽고 있어요..."):
    env_df = read_csv(env_file)
    growth_df = read_csv(growth_file)

    env_df_std = std_cols(env_df) if not env_df.empty else pd.DataFrame()
    growth_df_std = std_cols(growth_df) if not growth_df.empty else pd.DataFrame()

school_env = get_school_name(env_file.name if env_file else None)
school_growth = get_school_name(growth_file.name if growth_file else None)

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(page_title="생육 결과 분석", layout="wide")
st.title("🌱 생육 결과 분석")

# 학교명 안내
if school_env or school_growth:
    names = [n for n in [school_env, school_growth] if n]
    st.caption(f"📍 분석 대상: {', '.join(names)} (업로드한 파일 기준)")


# ──────────────────────────────────────────────
# 탭 구성
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌡️ 환경 데이터", "🌱 생육 데이터", "📊 종합 분석"])


# ══════════════════════════════════════════════
# 탭 1: 환경 데이터
# ══════════════════════════════════════════════
with tab1:
    try:
        if env_df_std.empty:
            st.info("📁 사이드바에서 환경 CSV를 올려주세요.")
        else:
            st.subheader(f"🌡️ {school_env or '환경'} 데이터 개요")

            # 지표 카드 (있는 열만)
            metric_cols = {
                "temperature": "평균 온도",
                "humidity": "평균 습도",
                "ph": "평균 pH",
                "ec": "평균 EC",
            }
            for col_key, label in metric_cols.items():
                if col_key in env_df_std.columns:
                    vals = env_df_std[col_key].dropna()
                    if len(vals) >= 2:
                        st.metric(label, f"{vals.mean():.2f}")

            st.markdown("---")

            # 시계열 그래프 (있는 열만)
            time_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df_std.columns]
            if "time" in env_df_std.columns and time_cols:
                fig_trend = go.Figure()
                for col in time_cols:
                    series = env_df_std[["time", col]].dropna()
                    if len(series) >= 2:
                        fig_trend.add_trace(
                            go.Scatter(
                                x=series["time"],
                                y=series[col],
                                mode="lines",
                                name=col,
                            )
                        )
                fig_trend.update_layout(
                    title="환경 지표 시계열",
                    xaxis_title="시간",
                    yaxis_title="값",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_trend, width="stretch")
            else:
                st.info("📈 표시할 환경 지표가 없어요. (temperature, humidity, ph, ec 중 하나가 필요)")

            st.markdown("---")

            # 상관 분석: pH와 EC
            if "ph" in env_df_std.columns and "ec" in env_df_std.columns:
                ph_ec = env_df_std[["ph", "ec"]].dropna()
                if len(ph_ec) >= 2:
                    corr_pH_EC = ph_ec["ph"].corr(ph_ec["ec"])
                    st.markdown("### 🔗 pH와 EC의 관계")
                    st.info(
                        "상관계수(r)는 두 숫자가 함께 움직이는 정도를 -1~1 사이로 나타낸 값이에요. "
                        "0에 가까우면 관계가 약하고, ±1에 가까울수록 강한 관계예요."
                    )
                    st.metric("pH-EC 상관계수 (r)", f"{corr_pH_EC:.3f}")

                    # pH-EC 산점도 + 추세선
                    fig_ph_ec = go.Figure()
                    fig_ph_ec.add_trace(
                        go.Scatter(
                            x=ph_ec["ph"],
                            y=ph_ec["ec"],
                            mode="markers",
                            name="측정값",
                            marker=dict(size=6, color="steelblue"),
                        )
                    )
                    if len(ph_ec) >= 2:
                        slope, intercept = np.polyfit(ph_ec["ph"], ph_ec["ec"], 1)
                        x_vals = np.linspace(ph_ec["ph"].min(), ph_ec["ph"].max(), 100)
                        fig_ph_ec.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=slope * x_vals + intercept,
                                mode="lines",
                                name="추세선",
                                line=dict(color="red", dash="dash"),
                            )
                        )
                    fig_ph_ec.update_layout(
                        title="pH vs EC 산점도",
                        xaxis_title="pH",
                        yaxis_title="EC",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    st.plotly_chart(fig_ph_ec, width="stretch")
                else:
                    st.info("pH-EC 분석에 사용할 데이터가 부족해요.")
            else:
                st.info("📊 pH와 EC 열이 모두 있어야 산점도를 그릴 수 있어요.")

            st.markdown("---")

            # 환경 데이터 다운로드
            st.subheader("📥 환경 데이터 다운로드")
            csv_bytes = env_df_std.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="CSV 다운로드 (utf-8-sig)",
                data=csv_bytes,
                file_name=f"{school_env or '환경'}_표준화.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# 탭 2: 생육 데이터
# ══════════════════════════════════════════════
with tab2:
    try:
        if growth_df_std.empty:
            st.info("📁 사이드바에서 생육 CSV를 올려주세요.")
        else:
            st.subheader(f"🌱 {school_growth or '생육'} 데이터 개요")

            # 개체 수
            if "개체번호" in growth_df_std.columns:
                st.metric("개체 수", len(growth_df_std["개체번호"].dropna()))

            # 생육 지표 카드
            growth_metric_map = {
                "잎 수": "평균 잎 수",
                "지상부 길이": "평균 지상부 길이 (mm)",
                "지하부 길이": "평균 지하부 길이 (mm)",
                "생중량": "평균 생중량 (g)",
            }
            for col_key, label in growth_metric_map.items():
                if col_key in growth_df_std.columns:
                    vals = growth_df_std[col_key].dropna()
                    if len(vals) >= 2:
                        st.metric(label, f"{vals.mean():.2f}")

            st.markdown("---")

            # 생육 지표 분포 (box plot)
            dist_cols = [c for c in selected_growth if c in growth_df_std.columns]
            if dist_cols:
                fig_box = make_subplots(rows=1, cols=len(dist_cols), shared_yaxes=False)
                for i, col in enumerate(dist_cols):
                    vals = growth_df_std[col].dropna()
                    if len(vals) >= 2:
                        fig_box.add_trace(
                            go.Box(
                                y=vals,
                                name=col,
                                fillcolor="lightblue",
                                boxmean=True,
                            ),
                            row=1,
                            col=i + 1,
                        )
                fig_box.update_layout(
                    title="생육 지표 분포 (박스 플롯)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, width="stretch")
            else:
                st.info("표시할 생육 지표가 선택되지 않았어요.")

            st.markdown("---")

            # 개체별 생중량 순위 (있는 경우)
            if "생중량" in growth_df_std.columns and "개체번호" in growth_df_std.columns:
                growth_sorted = growth_df_std[["개체번호", "생중량"]].dropna().sort_values(
                    "생중량", ascending=False
                )
                if len(growth_sorted) >= 2:
                    top_n = min(5, len(growth_sorted))
                    bottom_n = min(5, len(growth_sorted))

                    st.markdown("### 🏆 상위 개체")
                    top_df = growth_sorted.head(top_n)
                    st.dataframe(
                        top_df.assign(
                            개체번호=top_df["개체번호"].astype(str)
                        ),
                        width="stretch",
                    )

                    st.markdown("### 📉 하위 개체")
                    bottom_df = growth_sorted.tail(bottom_n).iloc[::-1]
                    st.dataframe(
                        bottom_df.assign(
                            개체번호=bottom_df["개체번호"].astype(str)
                        ),
                        width="stretch",
                    )

                    # 생중량 막대 그래프
                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            x=growth_sorted["개체번호"].astype(str),
                            y=growth_sorted["생중량"],
                            marker=dict(color="teal"),
                            name="생중량",
                        )
                    )
                    fig_bar.update_layout(
                        title="개체별 생중량",
                        xaxis_title="개체번호",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_bar, width="stretch")

            st.markdown("---")

            # 생육 데이터 다운로드
            st.subheader("📥 생육 데이터 다운로드")
            csv_bytes = growth_df_std.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="CSV 다운로드 (utf-8-sig)",
                data=csv_bytes,
                file_name=f"{school_growth or '생육'}_표준화.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# 탭 3: 종합 분석
# ══════════════════════════════════════════════
with tab3:
    try:
        # 두 데이터 모두 필요
        if env_df_std.empty or growth_df_std.empty:
            if env_df_std.empty and growth_df_std.empty:
                st.info("📁 환경 CSV와 생육 CSV를 모두 올려주세요.")
            elif env_df_std.empty:
                st.info("📁 환경 CSV를 올려야 종합 분석을 할 수 있어요.")
            else:
                st.info("📁 생육 CSV를 올려야 종합 분석을 할 수 있어요.")
        else:
            st.subheader("📊 종합 분석")

            # ── 분석 1: 생중량 vs 지하부 길이 ──
            if "생중량" in growth_df_std.columns and "지하부 길이" in growth_df_std.columns:
                sub1 = growth_df_std[["생중량", "지하부 길이"]].dropna()
                if len(sub1) >= 2:
                    corr1 = sub1["생중량"].corr(sub1["지하부 길이"])
                    st.markdown("### ① 생중량 vs 지하부 길이")
                    st.info(
                        "생중량이 지하부 길이와 비례하는지 확인하는 분석이에요. "
                        "상관계수가 양수이면 지하부가 길수록 생중량이 큰 경향이 있어요."
                    )
                    st.metric("생중량-지하부길이 상관계수 (r)", f"{corr1:.3f}")

                    fig1 = go.Figure()
                    fig1.add_trace(
                        go.Scatter(
                            x=sub1["지하부 길이"],
                            y=sub1["생중량"],
                            mode="markers",
                            marker=dict(size=8, color="darkgreen"),
                            name="개체",
                        )
                    )
                    if len(sub1) >= 2:
                        s, i = np.polyfit(sub1["지하부 길이"], sub1["생중량"], 1)
                        x1 = np.linspace(sub1["지하부 길이"].min(), sub1["지하부 길이"].max(), 100)
                        fig1.add_trace(
                            go.Scatter(
                                x=x1,
                                y=s * x1 + i,
                                mode="lines",
                                name="추세선",
                                line=dict(color="orange", dash="dash"),
                            )
                        )
                    fig1.update_layout(
                        title="생중량 vs 지하부 길이",
                        xaxis_title="지하부 길이 (mm)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    st.plotly_chart(fig1, width="stretch")
                else:
                    st.info("생중량-지하부길이 분석에 데이터가 부족해요.")
            else:
                st.info("생중량과 지하부 길이 열이 둘 다 필요해요.")

            st.markdown("---")

            # ── 분석 2: pH와 EC 관계 ──
            if "ph" in env_df_std.columns and "ec" in env_df_std.columns:
                sub2 = env_df_std[["ph", "ec"]].dropna()
                if len(sub2) >= 2:
                    corr2 = sub2["ph"].corr(sub2["ec"])
                    st.markdown("### ② pH가 7에서 멀수록 EC가 높은가?")
                    st.info(
                        "'pH가 7에서 멀수록 EC가 높다'는 가설을 확인하는 분석이에요. "
                        "pH가 7보다 크거나 작을수록(절대값이 클수록) EC가 커지는지 살펴봐요."
                    )
                    st.metric("pH-EC 상관계수 (r)", f"{corr2:.3f}")

                    # pH 절대편차 vs EC
                    sub2["ph_abs_dev"] = (sub2["ph"] - 7).abs()
                    sub2b = sub2[["ph_abs_dev", "ec"]].dropna()
                    if len(sub2b) >= 2:
                        corr2b = sub2b["ph_abs_dev"].corr(sub2b["ec"])
                        st.metric("|pH-7| vs EC 상관계수 (r)", f"{corr2b:.3f}")

                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Scatter(
                                x=sub2b["ph_abs_dev"],
                                y=sub2b["ec"],
                                mode="markers",
                                marker=dict(size=8, color="purple"),
                                name="측정값",
                            )
                        )
                        if len(sub2b) >= 2:
                            s2, i2 = np.polyfit(sub2b["ph_abs_dev"], sub2b["ec"], 1)
                            x2 = np.linspace(
                                sub2b["ph_abs_dev"].min(), sub2b["ph_abs_dev"].max(), 100
                            )
                            fig2.add_trace(
                                go.Scatter(
                                    x=x2,
                                    y=s2 * x2 + i2,
                                    mode="lines",
                                    name="추세선",
                                    line=dict(color="red", dash="dash"),
                                )
                            )
                        fig2.update_layout(
                            title="|pH - 7| vs EC",
                            xaxis_title="|pH - 7| (7에서의 거리)",
                            yaxis_title="EC",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                        st.plotly_chart(fig2, width="stretch")
                    else:
                        st.info("|pH-7| vs EC 분석에 데이터가 부족해요.")
                else:
                    st.info("pH-EC 분석에 데이터가 부족해요.")
            else:
                st.info("pH와 EC 열이 환경에셋에 둘 다 필요해요.")

            st.markdown("---")

            # ── 분석 3: 생육 지표 간 상관 매트릭스 ──
            corr_cols = [c for c in selected_growth if c in growth_df_std.columns]
            if len(corr_cols) >= 2:
                corr_df = growth_df_std[corr_cols].dropna()
                if len(corr_df) >= 2:
                    corr_matrix = corr_df.corr()
                    st.markdown("### ③ 생육 지표 간 상관계수 행렬")
                    st.info(
                        "표에서 각 칸은 두 지표 사이의 상관계수(r)예요. "
                        "1에 가까울수록 함께 증가하고, -1에 가까울수록 함께 감소해요."
                    )

                    fig3 = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                    )
                    fig3.update_layout(
                        title="생육 지표 상관 행렬",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="",
                        yaxis_title="",
                    )
                    fig3.update_xaxes(side="top")
                    st.plotly_chart(fig3, width="stretch")
                else:
                    st.info("상관 행렬 분석에 데이터가 부족해요.")
            else:
                st.info("상관 행렬을 그리려면 생육 지표가 2개 이상 선택되어야 해요.")

            st.markdown("---")

            # ── 분석 4: 데이터 요약 ──
            st.subheader("📋 데이터 요약")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**환경 데이터**")
                env_summary = []
                for c in ["temperature", "humidity", "ph", "ec"]:
                    if c in env_df_std.columns:
                        vals = env_df_std[c].dropna()
                        if len(vals) >= 2:
                            env_summary.append(
                                {
                                    "지표": c,
                                    "평균": f"{vals.mean():.2f}",
                                    "최대": f"{vals.max():.2f}",
                                    "최소": f"{vals.min():.2f}",
                                    "측정횟수": len(vals),
                                }
                            )
                if env_summary:
                    st.dataframe(pd.DataFrame(env_summary), width="stretch")
                else:
                    st.info("환경 데이터 요약 정보가 없어요.")

            with col2:
                st.markdown("**생육 데이터**")
                growth_summary = []
                for c in selected_growth:
                    if c in growth_df_std.columns:
                        vals = growth_df_std[c].dropna()
                        if len(vals) >= 2:
                            growth_summary.append(
                                {
                                    "지표": c,
                                    "평균": f"{vals.mean():.2f}",
                                    "최대": f"{vals.max():.2f}",
                                    "최소": f"{vals.min():.2f}",
                                    "측정개체수": len(vals),
                                }
                            )
                if growth_summary:
                    st.dataframe(pd.DataFrame(growth_summary), width="stretch")
                else:
                    st.info("생육 데이터 요약 정보가 없어요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
