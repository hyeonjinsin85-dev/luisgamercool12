# =============================================================================
# 🌱 우리 학교 극지식물 개체별 생육 차이 분석 — Streamlit 대시보드
# 학교: 송도고
# 대상 파일:
#   · 환경 데이터: time, temperature, humidity, ph, ec
#   · 생육 결과 데이터: 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g)
# =============================================================================

import io
import platform
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# 한글 폰트 설정 (OS별 자동 선택)
# ──────────────────────────────────────────────────────────────────────────────
system = platform.system()

if system == "Darwin":  # macOS
    mpl_font = "AppleGothic"
elif system == "Windows":
    mpl_font = "Malgun Gothic"
else:  # Linux (Streamlit Cloud 등)
    mpl_font = "NanumGothic"

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = mpl_font
matplotlib.rcParams["axes.unicode_minus"] = False

# pip에 없는 폰트는 log만 내고 넘어감 (런타임 에러 방지)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        matplotlib.font_manager.findfont(mpl_font, fallback_to_default=False)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# CSV 읽기 헬퍼 — 인코딩 · 열 이름 변동 대응
# ──────────────────────────────────────────────────────────────────────────────
def read_env_csv(file) -> pd.DataFrame | None:
    """
    환경 CSV를 읽는다.
    - 바이트를 직접 꺼내 읽어 pd.read_csv 중복 호출 문제를 방지한다.
    - utf-8-sig → cp949 순서로 시도한다.
    - school마다 다른 열 이름·시간 형식을 최대한 정규화한다.
    """
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            if df.shape[0] > 0 and df.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        return None

    # 앞뒤 공백 정리
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).strip() if pd.notna(x) else x)

    # 시간 열 탐색 (학교마다 이름 다름)
    time_candidates = [
        "time", "Time", "일시", "시간", "timestamp", "date", "날짜",
        "categories", "측정일시",
    ]
    time_col = None
    for c in df.columns:
        if c in time_candidates or "time" in c.lower() or "일시" in c or "날짜" in c:
            time_col = c
            break
    if time_col is None:
        # 첫 번째 열이 날짜처럼 보이면 채택
        first_col = df.columns[0]
        sample = str(df[first_col].iloc[0])
        if "202" in sample or "_" in sample or "-" in sample:
            time_col = first_col

    if time_col is None:
        return None

    # 시간 파싱 — 형식 A: '2026-07-15 06:00:00' / 형식 B: '20260715_0'
    tm = df[time_col].astype(str).str.strip()
    if "_" in tm.iloc[0] and " " not in tm.iloc[0]:
        # 형식 B: '20260715_0' → '2026-07-15 00:00:00'
        tm = tm.str.replace("_", " ", regex=False)
        tm = pd.to_datetime(tm, format="%Y%m%d %H", errors="coerce")
    else:
        tm = pd.to_datetime(tm, errors="coerce")

    df["datetime"] = tm
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # 측정값 열 정규화 (소문자 영문 기준으로 매핑)
    col_map = {}
    for orig in df.columns:
        low = orig.lower()
        if low == "temperature" or low == "온도" or low == "temp":
            col_map[orig] = "temperature"
        elif low == "humidity" or low == "습도" or low == "hum":
            col_map[orig] = "humidity"
        elif low == "ph":
            col_map[orig] = "ph"
        elif low == "ec":
            col_map[orig] = "ec"
        elif low == "water_temp" or low == "수온":
            col_map[orig] = "water_temp"
        elif low == "co2":
            col_map[orig] = "co2"
        elif low == "비고" or low == "note" or low == "remark":
            col_map[orig] = "_note"
    df = df.rename(columns=col_map)

    # 숫자형 정리 (문자열 섞인 숫자 처리)
    for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [c for c in ["datetime", "temperature", "humidity", "ph", "ec",
                        "water_temp", "co2", "_note"] if c in df.columns]
    return df[keep]


def read_growth_csv(file) -> pd.DataFrame | None:
    """
    생육 결과 CSV를 읽는다.
    - utf-8-sig → cp949 자동 폴백
    - 학교마다 다른 열 이름(단위·띄어쓰기 포함) 대응
    - 생중량(g)이 없으면 지상부질량 + 지하부질량으로 계산
    """
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            if df.shape[0] > 0 and df.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        return None

    df.columns = [str(c).strip() for c in df.columns]

    # 개체번호 열 탐색
    id_candidates = ["개체번호", "개체", "번호", "id", "ID", "sample", "개체 id"]
    id_col = None
    for c in df.columns:
        if c in id_candidates or "개체" in c or "번호" in c or c.lower() in ("id", "sample"):
            id_col = c
            break
    if id_col is None:
        # 첫 번째 열을 개체번호로 간주
        id_col = df.columns[0]

    # 측정 항목 정규화 맵
    rename_map = {}
    for orig in df.columns:
        low = orig.replace(" ", "").lower().replace("(", "").replace(")", "")
        if "잎" in low or "leaf" in low:
            rename_map[orig] = "leaf_count"
        elif "지상부길이" in low or "지상부 길이" in low or "shoot" in low or "aboveground" in low:
            rename_map[orig] = "shoot_length"
        elif "지하부길이" in low or "지하부 길이" in low or "root" in low or "belowground" in low:
            rename_map[orig] = "root_length"
        elif "생중량" in low or "freshweight" in low or "fresh_weight" in low:
            rename_map[orig] = "fresh_weight"
        elif "지상부질량" in low or "지상부 질량" in low or "shootmass" in low or "abovegroundmass" in low:
            rename_map[orig] = "shoot_mass"
        elif "지하부질량" in low or "지하부 질량" in low or "rootmass" in low or "belowgroundmass" in low:
            rename_map[orig] = "root_mass"

    df = df.rename(columns=rename_map)
    df = df.rename(columns={id_col: "개체번호"})

    # 숫자열 정리
    num_cols = [c for c in ["leaf_count", "shoot_length", "root_length",
                            "fresh_weight", "shoot_mass", "root_mass"]
                if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 생중량 계산 (없을 때만)
    if "fresh_weight" not in df.columns:
        if "shoot_mass" in df.columns and "root_mass" in df.columns:
            df["fresh_weight"] = df["shoot_mass"] + df["root_mass"]
        elif "shoot_mass" in df.columns:
            df["fresh_weight"] = df["shoot_mass"]
        elif "root_mass" in df.columns:
            df["fresh_weight"] = df["root_mass"]

    keep = [c for c in ["개체번호", "leaf_count", "shoot_length", "root_length",
                        "fresh_weight", "shoot_mass", "root_mass"] if c in df.columns]
    df = df[keep].dropna(subset=["fresh_weight"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit 페이지 설정
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
st.caption("송도고 극지식물 재배 실험 데이터 대시보드 · 올려주신 CSV 그대로 사용 가능")

# ──────────────────────────────────────────────────────────────────────────────
# 사이드바 — CSV 업로드
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")

env_file = st.sidebar.file_uploader(
    "환경 데이터 CSV",
    type=["csv"],
    help="열 이름 예: time, temperature, humidity, ph, ec  (학교마다 다를 수 있음)",
)
growth_file = st.sidebar.file_uploader(
    "생육 결과 CSV",
    type=["csv"],
    help="열 이름 예: 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g)",
)

measure_cols = ["leaf_count", "shoot_length", "root_length", "fresh_weight"]
measure_labels = {
    "leaf_count": "잎 수",
    "shoot_length": "지상부 길이",
    "root_length": "지하부 길이",
    "fresh_weight": "생중량",
}
selected_measures = st.sidebar.multiselect(
    "📐 생육 측정 항목 선택",
    options=measure_cols,
    default=["fresh_weight", "leaf_count", "shoot_length", "root_length"],
    format_func=lambda x: measure_labels.get(x, x),
)

env_df = read_env_csv(env_file) if env_file else None
growth_df = read_growth_csv(growth_file) if growth_file else None

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 없을 때 안내
# ──────────────────────────────────────────────────────────────────────────────
if env_df is None and growth_df is None:
    st.info("⚡ 사이드바에서 CSV 파일 2개를 업로드하면 분석이 시작됩니다.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 통계 용어 설명 (고등학생 대상, 짧게)
# ──────────────────────────────────────────────────────────────────────────────
STAT_GLOSS = {
    "평균": "모든 값을 더해 개수로 나눈 representative 값. 데이터의 '중심'을 뜻한다.",
    "표준편차": "값들이 평균에서 얼마나 넓게 퍼져 있는지를 나타낸다. 클수록 개체 간 차이가 크다.",
    "변동계수(CV)": "표준편차를 평균으로 나눈 값(% 단위). 평균 크기가 다른 항목끼리 퍼짐 정도를 비교할 때 쓴다.",
    "상관계수(r)": "두 변수가 함께 움직이는 정도를 -1 ~ +1 사이 숫자로 나타낸다. +1에 가까울수록 함께 증가, -1이면 한쪽 증가 시 다른 쪽 감소.",
    "사분위수": "데이터를 크기순으로 정렬했을 때 4등분하는 지점. 상위 25% = 3사분위수(Q3) 이상, 하위 25% = 1사분위수(Q1) 이하.",
}


def stat_card(label, value, explanation=None):
    """핵심 지표 카드 하나."""
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.markdown(f"# {value}")
        if explanation:
            st.caption(explanation)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — 📖 실험 개요
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
])

with tab1:
    st.header("연구 배경 및 목적")

    st.markdown(
        """
**연구 배경**
극지식물은 낮은 온도, 강한 자외선, 물 부족 같은 혹독한 환경에서 살아남도록 적응한 식물이다.
같은 조건에서도 개체마다 자란 정도가 다를 수 있는데, 이 차이가 우연인지 아니면 측정 가능한
환경·형태 요인과 관련 있는지 확인하는 것이 실험의 출발점이다.

**연구 목적**
1. 개체별 생중량(fresh weight) 분포를 확인하고, 평균·표준편차·변동계수로 개체 간 차이를 수치화한다.
2. 잎 수, 지상부 길이와 생중량 사이에 어떤 선형 관계가 있는지 상관계수와 산점도로 살펴본다.
3. 생중량이 큰 개체군(상위 25%)과 작은 개체군(하위 25%)을 비교해 어떤 형태적 차이가 있는지 알아본다.
"""
    )

    # 데이터 요약 표
    st.subheader("📋 데이터 요약")

    n_growth = len(growth_df) if growth_df is not None else 0
    n_env = len(env_df) if env_df is not None else 0

    if env_df is not None:
        env_start = env_df["datetime"].min()
        env_end = env_df["datetime"].max()
        env_days = (env_end - env_start).days + 1
    else:
        env_start = env_end = env_days = None

    if growth_df is not None:
        gw_cols = [c for c in ["leaf_count", "shoot_length", "root_length", "fresh_weight"]
                   if c in growth_df.columns]
    else:
        gw_cols = []

    summary_data = {
        "항목": [
            "총 개체 수",
            "측정 기간 (환경 센서)",
            "환경 측정 항목 수",
            "생육 측정 항목 수",
            "측정 항목 목록",
        ],
        "값": [
            f"{n_growth}개체",
            f"{env_start.strftime('%Y-%m-%d') if env_start is not None else '—'} ~ "
            f"{env_end.strftime('%Y-%m-%d') if env_end is not None else '—'}"
            f" (총 {env_days}일)" if env_days else "—",
            f"{len([c for c in ['temperature','humidity','ph','ec','water_temp','co2'] if c in (env_df.columns if env_df is not None else [])])}개",
            f"{len(gw_cols)}개",
            ", ".join(measure_labels.get(c, c) for c in gw_cols) if gw_cols else "—",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data).set_index("항목"), hide_index=True, use_container_width=True)

    # 주요 지표 카드 4개
    st.subheader("📊 주요 지표")
    c1, c2, c3, c4 = st.columns(4)

    if growth_df is not None and "fresh_weight" in growth_df.columns:
        mean_fw = growth_df["fresh_weight"].mean()
        card4_val = f"{mean_fw:.2f} g"
        card4_exp = f"개체 {n_growth}마리 기준 평균 생중량. 표준편차: {growth_df['fresh_weight'].std():.2f} g"
    else:
        card4_val = "—"
        card4_exp = None

    with c1:
        stat_card("총 개체 수", f"{n_growth}개체",
                  STAT_GLOSS["평균"].split(".")[0] + " (개체 수를 뜻함)")
    with c2:
        if env_df is not None and "temperature" in env_df.columns:
            mean_temp = env_df["temperature"].mean()
            stat_card("평균 온도", f"{mean_temp:.1f} °C",
                      "센서 측정 전체 기간의 시간별 온도 평균.")
        else:
            stat_card("평균 온도", "—", None)
    with c3:
        if env_df is not None and "humidity" in env_df.columns:
            mean_hum = env_df["humidity"].mean()
            stat_card("평균 습도", f"{mean_hum:.1f} %",
                      "센서 측정 전체 기간의 시간별 습도 평균.")
        else:
            stat_card("평균 습도", "—", None)
    with c4:
        stat_card("평균 생중량", card4_val, card4_exp)

    if growth_df is not None and "fresh_weight" in growth_df.columns:
        st.caption("📌 용어: 생중량(fresh weight)은 수확 직후 물기를 빼지 않은 상태의 무게(g). "
                   "식물의 전체 생체량을 간단히 나타내는 지표다.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — 🌡️ 환경 데이터
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if env_df is None:
        st.warning("환경 데이터 CSV를 업로드하면 이 탭의 분석이 표시됩니다.")
        st.stop()

    st.header("🌡️ 환경 데이터 요약")

    # 요약 표 (평균·최소·최대)
    env_stats_cols = [c for c in ["temperature", "humidity", "ph", "ec",
                                   "water_temp", "co2"]
                      if c in env_df.columns]
    if env_stats_cols:
        rows = []
        for c in env_stats_cols:
            s = env_df[c].dropna()
            rows.append({
                "측정 항목": measure_labels.get(c, c),
                "평균": f"{s.mean():.2f}",
                "최소": f"{s.min():.2f}",
                "최대": f"{s.max():.2f}",
                "표준편차": f"{s.std():.2f}",
                "측정 횟수": f"{len(s)}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("측정값 수 columns(temperature, humidity, ph, ec 등)이 없어 요약을 표시할 수 없습니다.")

    st.divider()

    # 일별 시계열
    st.subheader("📈 날짜별 환경 변화 (일별 평균)")

    env_daily = env_df.set_index("datetime").resample("D").mean(numeric_only=True).reset_index()
    env_daily = env_daily.dropna(subset=["datetime"])

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("#### 온도 변화")
        if "temperature" in env_daily.columns:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(env_daily["datetime"], env_daily["temperature"],
                    color="#e74c3c", marker="o", markersize=3, linewidth=1.6)
            ax.set_ylabel("온도 (°C)")
            ax.set_xlabel("")
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("temperature 열이 없어 온도 그래프를 그릴 수 없습니다.")

    with t2:
        st.markdown("#### 습도 변화")
        if "humidity" in env_daily.columns:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(env_daily["datetime"], env_daily["humidity"],
                    color="#3498db", marker="o", markersize=3, linewidth=1.6)
            ax.set_ylabel("습도 (%)")
            ax.set_xlabel("")
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("humidity 열이 없어 습도 그래프를 그릴 수 없습니다.")

    # pH·EC 변화 (열이 있을 때만)
    phec_cols = [c for c in ["ph", "ec"] if c in env_daily.columns]
    if phec_cols:
        st.markdown("#### pH · EC 변화")
        fig, axes = plt.subplots(1, len(phec_cols), figsize=(5 * len(phec_cols), 3.2))
        if len(phec_cols) == 1:
            axes = [axes]
        colors_phec = {"ph": "#27ae60", "ec": "#8e44ad"}
        for ax, c in zip(axes, phec_cols):
            ax.plot(env_daily["datetime"], env_daily[c],
                    color=colors_phec.get(c, "#333"), marker="o",
                    markersize=3, linewidth=1.6)
            ax.set_ylabel(f"{c.upper()} " + ("(pH)" if c == "ph" else "(mS/cm)"))
            ax.set_xlabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        fig.autofmt_xdate()
        st.pyplot(fig, use_container_width=True)

    # 환경 원본 테이블 + CSV 다운로드
    with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
        st.dataframe(env_df.drop(columns=["datetime"], errors="ignore").head(50),
                    hide_index=True, use_container_width=True)
        csv_out = env_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="CSV 다운로드",
            data=csv_out,
            file_name="환경데이터_원본.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — 📊 생육 결과
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if growth_df is None:
        st.warning("생육 결과 CSV를 업로드하면 이 탭의 분석이 표시됩니다.")
        st.stop()

    st.header("📊 생육 결과 분석")

    fw_col = "fresh_weight"
    gw_base = [c for c in selected_measures if c in growth_df.columns]
    fw_present = fw_col in growth_df.columns

    if not fw_present:
        st.warning("생중량(fresh_weight) 열이 없습니다. 지상부질량 + 지하부질량으로 계산하거나, "
                   "업로드 파일에 생중량 열을 포함해 주세요.")
        st.stop()

    fw = growth_df[fw_col].dropna()
    n = len(fw)

    # ── 핵심 결과 카드: 평균 생중량, 최대·최소 개체 ──
    mean_fw = fw.mean()
    std_fw = fw.std()
    cv_fw = (std_fw / mean_fw) * 100 if mean_fw > 0 else float("nan")

    max_idx = fw.idxmax()
    min_idx = fw.idxmin()
    max_row = growth_df.loc[max_idx]
    min_row = growth_df.loc[min_idx]

    st.subheader("🥇 핵심 결과")

    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        stat_card(
            "평균 생중량",
            f"{mean_fw:.2f} g",
            f"표준편차 ±{std_fw:.2f} g · 변동계수(CV) {cv_fw:.1f}%  "
            f"(CV가 클수록 개체 간 차이가 큼) ",
        )
    with kc2:
        stat_card(
            "최대 생중량 개체",
            f"개체 {max_row['개체번호']}",
            f"생중량 {max_row[fw_col]:.2f} g · "
            f"잎 {int(max_row['leaf_count']) if 'leaf_count' in max_row else '—'}장 · "
            f"지상부 {max_row['shoot_length']:.0f} mm"
            if "shoot_length" in max_row else "—",
        )
    with kc3:
        stat_card(
            "최소 생중량 개체",
            f"개체 {min_row['개체번호']}",
            f"생중량 {min_row[fw_col]:.2f} g · "
            f"잎 {int(min_row['leaf_count']) if 'leaf_count' in min_row else '—'}장 · "
            f"지상부 {min_row['shoot_length']:.0f} mm"
            if "shoot_length" in min_row else "—",
        )

    st.caption(
        "📌 용어 정리 — "
        + " · ".join(f"**{k}**: {v}" for k, v in STAT_GLOSS.items()
                    if k in ("평균", "표준편차", "변동계수(CV)"))
    )

    st.divider()

    # ── 개체별 생육 비교 막대 그래프 ──
    st.subheader("📊 개체별 생육 비교 (막대 그래프)")

    bar_targets = [c for c in selected_measures if c != fw_col and c in growth_df.columns]
    plot_targets = [fw_col] + bar_targets

    if plot_targets:
        n_plots = len(plot_targets)
        fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 3.5))
        if n_plots == 1:
            axes = [axes]
        colors_bar = {"fresh_weight": "#2ecc71", "leaf_count": "#f39c12",
                      "shoot_length": "#3498db", "root_length": "#e74c3c"}
        for ax, tgt in zip(axes, plot_targets):
            vals = growth_df[tgt].dropna()
            ax.bar(range(len(vals)), vals,
                   color=colors_bar.get(tgt, "#95a5a6"),
                   edgecolor="white", linewidth=0.5)
            ax.axhline(vals.mean(), color="black", linestyle="--",
                       linewidth=1, label=f"평균 {vals.mean():.1f}")
            ax.set_title(measure_labels.get(tgt, tgt), fontsize=11)
            ax.set_xlabel("개체 번호 (정렬 순서)")
            ax.set_ylabel(measure_labels.get(tgt, tgt))
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # ── 생중량 분포 (박스플롯) ──
    st.subheader("📦 생중량 분포 (박스플롯)")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bp = ax.boxplot(fw, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#2ecc71", alpha=0.6),
                    medianprops=dict(color="black", linewidth=2))
    ax.set_ylabel(f"생중량 ({fw_col}) (g)")
    ax.set_xticklabels(["개체 전체"])
    ax.axhline(mean_fw, color="red", linestyle="--", linewidth=1,
               label=f"평균 {mean_fw:.2f} g")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.caption(
        "📌 박스플롯 읽기: 상자(Box)는 중앙 50% 데이터의 범위(Q1~Q3), "
        "가운데 선은 중앙값, 수염(Whisker)은 전형적 범위, "
        "점들은 이상치 후보를 뜻한다. "
    )

    # ── 상관관계 분석 ──
    st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")

    corr_targets = [c for c in ["leaf_count", "shoot_length", "root_length"]
                    if c in growth_df.columns]
    if corr_targets:
        tc1, tc2 = st.columns(2)
        plot_idx = 0
        for tgt in corr_targets:
            tgt_col = growth_df[tgt].dropna()
            common = fw.index.intersection(tgt_col.index)
            x = tgt_col.loc[common].values
            y = fw.loc[common].values

            if len(x) < 3:
                continue

            r = np.corrcoef(x, y)[0, 1]

            with (tc1 if plot_idx == 0 else tc2):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.scatter(x, y, color="#2980b9", alpha=0.7, edgecolors="white",
                           linewidth=0.5, s=55)
                # 추세선 (1차 선형 회귀)
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, slope * xs + intercept, color="#e74c3c",
                        linewidth=1.8, linestyle="--",
                        label=f"추세선 (r = {r:.2f})")
                ax.set_xlabel(measure_labels.get(tgt, tgt))
                ax.set_ylabel(f"생중량 (g)")
                ax.set_title(f"{measure_labels.get(tgt, tgt)} vs 생중량", fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                st.caption(
                    f"상관계수 r = {r:.2f} — "
                    + ("강한 양의 상관" if r > 0.7
                       else "중간 양의 상관" if r > 0.4
                       else "약한 관계" if r > 0
                       else "약한 음의 관계" if r > -0.4
                       else "중간 음의 상관" if r > -0.7
                       else "강한 음의 상관")
                    + f" "
                )
                plot_idx += 1
        if plot_idx == 0:
            st.info("상관관계 분석에 필요한 열(잎 수·지상부 길이·지하부 길이)이 없습니다.")
    else:
        st.info("상관관계 분석에 필요한 열(잎 수, 지상부 길이, 지하부 길이) 중 하나가 없습니다.")

    # ── 상위 25% vs 하위 25% 개체 비교 ──
    st.divider()
    st.subheader("🏆 생중량 상위 25% vs 하위 25% 개체 비교")

    q1 = fw.quantile(0.25)
    q3 = fw.quantile(0.75)
    top_mask = fw >= q3
    bot_mask = fw <= q1
    top_ids = growth_df.loc[top_mask[top_mask].index, "개체번호"].tolist()
    bot_ids = growth_df.loc[bot_mask[bot_mask].index, "개체번호"].tolist()

    comp_cols = st.columns(2)
    with comp_cols[0]:
        st.markdown("#### 🔝 상위 25% (생중량 큰 개체)")
        st.markdown(f"기준: 생중량 ≥ Q3 = **{q3:.2f} g**  (상위 사분위수)")
        st.markdown(f"개체 수: **{len(top_ids)}마리**  "
                    f"({', '.join(str(i) for i in top_ids)})")
        top_tbl = growth_df.loc[top_mask[top_mask].index].copy()
        if "leaf_count" in top_tbl.columns:
            st.markdown(
                f"평균 잎 수: **{top_tbl['leaf_count'].mean():.1f}장** · "
                f"평균 지상부 길이: **{top_tbl['shoot_length'].mean():.0f} mm**"
                + (f" · 평균 지하부 길이: **{top_tbl['root_length'].mean():.0f} mm**"
                   if "root_length" in top_tbl.columns else "")
            )

    with comp_cols[1]:
        st.markdown("#### 🔻 하위 25% (생중량 작은 개체)")
        st.markdown(f"기준: 생중량 ≤ Q1 = **{q1:.2f} g**  (하위 사분위수)")
        st.markdown(f"개체 수: **{len(bot_ids)}마리**  "
                    f"({', '.join(str(i) for i in bot_ids)})")
        bot_tbl = growth_df.loc[bot_mask[bot_mask].index].copy()
        if "leaf_count" in bot_tbl.columns:
            st.markdown(
                f"평균 잎 수: **{bot_tbl['leaf_count'].mean():.1f}장** · "
                f"평균 지상부 길이: **{bot_tbl['shoot_length'].mean():.0f} mm**"
                + (f" · 평균 지하부 길이: **{bot_tbl['root_length'].mean():.0f} mm**"
                   if "root_length" in bot_tbl.columns else "")
            )

    st.caption(
        "📌 사분위수 기준: Q1(하위 25 경계) = "
        + STAT_GLOSS["사분위수"].split(".")[0]
        + f", Q3(상위 25 경계) = 같은 방식의 상위 경계. "
        + "이 기준은 분석 관행이며 절대적 기준은 아니다."
    )

    # 생육 원본 테이블 + CSV 다운로드
    with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
        st.dataframe(growth_df, hide_index=True, use_container_width=True)
        csv_out = growth_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="CSV 다운로드",
            data=csv_out,
            file_name="생육결과_원본.csv",
            mime="text/csv",
        )
