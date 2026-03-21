import streamlit as st
from openai import OpenAI
import pandas as pd
import re
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import gspread
from google.oauth2.service_account import Credentials

# 1. API 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="SKIM Ensemble System", layout="wide")
st.title("🧠 소백현: 앙상블 CRP 분석 엔진 (Lite)")
st.markdown("3회 교차 검증 **앙상블**을 통해 신뢰도를 높이고 인지 분석 리포트를 생성합니다.")
st.divider()

INDICATORS = ['MTI', 'REC', 'RECON', 'ORC', 'SRR', 'TTR', 'POI']
ENSEMBLE_N = 3
SHEET_NAME = "SKIM_Timeseries"  # Google Sheets 파일명

# ─────────────────────────────────────────────
# Google Sheets 연결
# ─────────────────────────────────────────────

@st.cache_resource
def get_gsheet_client():
    """서비스 계정 키로 Google Sheets 클라이언트 반환 (캐싱으로 재연결 방지)"""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],  # secrets.toml에 JSON 키 등록
        scopes=scopes
    )
    return gspread.authorize(creds)

def get_worksheet(u_id: str):
    """학습자 ID별 워크시트 반환. 없으면 자동 생성 후 헤더 삽입"""
    gc = get_gsheet_client()
    spreadsheet = gc.open(SHEET_NAME)
    try:
        ws = spreadsheet.worksheet(u_id)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=u_id, rows=1000, cols=20)
        ws.append_row([
            "timestamp", "MTI", "Rec", "Recon", "Orc",
            "SRR", "TTR", "POI", "insight_summary"
        ])
    return ws


# ─────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────

def normalize_score(v: float) -> float:
    if v > 100:
        v = v / 10
    elif v > 10:
        v = v / 10
    return max(1.0, min(10.0, round(v, 2)))


def parse_response(text: str) -> tuple[list[float], str]:
    data_match = re.search(r"\[DATA\](.*?)\[INSIGHT\]", text, re.S | re.I)
    if not data_match:
        raise ValueError("응답에서 [DATA]...[INSIGHT] 구조를 찾을 수 없습니다.")
    raw_vals = re.findall(
        r"(?:MTI|REC|RECON|ORC|SRR|TTR|POI)\s*:\s*(\d+\.?\d*)",
        data_match.group(1), re.I
    )
    if len(raw_vals) < len(INDICATORS):
        raise ValueError(f"지표 {len(INDICATORS)}개 필요, {len(raw_vals)}개만 파싱됨.")
    scores = [normalize_score(float(v)) for v in raw_vals[:len(INDICATORS)]]
    insight = text.split("[INSIGHT]", 1)[-1].strip()
    if not insight:
        raise ValueError("INSIGHT 섹션이 비어 있습니다.")
    return scores, insight


# ─────────────────────────────────────────────
# 시계열 저장 / 로드
# ─────────────────────────────────────────────

def save_timeseries(u_id: str, avg: pd.Series, insight: str) -> None:
    """분석 결과를 Google Sheets에 누적 저장"""
    ws = get_worksheet(u_id)
    ws.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        round(float(avg["MTI"]), 2), round(float(avg["Rec"]), 2),
        round(float(avg["Recon"]), 2), round(float(avg["Orc"]), 2),
        round(float(avg["SRR"]), 2), round(float(avg["TTR"]), 2),
        round(float(avg["POI"]), 2),
        insight[:100] + ("..." if len(insight) > 100 else "")
    ])

def load_timeseries(u_id: str) -> pd.DataFrame | None:
    """Google Sheets에서 학습자 시계열 데이터 로드"""
    try:
        ws = get_worksheet(u_id)
        records = ws.get_all_records()
        if not records:
            return None
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None


# ─────────────────────────────────────────────
# PDF 생성
# ─────────────────────────────────────────────

def create_ensemble_pdf(content, u_id, time, chart_img, avg):
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("Nanum", "", "NanumGothic.ttf")
        pdf.add_font("Nanum", "B", "NanumGothicBold.ttf")
        tf, bf, sf = ("Nanum", "", 11), ("Nanum", "B", 16), ("Nanum", "B", 12)
    except Exception:
        tf, bf, sf = ("Helvetica", "", 11), ("Helvetica", "B", 16), ("Helvetica", "B", 12)

    pdf.set_font(*bf)
    pdf.cell(0, 15, "SKIM Ensemble Cognitive Report", ln=True, align="C")
    pdf.set_font(*tf)
    pdf.cell(0, 5, f"ID: {u_id} | Date: {time} | Ensemble N={ENSEMBLE_N}", ln=True, align="C")
    pdf.ln(10)

    chart_img.seek(0)
    chart_y = pdf.get_y()
    pdf.image(chart_img, x=15, w=170)          # x 여백 통일, w 확대
    pdf.ln(8)                                   # 차트 하단 여백 (고정값 제거)

    pdf.set_font(*sf)
    pdf.cell(0, 10, " [ Ensemble Analytics Summary ]", ln=True)
    pdf.set_font(*tf)
    for k in ["MTI", "Rec", "Recon", "Orc"]:
        if k in avg:
            pdf.cell(0, 8, f"• Average {k}: {avg[k]:.2f} / 10.0", ln=True)
    pdf.ln(5)

    pdf.set_font(*sf)
    pdf.cell(0, 10, " [ Expert Insight ]", ln=True)
    pdf.set_font(*tf)
    pdf.multi_cell(0, 8, txt=content.replace("#", "").replace("*", "").strip())

    pdf_out = pdf.output(dest="S")
    return bytes(pdf_out) if isinstance(pdf_out, (str, bytearray)) else pdf_out


# ─────────────────────────────────────────────
# 탭 구성: 분석 탭 / 시계열 대시보드 탭
# ─────────────────────────────────────────────

tab_analyze, tab_dashboard = st.tabs(["🔬 분석", "📈 시계열 대시보드"])

# ══════════════════════════════════════════════
# TAB 1: 분석
# ══════════════════════════════════════════════
with tab_analyze:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("👤 학습자 정보")
        user_id = st.text_input("학습자 ID", max_chars=10, value="sj")
        log_input = st.text_area("분석할 [Conversation Log] 입력", height=500)
        analyze_btn = st.button(
            f"🚀 앙상블 분석 ({ENSEMBLE_N}회 교차 검증)",
            use_container_width=True
        )

    if analyze_btn and user_id and log_input:
        with col2:
            analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scores_list, errors, last_insight = [], [], ""

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(ENSEMBLE_N):
                status_text.text(f"분석 중... ({i+1}/{ENSEMBLE_N}회)")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "너는 CRP 분석 시스템이다.\n"
                                    "반드시 아래 형식만 사용하라:\n"
                                    "[DATA]\n"
                                    "MTI: <0-10>\nREC: <0-10>\nRECON: <0-10>\n"
                                    "ORC: <0-10>\nSRR: <0-10>\nTTR: <0-10>\nPOI: <0-10>\n"
                                    "[INSIGHT]\n<한국어 전문가 해설>"
                                )
                            },
                            {"role": "user", "content": f"ID: {user_id}\nLog:\n{log_input}"}
                        ],
                        temperature=0.1
                    )
                    scores, insight = parse_response(response.choices[0].message.content)
                    scores_list.append(scores)
                    last_insight = insight
                except ValueError as ve:
                    errors.append(f"회차 {i+1} 파싱 오류: {ve}")
                except Exception as e:
                    errors.append(f"회차 {i+1} API 오류: {e}")

                progress_bar.progress(int((i + 1) / ENSEMBLE_N * 100))

            status_text.empty()

            if errors:
                with st.expander(f"⚠️ 오류 {len(errors)}건"):
                    for err in errors:
                        st.warning(err)

            if scores_list:
                df = pd.DataFrame(scores_list,
                                  columns=["MTI", "Rec", "Recon", "Orc", "SRR", "TTR", "POI"])
                avg = df.mean()
                std = df.std() if len(scores_list) > 1 else None

                st.success(f"✅ {len(scores_list)}/{ENSEMBLE_N}회 분석 완료")

                if std is not None:
                    st.caption(
                        "📊 앙상블 표준편차 — "
                        + " | ".join(f"{c}: ±{std[c]:.2f}" for c in ["Rec", "Recon", "Orc"])
                    )

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(
                    ["Rec", "Recon", "Orc"],
                    [avg["Rec"], avg["Recon"], avg["Orc"]],
                    color=["#3498db", "#e74c3c", "#2ecc71"],
                    yerr=[std["Rec"], std["Recon"], std["Orc"]] if std is not None else None,
                    capsize=5
                )
                ax.set_ylim(0, 10)
                ax.set_title("Ensemble Average (with std)")

                img_buf = io.BytesIO()
                plt.savefig(img_buf, format="png", bbox_inches="tight")
                plt.close(fig)
                img_buf.seek(0)
                st.image(img_buf)

                # ── 학습자용 한 줄 해석 카드 ──
                st.markdown("### 📌 오늘 나의 인지 상태")
                INDICATOR_GUIDE = {
                    "MTI":   ("사고 전환",  [(1,4,"새로운 시각으로 전환하는 데 어려움이 있어요."),
                                             (4,7,"고정된 틀에서 벗어나려는 시도가 보여요."),
                                             (7,11,"유연하게 관점을 바꾸며 사고하고 있어요.")]),
                    "Rec":   ("재인식",     [(1,4,"기존 지식을 새 맥락에 연결하기 어려워하고 있어요."),
                                             (4,7,"알던 것을 다르게 볼 줄 알지만 아직 깊이가 붙고 있어요."),
                                             (7,11,"경험과 개념을 잘 연결해 재발견하고 있어요.")]),
                    "Recon": ("재구성",     [(1,4,"개념들을 하나의 구조로 엮는 데 연습이 필요해요."),
                                             (4,7,"개념 간 연결 시도가 늘고 있어요."),
                                             (7,11,"여러 개념을 새로운 구조로 능동적으로 재조합하고 있어요.")]),
                    "Orc":   ("인지 조율",  [(1,4,"사고 흐름을 스스로 조절하는 데 어려움이 있어요."),
                                             (4,7,"흐름은 일정하지만 자기 조절이 아직 수동적이에요."),
                                             (7,11,"인지 자원을 전략적으로 배분하며 사고하고 있어요.")]),
                }
                cols = st.columns(4)
                for col, (key, (label, levels)) in zip(cols, INDICATOR_GUIDE.items()):
                    score = avg.get(key, avg.get(key.capitalize(), 0))
                    comment = next(c for lo, hi, c in levels if lo <= score < hi or score >= 7)
                    delta_color = "normal" if score >= 7 else "inverse"
                    with col:
                        st.metric(label=f"{label} ({key})", value=f"{score:.2f}", delta_color=delta_color)
                        st.caption(comment)

                st.markdown("### 💡 전문가 종합 해설")
                st.write(last_insight)

                # ── 시계열 저장 ──
                save_timeseries(user_id, avg, last_insight)
                st.info(f"💾 {user_id}님의 데이터가 시계열 저장소에 기록되었습니다.")

                pdf_data = create_ensemble_pdf(
                    last_insight, user_id, analysis_time, img_buf, avg
                )
                if pdf_data:
                    st.download_button(
                        label="📄 PDF 리포트 다운로드",
                        data=pdf_data,
                        file_name=f"SKIM_{user_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.error("❌ 모든 회차 분석 실패. 로그 형식 또는 API 상태를 확인하세요.")


# ══════════════════════════════════════════════
# TAB 2: 시계열 대시보드
# ══════════════════════════════════════════════
with tab_dashboard:
    st.subheader("📈 시계열 대시보드")

    dash_id = st.text_input("조회할 학습자 ID", max_chars=10, value="sj", key="dash_id")
    hist_df = load_timeseries(dash_id) if dash_id else None

    if hist_df is None or hist_df.empty:
        st.info("저장된 시계열 데이터가 없습니다. 먼저 분석을 실행하세요.")
    else:
        st.caption(f"총 {len(hist_df)}회 분석 기록 | "
                   f"{hist_df['timestamp'].min().date()} ~ {hist_df['timestamp'].max().date()}")

        # ── 1. 인지 성장 선 그래프 ──
        st.markdown("#### 📉 인지 성장 추이 (Orc · MTI)")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.plot(hist_df["timestamp"], hist_df["Orc"],
                 marker="o", label="Orc", color="#e74c3c")
        ax2.plot(hist_df["timestamp"], hist_df["MTI"],
                 marker="s", label="MTI", color="#3498db", linestyle="--")
        ax2.set_ylim(0, 10)
        ax2.set_xlabel("날짜")
        ax2.set_ylabel("점수")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=30)
        st.pyplot(fig2)
        plt.close(fig2)

        # ── 2. 패턴 발견: 급변 구간 하이라이트 ──
        st.markdown("#### 🔍 패턴 감지 (Orc 기준 급변 구간)")
        if len(hist_df) >= 2:
            hist_df["orc_delta"] = hist_df["Orc"].diff().abs()
            threshold = hist_df["orc_delta"].mean() + hist_df["orc_delta"].std()
            spikes = hist_df[hist_df["orc_delta"] > threshold]

            if spikes.empty:
                st.success("급변 구간 없음 — 안정적인 성장 패턴입니다.")
            else:
                for _, row in spikes.iterrows():
                    direction = "📈 급상승" if hist_df.loc[row.name, "Orc"] > hist_df.loc[row.name - 1, "Orc"] else "📉 급하락"
                    st.warning(
                        f"{direction} | {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                        f"Orc 변화량: ±{row['orc_delta']:.2f} | "
                        f"해설 요약: {row['insight_summary']}"
                    )
        else:
            st.info("패턴 분석은 2회 이상 데이터 필요합니다.")

        # ── 3. 리포트 히스토리 테이블 ──
        st.markdown("#### 🗂️ 분석 히스토리")
        display_cols = ["timestamp", "MTI", "Rec", "Recon", "Orc", "insight_summary"]
        st.dataframe(
            hist_df[display_cols].rename(columns={"timestamp": "분석 일시",
                                                   "insight_summary": "해설 요약"}),
            use_container_width=True
        )

        # CSV 다운로드
        csv_bytes = hist_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ 시계열 데이터 CSV 다운로드",
            data=csv_bytes,
            file_name=f"SKIM_history_{dash_id}.csv",
            mime="text/csv"
        )