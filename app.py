import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import pandas as pd
import re
from datetime import datetime
import pytz
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="소백현 CRP", layout="wide")
st.title("🧠 소백현: CRP 분석 엔진")
st.markdown("3회 교차 검증을 통해 신뢰도를 높이고 인지 분석 리포트를 생성합니다.")
st.divider()

INDICATORS = ['MTI', 'REC', 'RECON', 'ORC']   # GPT 채점 4개
ENSEMBLE_N = 3
SHEET_NAME = "SKIM_Timeseries"
KST        = pytz.timezone("Asia/Seoul")

tab_analyze, tab_dashboard, tab_inflection, tab4 = st.tabs([
    "🔬 분석", "📈 시계열 대시보드", "🔍 변곡점 정밀 분석", "📋 CRP info"
])

with tab4:
    components.html(open("landing.html", encoding="utf-8").read(), height=920)


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────

def normalize_score(v: float) -> float:
    if v > 100:
        v = v / 10
    elif v > 10:
        v = v / 10
    return max(1.0, min(10.0, round(v, 2)))


def parse_response(text: str) -> tuple[list[float], str, str]:
    data_match = re.search(r"\[DATA\](.*?)\[INSIGHT_KO\]", text, re.S | re.I)
    if not data_match:
        raise ValueError("응답에서 [DATA]...[INSIGHT_KO] 구조를 찾을 수 없습니다.")
    raw_vals = re.findall(
        r"(?:MTI|REC|RECON|ORC)\s*:\s*(\d+\.?\d*)",
        data_match.group(1), re.I
    )
    if len(raw_vals) < len(INDICATORS):
        raise ValueError(f"지표 {len(INDICATORS)}개 필요, {len(raw_vals)}개만 파싱됨.")
    scores = [normalize_score(float(v)) for v in raw_vals[:len(INDICATORS)]]

    ko_match = re.search(r"\[INSIGHT_KO\](.*?)\[INSIGHT_EN\]", text, re.S | re.I)
    en_match = re.search(r"\[INSIGHT_EN\](.*?)$",              text, re.S | re.I)
    insight_ko = ko_match.group(1).strip() if ko_match else ""
    insight_en = en_match.group(1).strip() if en_match else ""

    if not insight_ko or not insight_en:
        raise ValueError("INSIGHT_KO 또는 INSIGHT_EN 섹션이 비어 있습니다.")
    return scores, insight_ko, insight_en


# ─────────────────────────────────────────────
# Google Sheets
# ─────────────────────────────────────────────

@st.cache_resource
def get_gsheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    return gspread.authorize(creds)


def get_worksheet(u_id: str):
    gc = get_gsheet_client()
    ss = gc.open(SHEET_NAME)
    try:
        ws = ss.worksheet(u_id)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=u_id, rows=1000, cols=20)
        ws.append_row([
            "timestamp", "MTI", "Rec", "Recon", "Orc", "insight_summary"
        ])
    return ws


def save_timeseries(u_id: str, avg: pd.Series, insight_ko: str) -> None:
    ws = get_worksheet(u_id)
    ws.append_row([
        datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        round(float(avg["MTI"]),   2), round(float(avg["Rec"]),   2),
        round(float(avg["Recon"]), 2), round(float(avg["Orc"]),   2),
        insight_ko[:150] + ("..." if len(insight_ko) > 150 else "")
    ])


def load_timeseries(u_id: str) -> pd.DataFrame | None:
    try:
        ws  = get_worksheet(u_id)
        rec = ws.get_all_records(expected_headers=[
            "timestamp", "MTI", "Rec", "Recon", "Orc", "insight_summary"
        ])
        if not rec:
            return None
        df = pd.DataFrame(rec)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].str.replace(" KST", "", regex=False)
        )
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None


# ─────────────────────────────────────────────
# PDF 생성
# ─────────────────────────────────────────────

def _ascii_bar(score: float, width: int = 20) -> str:
    filled = round(score / 10 * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {score:.2f}"


def create_ensemble_pdf(insight_en: str, u_id: str, time: str,
                        avg: pd.Series) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Sobaekhyeon CRP Cognitive Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"ID: {u_id}   |   Date: {time}   |   Ensemble N={ENSEMBLE_N}",
             ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "[ Core Indicators ]", ln=True)
    pdf.set_font("Courier", "", 10)
    for k in ["MTI", "Rec", "Recon", "Orc"]:
        if k in avg:
            pdf.cell(0, 7, f"  {k:<6} {_ascii_bar(avg[k])}", ln=True)
    pdf.ln(4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "[ Expert Insight ]", ln=True)
    pdf.set_font("Helvetica", "", 10)
    safe_text = insight_en.encode("latin1", errors="ignore").decode("latin1")
    pdf.multi_cell(0, 6, txt=safe_text)
    pdf.ln(4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 10, "Sobaekhyeon CRP  |  v5.0  |  Confidential",
             ln=True, align="C")

    pdf_out = pdf.output(dest="S")
    if isinstance(pdf_out, bytes):
        return pdf_out
    elif isinstance(pdf_out, bytearray):
        return bytes(pdf_out)
    else:
        return pdf_out.encode("latin1")


# ─────────────────────────────────────────────
# 학습자용 지표 해석
# ─────────────────────────────────────────────

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


def get_indicator_comment(key: str, score: float) -> str:
    for lo, hi, comment in INDICATOR_GUIDE[key][1]:
        if lo <= score < hi:
            return comment
    return INDICATOR_GUIDE[key][1][-1][2]


# ─────────────────────────────────────────────
# TAB 1: 분석
# ─────────────────────────────────────────────
with tab_analyze:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("👤 학습자 정보")
        user_id   = st.text_input("학습자 ID 입력 후 기억하세요", max_chars=10, value="id")
        log_input = st.text_area("분석할 [Conversation Log] 입력", height=400)
        analyze_btn = st.button(
            f"🚀 분석 시작 ({ENSEMBLE_N}회 교차 검증)",
            use_container_width=True
        )

    if analyze_btn and user_id and log_input:
        with col2:
            analysis_time    = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
            scores_list:     list[list[float]] = []
            errors:          list[str]         = []
            last_insight_ko: str               = ""
            last_insight_en: str               = ""

            progress_bar = st.progress(0)
            status_text  = st.empty()

            for i in range(ENSEMBLE_N):
                status_text.text(f"분석 중... ({i+1}/{ENSEMBLE_N}회)")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a CRP (Cognitive Re-configuration Protocol) analysis system.\n"
                                    "Score each indicator from 1 to 10 based on the rubric below.\n"
                                    "Always use ONLY this format:\n\n"
                                    "[DATA]\n"
                                    "MTI: <1-10>\nREC: <1-10>\nRECON: <1-10>\nORC: <1-10>\n"
                                    "[INSIGHT_KO]\n"
                                    "<학습자의 인지 행동을 관찰하여 해설하라. "
                                    "로그의 내용이 아닌 학습자가 어떻게 사고했는지를 중심으로 써라.>\n"
                                    "[INSIGHT_EN]\n"
                                    "<Same insight in English only. "
                                    "Focus on how the learner thought, not what the log contains.>\n\n"

                                    "--- SCORING RUBRIC ---\n\n"

                                    # MTI (사고 전환 · Meta-cognitive Tension Index)
                                    # 학습자가 기존 논리를 스스로 수정하거나 관점을 전환하는
                                    # 메타인지적 역동성을 측정한다.
                                    # 자기 수정 발화 패턴("아니다","바꿔야겠다","틀렸다" 등),
                                    # 충돌 후 전환까지 소요된 대화 턴 수, 성찰 발화의 심층도가 기준이다.
                                    "MTI (Meta-cognitive Tension Index):\n"
                                    "  1-3: No self-correction. Accepts all AI outputs passively. "
                                    "No negation phrases or strategy revision detected.\n"
                                    "  4-6: Occasionally uses negation or revision phrases "
                                    "(e.g. 'that is wrong', 'I need to change this'). "
                                    "Some strategic adjustment after cognitive conflict, "
                                    "but transition takes many turns.\n"
                                    "  7-10: Frequent self-negation and rapid perspective shifts "
                                    "(within 1-2 turns after conflict). Actively restructures logic. "
                                    "Deep reflective engagement with explicit reasoning about the change.\n\n"

                                    # REC (재인식 · Recognition)
                                    # 대화 흐름 속에서 핵심 패턴과 원리를 식별하는 능력을 측정한다.
                                    # 단순 키워드 반복이 아닌 개념 간 관계 파악, 구조적 서술,
                                    # 정보의 중요도를 핵심과 주변부로 분류하는 능력이 기준이다.
                                    "REC (Recognition):\n"
                                    "  1-3: Simple keyword repetition. Cannot distinguish core concepts "
                                    "from peripheral details. No relational understanding.\n"
                                    "  4-6: Identifies some connections between concepts. "
                                    "Partial pattern recognition. Begins to separate important from "
                                    "less important information but inconsistently.\n"
                                    "  7-10: Articulates structural relationships clearly. "
                                    "Accurately classifies information by importance. "
                                    "Extracts core principles and maps concept interdependencies independently.\n\n"

                                    # RECON (재구성 · Reconfiguration)
                                    # 기존 지식을 해체하고 새로운 구조로 통합·재조립하는 역량을 측정한다.
                                    # AI 제안 수용 대비 수정·재설계 비율, 독창적 대안 제시 여부,
                                    # 초기 설계안과 최종 산출물 사이의 구조적 차이가 기준이다.
                                    # Piaget의 조절(Accommodation)이 실제로 일어나고 있는지를 반영한다.
                                    "RECON (Reconfiguration):\n"
                                    "  1-3: Accepts AI output as-is. No restructuring or alternative proposals. "
                                    "Final output is structurally identical to initial framing.\n"
                                    "  4-6: Partially modifies AI suggestions. Some novel connections attempted. "
                                    "Moderate structural difference between initial and final design.\n"
                                    "  7-10: Independently redesigns solutions beyond AI input (Piaget Accommodation). "
                                    "Creates new conceptual structures. Final output shows clear structural "
                                    "transformation from initial framing. Original alternatives proposed.\n\n"

                                    # ORC (조율 · Orchestration)
                                    # AI와 자원을 전략적으로 운용하는 지휘 능력을 측정한다.
                                    # 프롬프트 정교함, 작업 분해 및 순서화, AI 출력 검증 행동,
                                    # 자원 배분의 타이밍이 기준이다.
                                    # Orc가 안정적으로 유지되면 기저 조율 능력 확보의 신호로 해석한다.
                                    "ORC (Orchestration):\n"
                                    "  1-3: Vague, unstructured prompts. No task decomposition. "
                                    "Passive AI use with no verification. Poor timing of resource allocation.\n"
                                    "  4-6: Some structure in prompts with partial conditions. "
                                    "Partial task sequencing. Occasionally reviews AI outputs. "
                                    "Resource allocation timing is inconsistent.\n"
                                    "  7-10: Precise, conditional prompts with clear constraints. "
                                    "Systematic task decomposition and sequencing. "
                                    "Actively verifies and redirects AI outputs at appropriate timing. "
                                    "Stable orchestration across sessions signals consolidated baseline capability.\n"
                                )
                            },
                            {"role": "user", "content": f"ID: {user_id}\nLog:\n{log_input}"}
                        ],
                        temperature=0.1
                    )
                    scores, insight_ko, insight_en = parse_response(
                        response.choices[0].message.content
                    )
                    scores_list.append(scores)
                    last_insight_ko = insight_ko
                    last_insight_en = insight_en

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
                df  = pd.DataFrame(scores_list, columns=["MTI","Rec","Recon","Orc"])
                avg = df.mean()
                std = df.std() if len(scores_list) > 1 else None

                st.success(f"✅ {len(scores_list)}/{ENSEMBLE_N}회 분석 완료")

                if std is not None:
                    st.caption(
                        "📊 앙상블 표준편차 — "
                        + " | ".join(f"{c}: ±{std[c]:.2f}" for c in ["MTI","Rec","Recon","Orc"])
                    )

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(
                    ["MTI","Rec","Recon","Orc"],
                    [avg["MTI"], avg["Rec"], avg["Recon"], avg["Orc"]],
                    color=["#9b59b6","#3498db","#e74c3c","#2ecc71"],
                    yerr=[std["MTI"], std["Rec"], std["Recon"], std["Orc"]] if std is not None else None,
                    capsize=5
                )
                ax.set_ylim(0, 10)
                ax.set_title("Ensemble Average (with std)")
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format="png", bbox_inches="tight")
                plt.close(fig)
                img_buf.seek(0)
                st.image(img_buf)

                st.markdown("### 📌 나의 인지 상태")
                cols = st.columns(4)
                for col, key in zip(cols, ["MTI","Rec","Recon","Orc"]):
                    label   = INDICATOR_GUIDE[key][0]
                    score   = avg[key]
                    comment = get_indicator_comment(key, score)
                    with col:
                        st.metric(label=f"{label} ({key})", value=f"{score:.2f}")
                        st.caption(comment)

                st.markdown("### 💡 종합 해설")
                st.write(last_insight_ko)

                save_timeseries(user_id, avg, last_insight_ko)
                st.info(f"💾 {user_id}님의 데이터가 시계열 저장소에 기록되었습니다.")

                pdf_data = create_ensemble_pdf(
                    last_insight_en, user_id, analysis_time, avg
                )
                if pdf_data:
                    st.download_button(
                        label="📄 PDF 요약 다운로드",
                        data=pdf_data,
                        file_name=f"CRP_{user_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.error("❌ 모든 회차 분석 실패. 로그 형식 또는 API 상태를 확인하세요.")


# ─────────────────────────────────────────────
# TAB 2: 시계열 대시보드
# ─────────────────────────────────────────────
with tab_dashboard:
    st.subheader("📈 시계열 대시보드")

    dash_id = st.text_input("조회할 학습자 ID", max_chars=10, value="sj", key="dash_id")
    hist_df = load_timeseries(dash_id) if dash_id else None

    if hist_df is None or hist_df.empty:
        st.info("저장된 시계열 데이터가 없습니다. 먼저 분석을 실행하세요.")
    else:
        st.caption(
            f"총 {len(hist_df)}회 분석 기록 | "
            f"{hist_df['timestamp'].min().date()} ~ {hist_df['timestamp'].max().date()}"
        )

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

        st.markdown("#### 🔍 패턴 감지 (Orc 기준 급변 구간)")
        if len(hist_df) >= 2:
            hist_df["orc_delta"] = hist_df["Orc"].diff().abs()
            threshold = hist_df["orc_delta"].mean() + hist_df["orc_delta"].std()
            spikes    = hist_df[hist_df["orc_delta"] > threshold]

            if spikes.empty:
                st.success("급변 구간 없음 — 안정적인 성장 패턴입니다.")
            else:
                for _, row in spikes.iterrows():
                    prev_orc  = hist_df.loc[row.name - 1, "Orc"] if row.name > 0 else row["Orc"]
                    direction = "📈 급상승" if row["Orc"] > prev_orc else "📉 급하락"
                    st.warning(
                        f"{direction} | {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                        f"Orc 변화량: ±{row['orc_delta']:.2f} | "
                        f"해설 요약: {row['insight_summary']}"
                    )
        else:
            st.info("패턴 분석은 2회 이상 데이터 필요합니다.")

        st.markdown("#### 🗂️ 분석 히스토리")
        display_cols = ["timestamp","MTI","Rec","Recon","Orc","insight_summary"]
        st.dataframe(
            hist_df[display_cols].rename(columns={
                "timestamp": "분석 일시", "insight_summary": "해설 요약"
            }),
            use_container_width=True
        )

        csv_bytes = hist_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ 시계열 데이터 CSV 다운로드",
            data=csv_bytes,
            file_name=f"CRP_history_{dash_id}.csv",
            mime="text/csv"
        )


# ─────────────────────────────────────────────
# TAB 3: 변곡점 정밀 분석 (Grokking — 시스템 계산)
# ─────────────────────────────────────────────
with tab_inflection:
    st.subheader("🔍 Cognitive Inflection Point Analysis")
    inflect_id  = st.text_input("분석할 학습자 ID", value="sj", key="inflect_id")
    inflect_btn = st.button("🔍 변곡점 분석 조회", key="inflect_btn", use_container_width=True)

    if inflect_btn and inflect_id:
        data = load_timeseries(inflect_id)
        if data is not None and len(data) >= 2:
            core_metrics = ["MTI", "Rec", "Recon", "Orc"]
            colors       = ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"]

            fig3, ax3 = plt.subplots(figsize=(11, 5))
            for metric, color in zip(core_metrics, colors):
                ax3.plot(data["timestamp"], data[metric],
                         marker="o", label=metric, color=color, alpha=0.6, linewidth=1.5)

            # Grokking: velocity 기반 시스템 계산
            data["momentum"]     = data[core_metrics].mean(axis=1)
            data["velocity"]     = data["momentum"].diff()
            data["acceleration"] = data["velocity"].diff()

            v_std     = data["velocity"].std()
            threshold = v_std * 1.2 if v_std > 0 else 0.5
            inflections = data[data["velocity"].abs() > threshold]

            for _, row in inflections.iterrows():
                if row["velocity"] > 0:
                    label_txt, line_color = "JUMP",  "#2ecc71"
                else:
                    label_txt, line_color = "PIVOT", "#95a5a6"
                ax3.axvline(x=row["timestamp"], color=line_color, linestyle="--", alpha=0.4)
                ax3.text(row["timestamp"], 10.3, label_txt,
                         ha="center", fontsize=8, fontweight="bold", color=line_color)

            ax3.set_ylim(0, 11)
            ax3.set_ylabel("Cognitive Score")
            ax3.set_title(f"Cognitive Trajectory: {inflect_id}", fontsize=12, pad=20)
            ax3.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
            ax3.grid(axis="y", linestyle=":", alpha=0.3)
            ax3.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            st.markdown("#### 📊 Cognitive Momentum")
            fig4, ax4 = plt.subplots(figsize=(11, 2.5))
            ax4.fill_between(data["timestamp"], data["momentum"], alpha=0.3, color="#3498db")
            ax4.plot(data["timestamp"], data["momentum"], color="#3498db", linewidth=2)
            ax4.set_ylim(0, 10)
            ax4.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            st.markdown("#### 🚩 Analysis Details")
            if not inflections.empty:
                for _, row in inflections.iterrows():
                    tag = "Growth" if row["velocity"] > 0 else "Adjustment"
                    with st.expander(
                        f"[{tag}] {row['timestamp'].strftime('%m/%d %H:%M')} "
                        f"| Velocity: {row['velocity']:+.2f}"
                    ):
                        col_a, col_b = st.columns([1, 4])
                        col_a.metric("Velocity",     f"{row['velocity']:+.2f}")
                        col_a.metric("Acceleration", f"{row['acceleration']:+.2f}"
                                     if pd.notna(row['acceleration']) else "N/A")
                        col_b.write(f"**Insight Summary:** {row['insight_summary']}")

                    with st.spinner("AI 심층 분석 중..."):
                        try:
                            session_series = "\n".join([
                                f"세션 {i+1} ({r['timestamp'].strftime('%m/%d %H:%M')}): "
                                f"MTI {r['MTI']:.2f} / Rec {r['Rec']:.2f} / "
                                f"Recon {r['Recon']:.2f} / Orc {r['Orc']:.2f}"
                                for i, r in data.iterrows()
                            ])
                            prompt = f"""
아래는 학습자 {inflect_id}의 전체 세션 인지 수치 흐름이다.
변곡점 위치를 기준으로, 전후 수치 변화를 학습자의 '인지 전략' 관점에서 3~5문장으로 해설하라.
대학생이 이해할 수 있는 한국어로 써라.
데이터가 {len(data)}회차임을 감안하여 단정보다 가능성으로 표현하라.

[전체 수치 흐름]
{session_series}

변곡점: 세션 {row.name+1} ({row['timestamp'].strftime('%m/%d %H:%M')})
유형: {tag} / Velocity: {row['velocity']:+.2f}
"""
                            res = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=500
                            )
                            st.markdown(f"**🧠 AI 심층 분석**\n\n{res.choices[0].message.content.strip()}")
                            st.divider()
                        except Exception as e:
                            st.warning(f"분석 오류: {e}")
            else:
                st.info("No significant inflection points detected in current interval.")

        elif data is not None and len(data) < 2:
            st.info(f"Minimum 2 sessions required. (Current: {len(data)})")
        else:
            st.info("No data found. Please run analysis first.")