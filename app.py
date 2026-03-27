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

INDICATORS  = ['MTI', 'REC', 'RECON', 'ORC']
ENSEMBLE_N  = 3
SHEET_NAME  = "SKIM_Timeseries"
KST         = pytz.timezone("Asia/Seoul")
DIVERGENCE_THRESHOLD = 2.0  # τ_d

# ─── Tab 3개로 축소 ───
tab_analyze, tab_inflection, tab_info = st.tabs([
    "🔬 분석", "🔍 변곡점 정밀 분석", "📋 CRP info"
])

with tab_info:
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

    ko_match   = re.search(r"\[INSIGHT_KO\](.*?)\[INSIGHT_EN\]", text, re.S | re.I)
    en_match   = re.search(r"\[INSIGHT_EN\](.*?)$",              text, re.S | re.I)
    insight_ko = ko_match.group(1).strip() if ko_match else ""
    insight_en = en_match.group(1).strip() if en_match else ""

    if not insight_ko or not insight_en:
        raise ValueError("INSIGHT_KO 또는 INSIGHT_EN 섹션이 비어 있습니다.")
    return scores, insight_ko, insight_en


# ─────────────────────────────────────────────
# MTI 레이어 불일치 자기진단
# ─────────────────────────────────────────────

def diagnose_layer_divergence(scores_list: list[list[float]]) -> dict | None:
    if len(scores_list) < 3:
        return None
    l1, l2, l3 = scores_list[0][0], scores_list[1][0], scores_list[2][0]
    divergence  = max(l1, l2, l3) - min(l1, l2, l3)

    if divergence <= DIVERGENCE_THRESHOLD:
        return {"divergence": divergence, "flag": False,
                "type": "-", "message": "레이어 간 편차 정상 범위"}

    if l1 < l2 and l1 < l3:
        diag_type = "L1 낮음 · L2 높음"
        message   = "언어 패턴 풀 보완 필요 — 루브릭 Class 기준 재검토 권장"
    elif l1 > l2 and l3 < l2:
        diag_type = "L1 높음 · L3 낮음"
        message   = "명시적 수정은 있으나 구조적 변화 없음 — 표면적 메타인지 가능성"
    else:
        diag_type = "L2 낮음 · L3 높음"
        message   = "암묵적 패턴은 있으나 LLM 판단 미포착 — 루브릭 보정 필요"

    return {"divergence": divergence, "flag": True,
            "type": diag_type, "message": message}


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
        ws.append_row(["timestamp", "MTI", "Rec", "Recon", "Orc", "insight_summary"])
    return ws


def save_timeseries(u_id: str, avg: pd.Series, insight_ko: str) -> None:
    ws = get_worksheet(u_id)
    ws.append_row([
        datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        round(float(avg["MTI"]),   2), round(float(avg["Rec"]),   2),
        round(float(avg["Recon"]), 2), round(float(avg["Orc"]),   2),
        insight_ko[:350] + ("..." if len(insight_ko) > 350 else "")
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


def _safe(text: str) -> str:
    return text.encode("latin1", errors="ignore").decode("latin1")


def create_ensemble_pdf(insight_en: str, u_id: str, time: str,
                        avg: pd.Series) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Sobaekhyeon CRP Cognitive Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5,
             f"ID: {_safe(u_id)}   |   Date: {_safe(time)}   |   Ensemble N={ENSEMBLE_N}",
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
    pdf.multi_cell(0, 6, txt=_safe(insight_en))
    pdf.ln(4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 10, "Sobaekhyeon CRP  |  v6.1  |  Confidential",
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
    "MTI": ("사고 전환", [
        (1,  4,  "AI 출력을 그대로 수용하고 있어요. 사고의 방향이 외부에 의해 결정되고 있습니다."),
        (4,  7,  "기존 논리를 수정하거나 관점을 바꾸려는 시도가 보여요."),
        (7,  10, "스스로 사고의 방향을 바꾸고 있어요."),
        (10, 11, "문제 자체를 재정의하는 수준의 사고 전환이 감지됩니다. (Class A+)")
    ]),
    "Rec": ("재인식", [
        (1, 4,  "기존 지식을 새 맥락에 연결하기 어려워하고 있어요."),
        (4, 7,  "알던 것을 다르게 볼 줄 알지만 아직 깊이가 붙고 있어요."),
        (7, 11, "경험과 개념을 잘 연결해 재발견하고 있어요.")
    ]),
    "Recon": ("재구성", [
        (1, 4,  "개념들을 하나의 구조로 엮는 데 연습이 필요해요."),
        (4, 7,  "개념 간 연결 시도가 늘고 있어요."),
        (7, 11, "여러 개념을 새로운 구조로 능동적으로 재조합하고 있어요.")
    ]),
    "Orc": ("인지 조율", [
        (1, 4,  "사고 흐름을 스스로 조절하는 데 어려움이 있어요."),
        (4, 7,  "흐름은 일정하지만 자기 조절이 아직 수동적이에요."),
        (7, 11, "인지 자원을 전략적으로 배분하며 사고하고 있어요.")
    ]),
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
        user_id     = st.text_input("학습자 ID", max_chars=10, value="id")
        log_input   = st.text_area("분석할 [Conversation Log] 입력", height=500)
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

                                    "MTI (Meta-cognitive Tension Index):\n"
                                    "  Class hierarchy (Veenman 2011): A+ > A > B > C.\n"
                                    "  Upper class structurally subsumes lower. Apply highest class only.\n\n"
                                    "  1-3 [Class C or none]: No self-correction. "
                                    "Accepts all AI outputs passively. "
                                    "No negation or strategy revision detected.\n"
                                    "  4-6 [Class B-C]: Occasionally revises approach or reflects. "
                                    "Uses negation or revision phrases but transition takes many turns.\n"
                                    "  7-9 [Class A]: Frequent self-negation "
                                    "(e.g. 'I was wrong', 'I need to change this'). "
                                    "Rapid perspective shifts within 1-2 turns after conflict. "
                                    "Actively restructures logic with explicit reasoning.\n"
                                    "  9-10 [Class A+]: Redefines the problem or task premise itself. "
                                    "Challenges the frame of the question "
                                    "(e.g. 'This question is wrongly posed', "
                                    "'I need to redefine the problem'). "
                                    "Why-layer thinking: questions the validity of the task itself, "
                                    "not just the answer. Must cite exact learner utterance.\n\n"

                                    "REC (Recognition):\n"
                                    "  1-3: Simple keyword repetition. Cannot distinguish core concepts "
                                    "from peripheral details. No relational understanding.\n"
                                    "  4-6: Identifies some connections between concepts. "
                                    "Partial pattern recognition. Begins to separate important from "
                                    "less important information but inconsistently.\n"
                                    "  7-10: Articulates structural relationships clearly. "
                                    "Accurately classifies information by importance. "
                                    "Extracts core principles and maps concept interdependencies independently.\n\n"

                                    "RECON (Reconfiguration):\n"
                                    "  1-3: Accepts AI output as-is. No restructuring or alternative proposals. "
                                    "Final output is structurally identical to initial framing.\n"
                                    "  4-6: Partially modifies AI suggestions. Some novel connections attempted. "
                                    "Moderate structural difference between initial and final design.\n"
                                    "  7-10: Independently redesigns solutions beyond AI input (Piaget Accommodation). "
                                    "Creates new conceptual structures. Final output shows clear structural "
                                    "transformation from initial framing. Original alternatives proposed.\n\n"

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

                diag = diagnose_layer_divergence(scores_list)
                if diag:
                    if diag["flag"]:
                        st.warning(
                            f"⚠️ **MTI 레이어 불일치 감지** | "
                            f"편차: {diag['divergence']:.2f} (τ_d={DIVERGENCE_THRESHOLD})\n\n"
                            f"**유형:** {diag['type']}\n\n"
                            f"**진단:** {diag['message']}"
                        )
                    else:
                        st.success(
                            f"✅ MTI 레이어 편차 정상 ({diag['divergence']:.2f} ≤ {DIVERGENCE_THRESHOLD})"
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
# TAB 2: 변곡점 정밀 분석 + 히스토리 테이블 + CSV
# ─────────────────────────────────────────────
with tab_inflection:
    st.subheader("🔍 Cognitive Inflection Point Analysis")
    inflect_id  = st.text_input("분석할 학습자 ID", value="sj", key="inflect_id")
    inflect_btn = st.button("🔍 변곡점 분석 조회", key="inflect_btn", use_container_width=True)

    if inflect_btn and inflect_id:
        data = load_timeseries(inflect_id)

        if data is None or data.empty:
            st.info("No data found. Please run analysis first.")

        elif len(data) < 2:
            st.info(f"Minimum 2 sessions required. (Current: {len(data)})")

        else:
            core_metrics = ["MTI", "Rec", "Recon", "Orc"]
            colors       = ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"]

            # ── 4지표 시계열 그래프 ──
            fig3, ax3 = plt.subplots(figsize=(11, 5))
            for metric, color in zip(core_metrics, colors):
                ax3.plot(data["timestamp"], data[metric],
                         marker="o", label=metric, color=color, alpha=0.6, linewidth=1.5)

            data["momentum"]     = data[core_metrics].mean(axis=1)
            data["velocity"]     = data["momentum"].diff()
            data["acceleration"] = data["velocity"].diff()

            v_std       = data["velocity"].std()
            threshold   = v_std * 1.2 if v_std > 0 else 0.5
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

            # ── Cognitive Momentum ──
            st.markdown("#### 📊 Cognitive Momentum")
            fig4, ax4 = plt.subplots(figsize=(11, 2.5))
            ax4.fill_between(data["timestamp"], data["momentum"], alpha=0.3, color="#3498db")
            ax4.plot(data["timestamp"], data["momentum"], color="#3498db", linewidth=2)
            ax4.set_ylim(0, 10)
            ax4.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            # ── 변곡점 상세 ──
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
                        col_a.metric("Acceleration",
                                     f"{row['acceleration']:+.2f}"
                                     if pd.notna(row["acceleration"]) else "N/A")
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
                            st.markdown(
                                f"**🧠 AI 심층 분석**\n\n"
                                f"{res.choices[0].message.content.strip()}"
                            )
                            st.divider()
                        except Exception as e:
                            st.warning(f"분석 오류: {e}")
            else:
                st.info("No significant inflection points detected in current interval.")

            # ── 히스토리 테이블 + CSV (Tab2에서 이전) ──
            st.markdown("#### 🗂️ 분석 히스토리")
            st.caption(
                f"총 {len(data)}회 분석 기록 | "
                f"{data['timestamp'].min().date()} ~ {data['timestamp'].max().date()}"
            )
            display_cols = ["timestamp","MTI","Rec","Recon","Orc","insight_summary"]
            st.dataframe(
                data[display_cols].rename(columns={
                    "timestamp": "분석 일시", "insight_summary": "해설 요약"
                }),
                use_container_width=True
            )
            csv_bytes = data.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="⬇️ 시계열 데이터 CSV 다운로드",
                data=csv_bytes,
                file_name=f"CRP_history_{inflect_id}.csv",
                mime="text/csv"
            )