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
# [v6.3 추가] 변곡점 delta 사전 계산
# ─────────────────────────────────────────────

def compute_pivot_differential(data: pd.DataFrame, pivot_row: pd.Series) -> dict:
    """
    변곡점 행을 기준으로 delta·낙폭 순위·MTI trait 안정 여부를 사전 계산.
    LLM에게 수치 해석을 위임하지 않고 구조화된 근거로 입력한다.
    """
    metrics = ["MTI", "Rec", "Recon", "Orc"]

    prior = data[data["timestamp"] < pivot_row["timestamp"]]
    if prior.empty:
        baseline = {m: float(pivot_row[m]) for m in metrics}
    else:
        # velocity = momentum[t] - momentum[t-1] (직전 세션 기준)
        # delta 기준도 직전 세션으로 통일해야 방향이 일치함
        # peak 기준을 쓰면 JUMP 직전에 고점이 있을 때 delta가 0/-로 역전됨
        prev_row = prior.iloc[-1]
        baseline = {m: float(prev_row[m]) for m in metrics}

    current = {m: float(pivot_row[m]) for m in metrics}
    deltas  = {m: round(current[m] - baseline[m], 2) for m in metrics}

    sorted_by_drop = sorted(metrics, key=lambda m: deltas[m])
    delta_ranks    = {m: sorted_by_drop.index(m) + 1 for m in metrics}
    avg_delta      = round(sum(deltas.values()) / len(metrics), 2)
    mti_trait_stable = deltas["MTI"] > avg_delta

    mti_v, recon_v, orc_v, rec_v = (
        current["MTI"], current["Recon"], current["Orc"], current["Rec"]
    )
    if mti_v >= 6.5 and recon_v < 6.0:
        class_hint = "확산집중형"
    elif orc_v >= 7.0 and rec_v >= 7.0 and mti_v < 6.5:
        class_hint = "실행편중형"
    elif mti_trait_stable and deltas["Recon"] < -2.0:
        class_hint = "인지포화형"
    elif max(current.values()) - min(current.values()) < 1.5:
        class_hint = "균형성장형"
    else:
        class_hint = "판단보류"

    return {
        "current":          current,
        "baseline":         baseline,
        "deltas":           deltas,
        "delta_ranks":      delta_ranks,
        "avg_delta":        avg_delta,
        "mti_trait_stable": mti_trait_stable,
        "max_drop_metric":  sorted_by_drop[0],
        "min_drop_metric":  sorted_by_drop[-1],
        "class_hint":       class_hint,
    }


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
        try:
            ws = ss.add_worksheet(title=u_id, rows=1000, cols=20)
            ws.append_row(["timestamp", "MTI", "Rec", "Recon", "Orc", "insight_summary"])
        except Exception:
            # 동시 생성 또는 이미 존재하는 경우 재시도
            ws = ss.worksheet(u_id)
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
            # 최소 임계값 1.0 하한선 — 세션 수 적거나 점수 밀집 시 과탐지 방지
            threshold   = max(v_std * 1.2 if v_std > 0 else 0.5, 1.0)
            # 최소 5세션 미만이면 변곡점 탐지 보류
            if len(data) < 5:
                st.info(
                    f"📊 현재 {len(data)}회 세션 — 변곡점 탐지는 5회 이상 누적 후 신뢰도가 높아집니다. "
                    "데이터를 계속 축적하세요."
                )
                inflections = data.iloc[0:0]  # 빈 DataFrame
            else:
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
                            # ── [v6.3] delta 사전 계산 ──────────────────────
                            diff = compute_pivot_differential(data, row)
                            is_pivot = row["velocity"] < 0  # PIVOT=낙폭분석 / JUMP=상승분석

                            # PIVOT: 낙폭 순위 라벨 / JUMP: 상승폭 순위 라벨
                            if is_pivot:
                                extreme_label = {1: "(↓최대)", 4: "(↓최소)"}
                                extreme_metric_label = ("낙폭 최대", "낙폭 최소")
                                direction_word = "낙폭"
                                mti_stable_label = (
                                    "YES — MTI 낙폭이 평균보다 작음 (trait 안정)"
                                    if diff["mti_trait_stable"]
                                    else "NO — MTI도 평균 이상 하락"
                                )
                            else:
                                extreme_label = {4: "(↑최대)", 1: "(↑최소)"}
                                extreme_metric_label = ("상승폭 최대", "상승폭 최소")
                                direction_word = "변화폭"
                                mti_stable_label = (
                                    "YES — MTI 상승폭이 평균 이상"
                                    if not diff["mti_trait_stable"]
                                    else "YES — MTI 상승폭이 평균보다 작음"
                                )

                            delta_str = "  ".join(
                                f"{m}: {diff['deltas'][m]:+.2f}"
                                + extreme_label.get(diff["delta_ranks"][m], "")
                                for m in ["MTI", "Rec", "Recon", "Orc"]
                            )
                            baseline_str = "  ".join(
                                f"{m}: {diff['baseline'][m]:.2f}"
                                for m in ["MTI", "Rec", "Recon", "Orc"]
                            )
                            current_str = "  ".join(
                                f"{m}: {diff['current'][m]:.2f}"
                                for m in ["MTI", "Rec", "Recon", "Orc"]
                            )

                            # ── [v6.4] 지표 의미 레이블 매핑 ─────────────
                            METRIC_LABELS = {
                                "MTI":   "스스로 틀렸음을 인식하고 사고를 전환하는 능력",
                                "Rec":   "새로운 맥락에서 개념을 재인식하는 능력",
                                "Recon": "개념들을 새 구조로 재조립하는 능력",
                                "Orc":   "사고 흐름을 전략적으로 조율하는 능력",
                            }

                            max_m = diff["max_drop_metric"]
                            min_m = diff["min_drop_metric"]

                            # ── [v6.4] 시스템 프롬프트 — 궤적 언어 6개 규칙 ─
                            system_prompt = (
                                "당신은 CRP(Cognitive Re-configuration Protocol) "
                                "인지 궤적 분석기입니다. "
                                "다음 규칙을 반드시 지키십시오.\n\n"

                                "규칙 1 — 궤적 서술: "
                                "출력은 현재 상태 판정이 아닌 변화 과정 서술이어야 합니다.\n"
                                "  ✗ '재조립 능력이 저하된 상태입니다'\n"
                                "  ✓ '재조립 능력이 가장 빠르게 하락하며 다른 능력과 격차가 벌어졌습니다'\n\n"

                                "규칙 2 — 비대칭 분석: "
                                "지표별 변화폭 차이가 분석 핵심입니다. "
                                "'모든 능력이 동일하게 변화했다'는 서술은 분석이 아닙니다.\n\n"

                                "규칙 3 — 사고 전환 능력의 안정성: "
                                "Veenman(2006)/Sweller(1988)에 따라 인지 부하·피로는 "
                                "실행 수준 능력(재인식·재조립·조율)을 먼저 떨어뜨리고, "
                                "메타 수준인 '스스로 틀렸음을 인식하는 능력'은 상대적으로 유지됩니다. "
                                "이 패턴이 데이터에 나타나면 명시적으로 서술하십시오.\n\n"

                                "규칙 4 — 분류 필수: "
                                "확산집중형 / 균형성장형 / 실행편중형 / 인지포화형 "
                                "중 하나를 반드시 출력하십시오.\n\n"

                                "규칙 5 — 지표 약어 금지: "
                                "MTI, Rec, Recon, Orc를 출력에 직접 쓰지 마십시오. "
                                "각 능력의 의미를 동사·명사로 풀어 쓰십시오.\n\n"

                                "규칙 6 — 헤징 금지: "
                                "'가능성이 있습니다' 형태의 미확정 서술을 사용하지 마십시오. "
                                "출력 구조: [변화 서술] → [패턴 판단] → [분류] → [다음 방향]"
                            )

                            # ── [v6.4] JUMP/PIVOT 방향에 맞게 주요 지표 선택 ─
                            # PIVOT: 가장 많이 떨어진(sorted[0]) vs 가장 덜 떨어진(sorted[-1])
                            # JUMP:  가장 많이 오른(sorted[-1]) vs 가장 덜 오른(sorted[0])
                            if is_pivot:
                                primary_m   = max_m   # sorted_by_drop[0] = 최대 낙폭
                                secondary_m = min_m   # sorted_by_drop[-1] = 최소 낙폭
                                primary_verb   = "가장 빠르게 하락"
                                secondary_verb = "상대적으로 덜 하락"
                                step2_label = "무엇이 버텼고 무엇이 먼저 꺾였는가"
                                step2_mti_desc = (
                                    f"인지 부하 상황에서도 "
                                    f"{'평균보다 덜 하락하며 안정성을 유지했습니다' if diff['mti_trait_stable'] else '평균 이상으로 하락했습니다'}."
                                )
                            else:
                                primary_m   = min_m   # sorted_by_drop[-1] = 최대 상승폭
                                secondary_m = max_m   # sorted_by_drop[0]  = 최소 상승폭
                                primary_verb   = "가장 크게 상승"
                                secondary_verb = "상대적으로 덜 상승"
                                step2_label = "무엇이 함께 올랐고 무엇이 덜 반응했는가"
                                step2_mti_desc = (
                                    f"다른 능력들이 상승하는 동안 "
                                    f"{'평균보다 덜 상승하며 상대적으로 낮은 반응을 보였습니다' if diff['mti_trait_stable'] else '평균 이상으로 함께 상승했습니다'}."
                                )

                            # ── [v6.4] 유저 프롬프트 — 궤적 중심 STEP ──────
                            prompt = (
                                f"[CRP 인지 궤적 분석 — {inflect_id} / 세션 {row.name+1} / {tag}]\n\n"
                                f"기준점(peak): {baseline_str}\n"
                                f"현재값:       {current_str}\n"
                                f"변화량(delta): {delta_str}\n"
                                f"평균 변화량:  {diff['avg_delta']:+.2f}\n"
                                f"사고 전환 능력 변화: {mti_stable_label}\n"
                                f"사전 분류 힌트: {diff['class_hint']}\n\n"

                                f"STEP 1 — 변화 속도 비교\n"
                                f"  '{METRIC_LABELS[primary_m]}'이(가) {diff['deltas'][primary_m]:+.2f}로 "
                                f"{primary_verb}했고, "
                                f"'{METRIC_LABELS[secondary_m]}'은(는) {diff['deltas'][secondary_m]:+.2f}로 "
                                f"{secondary_verb}했습니다.\n"
                                f"이 두 능력의 변화 속도 차이가 무엇을 나타내는지 1문장으로 서술하십시오.\n\n"

                                f"STEP 2 — {step2_label}\n"
                                f"  '스스로 틀렸음을 인식하고 사고를 전환하는 능력'({diff['deltas']['MTI']:+.2f})이 "
                                f"{step2_mti_desc}\n"
                                f"이 능력이 이번 변화에서 어떤 역할을 했는지 1문장으로 서술하십시오.\n\n"

                                f"STEP 3 — 이 변화 패턴이 나타내는 인지 전환\n"
                                f"  분류 힌트({diff['class_hint']})를 검토하고, "
                                f"STEP 1·2의 변화 흐름 근거로 최종 분류를 확정하십시오.\n\n"

                                f"STEP 4 — 다음 변화를 위한 방향\n"
                                f"  이 변화 흐름이 계속된다면 어떤 방향으로 이어질지, "
                                f"그리고 다음 세션에서 어디서부터 시작하면 좋을지 1문장으로 서술하십시오.\n\n"

                                f"전체 200자 이내로 작성하십시오."
                            )

                            res = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": prompt}
                                ],
                                temperature=0.1,   # 0.3 → 0.1
                                max_tokens=300     # 500 → 300
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