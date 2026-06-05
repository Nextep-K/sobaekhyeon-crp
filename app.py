# =============================================================================
# AIQ 파일럿 — Streamlit 앱
# Version: v9.0  (2026.06)
#
# 핵심 변경 (v8.x 전면 재설계):
#   [연결]  v7.4 방식 완전 복원 — gspread.authorize(creds), @st.cache_resource 없음
#           실제 작동이 증명된 유일한 방식
#   [저장]  단일 시트(responses) — participants/responses 분리 폐기
#           저장 내용 최소화 (LLM 채점 없음, 대화 로그 없음)
#   [측정]  주관식 대화 완전 제거 → 서사형 객관식 2개 시나리오로 대체
#           Q1(QLI) + Q1-followup, Q2-A + Q2-B + Q2-followup
#           선택값 조합으로 점수 직접 산출 — LLM 채점 0회
#   [UX]    API 호출: 대화 진행 0회, 채점 0회
#           저장 실패 지점: 1개 (단일 append_row)
# =============================================================================

import streamlit as st
from openai import OpenAI
import re
from datetime import datetime
import pytz
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
client    = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
SHEET_NAME = "AIQ_Pilot"
KST        = pytz.timezone("Asia/Seoul")

st.set_page_config(
    page_title="AIQ 진단",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# AIQ 20문항 정의 (VF1 확정본)
# ─────────────────────────────────────────────
QUESTIONS = [
    (1,  "AI에게 질문을 보내기 전에 내가 원하는 결과를 먼저 정의한다",        "설계자", False),
    (2,  "AI가 답을 주면 그 흐름에 맞춰 대화를 이어간다",                     "의존",   False),
    (3,  "AI가 새로운 시각을 제시해도 내 기존 생각을 쉽게 바꾸지 않는다",     "상상가", True),
    (4,  "AI의 지시나 제안을 최대한 빠르게 실행에 옮긴다",                     "실행",   False),
    (5,  "AI 대화에서 내가 놓친 전제나 조건을 스스로 되짚어본다",              "설계자", False),
    (6,  "AI가 틀려도 그 답을 기준으로 판단하는 경우가 있다",                  "의존",   False),
    (7,  "AI가 제안한 방향보다 내 아이디어로 대화를 이끌고 싶다",              "상상가", False),
    (8,  "AI 결과물은 내 기준에 맞게 반드시 수정해서 사용한다",                "설계자", True),
    (9,  "AI가 준 내용을 다른 도구나 방법과 결합해 바로 적용한다",             "실행",   False),
    (10, "AI와 대화하면서 예상치 못한 연결을 발견하는 것이 즐겁다",            "상상가", False),
    (11, "AI가 제안한 방식이 내 방식보다 낫다고 느끼면 그냥 따른다",           "의존",   False),
    (12, "AI가 내 질문 의도를 잘못 이해했을 때 즉시 교정 질문을 보낸다",       "설계자", False),
    (13, "AI 결과를 받으면 바로 적용하기보다 한 번 더 확인한다",               "실행",   True),
    (14, "AI와 대화하다 보면 처음 생각지 못한 가능성이 머릿속에 펼쳐진다",     "상상가", False),
    (15, "AI를 쓸 때 내가 원하는 출력 형식을 명시적으로 지정한다",             "설계자", False),
    (16, "AI가 준 단계별 지시를 순서대로 따라가는 것이 효율적이라고 생각한다", "실행",   False),
    (17, "AI와 대화할 때 '만약 ~라면'처럼 가정을 많이 사용한다",              "상상가", False),
    (18, "AI 결과가 마음에 들지 않으면 다시 시도하기보다 그냥 쓴다",           "의존",   True),
    (19, "AI가 준 코드나 템플릿을 즉시 실행해보며 결과를 확인한다",            "실행",   False),
    (20, "AI 없이 스스로 문제를 해결하는 것이 부담스럽다",                     "의존",   False),
]

TYPE_LABELS = {"설계자": "설계자형", "상상가": "상상가형", "실행": "실행형", "의존": "의존형"}

TYPE_AXIS = {
    "설계자형": ("QLI↑", "CRP↑"),
    "상상가형": ("QLI↑", "CRP↓"),
    "실행형":   ("QLI↓", "CRP↑"),
    "의존형":   ("QLI↓", "CRP↓"),
}

TYPE_DESC = {
    "설계자형": "AI를 도구로 부리는 사람. 질문 전에 목적을 정의하고, 결과를 받으면 재조립한다.",
    "상상가형": "질문 수준은 높지만 결과를 재구성하는 데 어려움이 있다. 아이디어가 풍부하고 실행이 약하다.",
    "실행형":   "AI 지시를 빠르게 실행하는 패턴이 우세하다. 구조보다 속도를 선택하는 경향이 있다.",
    "의존형":   "AI 출력에 의존도가 높다. 자기 검증이 낮으며 사고 역량 강화가 필요하다.",
}

# ─────────────────────────────────────────────
# 점수 매트릭스 (선택값 → 점수)
# ─────────────────────────────────────────────
# Q1 + Q1-followup 조합 → QLI 점수 (1~10)
QLI_MATRIX = {
    ("1", "a"): 3, ("1", "b"): 4, ("1", "c"): 5, ("1", "d"): 2,
    ("2", "a"): 5, ("2", "b"): 7, ("2", "c"): 9, ("2", "d"): 3,
    ("3", "a"): 5, ("3", "b"): 9, ("3", "c"): 6, ("3", "d"): 4,
    ("4", "a"): 6, ("4", "b"): 9, ("4", "c"): 7, ("4", "d"): 10,
}

# Q2-B + Q2-followup 조합 → MTI 점수 (1~10)
# Q2-B=1(전환없음) / Q2-B=2,3,4(전환)
MTI_MATRIX = {
    ("1", "a"): 4, ("1", "b"): 5, ("1", "c"): 6, ("1", "d"): 2,
    ("2", "a"): 6, ("2", "b"): 8, ("2", "c"): 10, ("2", "d"): 4,
    ("3", "a"): 6, ("3", "b"): 8, ("3", "c"): 10, ("3", "d"): 4,
    ("4", "a"): 6, ("4", "b"): 8, ("4", "c"): 10, ("4", "d"): 4,
}

def compute_aiq(qli: int, mti: int) -> int:
    """AIQ 지수 산출 — 100 기준, 범위 약 70~150"""
    return max(0, min(200, 100 + (qli + mti - 10) * 5))

# ─────────────────────────────────────────────
# 유형 산출
# ─────────────────────────────────────────────
def compute_type_scores(answers: dict) -> dict:
    scores = {"설계자": 0, "상상가": 0, "실행": 0, "의존": 0}
    for (no, text, typ, reverse) in QUESTIONS:
        val = answers.get(no, 2)
        if reverse:
            val = 5 - val
        scores[typ] += val
    return scores

def compute_top_types(type_scores: dict) -> tuple:
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    return TYPE_LABELS[sorted_types[0][0]], TYPE_LABELS[sorted_types[1][0]]

def validate_name(name: str) -> bool:
    if not name or not name.strip():
        return False
    return bool(re.match(r"^[가-힣A-Za-z\s]{2,20}$", name.strip()))

def validate_birth(birth: str) -> bool:
    if not birth or len(birth) != 8 or not birth.isdigit():
        return False
    try:
        d = datetime.strptime(birth, "%Y%m%d")
        return 1900 <= d.year <= datetime.now().year
    except ValueError:
        return False

# ─────────────────────────────────────────────
# Google Sheets — v7.4 방식 완전 복원
# @st.cache_resource 없음 — 캐시 시 HTTP 세션 손실로 오류 발생
# ─────────────────────────────────────────────
def get_gsheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    return gspread.authorize(creds)


def save_result(name: str, birth: str,
                type1: str, type2: str, type_scores: dict,
                q1: str, q1f: str,
                q2a: str, q2b: str, q2f: str, q2f_text: str,
                qli: int, mti: int, aiq: int,
                answers: dict) -> str | None:
    """
    진단 결과를 단일 시트(responses)에 저장한다.

    연결: v7.4 방식 (gspread.authorize, @cache 없음)
    저장: 단일 시트, 단일 append_row 1회
    채번: timestamp 기반 자동 생성
    """
    try:
        gc = get_gsheet_client()
        ss = gc.open(SHEET_NAME)
        ts     = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
        serial = "AIQ_" + datetime.now(KST).strftime("%m%d_%H%M%S")

        try:
            ws = ss.worksheet("responses")
        except gspread.WorksheetNotFound:
            ws = ss.add_worksheet(title="responses", rows=1, cols=20)
            ws.append_row([
                "serial", "timestamp", "name", "birth",
                "type1", "type2",
                "score_designer", "score_imaginer", "score_executor", "score_follower",
                "q1", "q1f", "q2a", "q2b", "q2f", "q2f_text",
                "QLI", "MTI", "AIQ_index",
                "q_answers"
            ])

        ws.append_row([
            serial, ts, name, birth,
            type1, type2,
            type_scores.get("설계자", 0), type_scores.get("상상가", 0),
            type_scores.get("실행", 0),   type_scores.get("의존", 0),
            q1, q1f, q2a, q2b, q2f, q2f_text,
            qli, mti, aiq,
            str(answers)
        ])
        return serial

    except Exception as e:
        st.warning(f"저장 오류: {e}")
        return None

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 760px; margin: 0 auto; }
    .step-bar { display:flex; align-items:center; padding:.6rem 0;
                margin-bottom:1.5rem; border-bottom:1px solid #e5e7eb; }
    .step-item { display:flex; align-items:center; gap:6px; flex:1; }
    .step-num  { width:22px; height:22px; border-radius:50%;
                 display:flex; align-items:center; justify-content:center;
                 font-size:11px; font-weight:600; }
    .step-on   { background:#EBF4FF; color:#1D6FA8; border:1.5px solid #93C5FD; }
    .step-done { background:#ECFDF5; color:#065F46; border:1.5px solid #6EE7B7; }
    .step-off  { background:#F9FAFB; color:#9CA3AF; border:1px solid #E5E7EB; }
    .step-lbl-on  { font-size:12px; font-weight:600; color:#111827; }
    .step-lbl-off { font-size:12px; color:#9CA3AF; }
    .step-div  { flex:none; width:20px; height:1px; background:#E5E7EB; margin:0 2px; }
    .scn-box   { background:#F0F7FF; border-left:3px solid #3B82F6;
                 border-radius:6px; padding:1rem 1.2rem; margin-bottom:1rem; }
    .scn-title { font-size:11px; font-weight:600; color:#6B7280;
                 text-transform:uppercase; letter-spacing:.06em; margin-bottom:.4rem; }
    .scn-text  { font-size:14px; color:#111827; line-height:1.7; margin:0; }
    .report-header { border-bottom:2px solid #1F3864; padding-bottom:.85rem; margin-bottom:1.5rem; }
    .report-title  { font-size:11px; font-weight:600; color:#1D6FA8;
                     letter-spacing:.15em; text-transform:uppercase; margin:0 0 .25rem; }
    .report-name   { font-size:22px; font-weight:500; color:#111827; margin:0 0 .5rem; }
    .report-meta   { display:flex; gap:1.5rem; font-size:12px; color:#6B7280; flex-wrap:wrap; }
    .report-meta span strong { color:#374151; font-weight:500; margin-right:.25rem; }
    .aiq-hero  { text-align:center; padding:2.5rem 1rem 2rem;
                 background:linear-gradient(180deg,#F0F7FF 0%,#FFFFFF 100%);
                 border-radius:12px; margin:0 0 1.5rem; border:1px solid #DBEAFE; }
    .aiq-label { font-size:13px; font-weight:500; color:#6B7280;
                 letter-spacing:.08em; text-transform:uppercase; margin:0 0 .5rem; }
    .aiq-value { font-size:96px; font-weight:700; color:#1D6FA8;
                 line-height:1; margin:0; letter-spacing:-2px; }
    .aiq-badge-row { margin-top:.75rem; display:flex; justify-content:center;
                     gap:8px; flex-wrap:wrap; }
    .aiq-badge { font-size:11px; font-weight:500; color:#1D6FA8;
                 background:#DBEAFE; padding:3px 10px; border-radius:4px; }
    .section-title { font-size:24px; font-weight:500; color:#111827; margin:1rem 0 .25rem; }
    .axis-tag      { font-size:13px; color:#6B7280; margin:0 0 .75rem; }
    .type-quote    { border-left:3px solid #D1D5DB; padding:0 0 0 1rem;
                     margin:.5rem 0 1.5rem; color:#4B5563; font-size:13px; line-height:1.7; }
    .coord-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin:.75rem 0; }
    .cc     { border:1px solid #E5E7EB; border-radius:6px; padding:8px 10px;
              font-size:12px; background:#F9FAFB; }
    .cc-hl  { background:#EBF4FF; border-color:#93C5FD; }
    .cc-name { font-weight:600; font-size:13px; display:block; color:#111827; }
    .cc-tag  { font-size:11px; color:#6B7280; }
    .cc-hl .cc-name { color:#1D6FA8; }
    .sub-section-title { font-size:13px; font-weight:600; color:#6B7280;
                         margin:0 0 .75rem; letter-spacing:.04em; text-transform:uppercase; }
    .sub-metrics { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin:.75rem 0; }
    .sm      { padding:.5rem .7rem; border:1px solid #E5E7EB; border-radius:6px; background:#F9FAFB; }
    .sm-lbl  { font-size:10px; color:#9CA3AF; margin:0 0 2px; }
    .sm-val  { font-size:14px; font-weight:500; color:#4B5563; margin:0; }
    .second-rank { font-size:12px; color:#6B7280; margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# session_state 초기화
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "stage":         0,
        "user_name":     "",
        "user_birth":    "",
        "consent_given": False,
        "answers":       {},    # 20문항 응답
        "type_scores":   {},
        "type1":         "",
        "type2":         "",
        # 시나리오 선택값
        "q1":    "",   # Q1 선택 (1~4)
        "q1f":   "",   # Q1 followup 선택 (a~d)
        "q2a":   "",   # Q2-A 선택
        "q2b":   "",   # Q2-B 선택
        "q2f":   "",   # Q2 followup 선택 (a~d or e)
        "q2f_text": "", # Q2 followup (e) 직접 입력
        # 결과
        "qli_score": 0,
        "mti_score": 0,
        "aiq_index": 0,
        "serial":    "",
        "saved":     False,
        "save_attempted": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# 단계 표시 바
# ─────────────────────────────────────────────
def step_bar(current: int):
    steps = [(1,"유형 진단"), (2,"시나리오 A"), (3,"시나리오 B"), (4,"결과")]
    html = '<div class="step-bar">'
    for i, (num, label) in enumerate(steps):
        if num < current:
            cn, cl, icon = "step-num step-done", "step-lbl-off", "✓"
        elif num == current:
            cn, cl, icon = "step-num step-on",   "step-lbl-on",  str(num)
        else:
            cn, cl, icon = "step-num step-off",  "step-lbl-off", str(num)
        html += f'<div class="step-item"><div class="{cn}">{icon}</div><span class="{cl}">{label}</span></div>'
        if i < len(steps)-1:
            html += '<div class="step-div"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# stage_0: 진입 화면
# ─────────────────────────────────────────────
def stage_0():
    st.markdown("## AIQ 파일럿 진단")
    st.markdown("""
AI와 나는 어떻게 함께 사고하는가를 측정합니다.

- **20개 행동 문항**으로 AI 협업 사고 유형을 분류합니다
- **시나리오 2개**로 질문 설계력과 사고 전환 점수를 산출합니다
- 소요 시간: 약 7분
    """)
    st.divider()

    st.markdown("### 응답자 정보")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("이름", max_chars=20, placeholder="예: 홍길동",
                             value=st.session_state.user_name)
    with col2:
        birth = st.text_input("생년월일 (YYYYMMDD)", max_chars=8,
                              placeholder="예: 19950315",
                              value=st.session_state.user_birth)
    st.caption("※ 결과 보고와 향후 재진단 시 식별을 위해 사용됩니다.")
    st.divider()

    consent = st.checkbox(
        "이름과 생년월일을 진단 결과 보고 및 데이터 분석 목적으로 수집·이용하는 것에 동의합니다.",
        value=st.session_state.consent_given
    )

    name_ok  = validate_name(name)
    birth_ok = validate_birth(birth)
    ready    = name_ok and birth_ok and consent

    if name  and not name_ok:  st.warning("이름은 2~20자의 한글/영문으로 입력해주세요.")
    if birth and not birth_ok: st.warning("생년월일은 YYYYMMDD 8자리 형식이어야 합니다.")
    if not consent:            st.caption("진단을 시작하려면 동의에 체크해주세요.")

    if st.button("진단 시작 →", type="primary", use_container_width=True, disabled=not ready):
        st.session_state.user_name     = name.strip()
        st.session_state.user_birth    = birth.strip()
        st.session_state.consent_given = True
        st.session_state.stage = 1
        st.rerun()

# ─────────────────────────────────────────────
# stage_1: 20문항 유형 진단
# ─────────────────────────────────────────────
def stage_1():
    step_bar(1)
    st.markdown("### AI와 나는 어떻게 함께 사고하는가")
    st.caption("20개 행동 문항에 응답해주세요. 정답이 없으며 평소 습관을 기준으로 선택하세요.")
    st.divider()

    SCALE = ["① 거의 안 그렇다", "② 가끔 그렇다", "③ 자주 그렇다", "④ 항상 그렇다"]

    with st.form("q_form"):
        answers = {}
        for (no, text, typ, reverse) in QUESTIONS:
            st.markdown(f"**Q{no:02d}.** {text}")
            prev       = st.session_state.answers.get(no)
            prev_index = (prev - 1) if prev else None
            val = st.radio(
                label=f"q{no}", options=[1,2,3,4],
                format_func=lambda x: SCALE[x-1],
                index=prev_index, horizontal=True,
                label_visibility="collapsed", key=f"q{no}"
            )
            answers[no] = val
            st.markdown("")

        answered = len([v for v in answers.values() if v is not None])
        st.caption(f"{answered} / 20 문항 완료")
        submitted = st.form_submit_button("✅ 완료 — 시나리오로 이동", use_container_width=True)

    if submitted:
        missing = [no for no, v in answers.items() if v is None]
        if missing:
            st.warning(f"아직 응답하지 않은 문항: Q{', Q'.join(f'{n:02d}' for n in missing)}")
        else:
            st.session_state.answers     = answers
            st.session_state.type_scores = compute_type_scores(answers)
            t1, t2 = compute_top_types(st.session_state.type_scores)
            st.session_state.type1 = t1
            st.session_state.type2 = t2
            st.session_state.stage = 2
            st.rerun()

# ─────────────────────────────────────────────
# stage_2: 시나리오 A — QLI 측정
# ─────────────────────────────────────────────
def stage_2():
    step_bar(2)
    st.markdown('<div class="scn-box"><p class="scn-title">두 사람의 조언</p><p class="scn-text">당신은 어떤 일을 시작할지 말지 고민 중이다. 믿을 만한 두 사람에게 의견을 구했더니, 한 사람은 "지금이 적기다, 바로 해라"라고 하고, 다른 한 사람은 "지금은 때가 아니다, 기다려라"라고 한다. 둘 다 당신을 잘 알고 진심으로 조언하고 있다.</p></div>', unsafe_allow_html=True)
    st.caption("정답이 없습니다. 가장 자연스럽게 떠오르는 것을 고르세요.")
    st.divider()

    # Q1
    q1 = st.radio(
        "**Q1. 이 상황에서 가장 먼저 하고 싶은 것은?**",
        options=["1","2","3","4"],
        format_func=lambda x: {
            "1": "① 더 신뢰하는 쪽의 말을 따른다",
            "2": "② 두 사람에게 각각 왜 그렇게 생각하는지 이유를 묻는다",
            "3": "③ 두 사람이 서로 다른 기준으로 보고 있을 수 있다고 생각한다",
            "4": "④ 지금 당장 결정해야 하는 상황인지부터 의심해본다",
        }[x],
        index=None, key="s2_q1"
    )

    if q1:
        st.divider()
        followup_questions = {
            "1": "그 사람을 더 신뢰하는 이유는?",
            "2": "이유를 물어보려는 것은 왜인가?",
            "3": "기준이 다르다면, 어떤 의미라고 생각하는가?",
            "4": '"지금 결정해야 하는가"를 의심하는 이유는?',
        }
        followup_options = {
            "1": {
                "a": "(a) 나를 더 오래, 더 잘 알기 때문에",
                "b": "(b) 비슷한 상황을 경험해봤기 때문에",
                "c": "(c) 과거에 그 사람 판단이 맞았던 적이 있기 때문에",
                "d": "(d) 딱히 이유는 없다 — 그냥 그 말이 더 끌린다",
            },
            "2": {
                "a": "(a) 둘 중 더 설득력 있는 쪽을 따르려고",
                "b": "(b) 각자 무엇을 기준으로 보고 있는지 파악하려고",
                "c": "(c) 두 사람의 전제 자체가 다를 수 있다고 생각해서",
                "d": "(d) 결정을 미루고 싶어서",
            },
            "3": {
                "a": "(a) 한 사람은 급하고 한 사람은 신중한 성격 차이일 것이다",
                "b": "(b) 둘이 '적기'를 판단하는 기준 자체가 다를 것이다",
                "c": "(c) 한 사람이 내 상황을 더 잘 알고 있어서 그럴 것이다",
                "d": "(d) 잘 모르겠다",
            },
            "4": {
                "a": "(a) 아직 정보가 부족해서",
                "b": "(b) 두 조언이 상충한다는 것 자체가 전제에 문제가 있다는 신호 같아서",
                "c": "(c) 결정을 미루면 상황이 더 명확해질 것 같아서",
                "d": "(d) '지금 시작할지 말지'라는 질문 자체가 잘못 설정됐을 수 있어서",
            },
        }

        q1f = st.radio(
            f"**Q1-F. {followup_questions[q1]}**",
            options=list(followup_options[q1].keys()),
            format_func=lambda x: followup_options[q1][x],
            index=None, key="s2_q1f"
        )

        if q1f:
            st.divider()
            if st.button("다음 시나리오로 →", type="primary", use_container_width=True):
                st.session_state.q1  = q1
                st.session_state.q1f = q1f
                st.session_state.stage = 3
                st.rerun()

# ─────────────────────────────────────────────
# stage_3: 시나리오 B — MTI 측정
# ─────────────────────────────────────────────
def stage_3():
    step_bar(3)
    # q1, q1f — stage_2에서 저장된 값 먼저 복원
    q1  = st.session_state.q1
    q1f = st.session_state.q1f

    st.markdown('<div class="scn-box"><p class="scn-title">친구의 부탁</p><p class="scn-text">친구가 주말에 자기 일을 도와달라고 부탁했고, 당신은 그러기로 했다. 그런데 그 주말에 당신에게도 중요한 일이 생겼다. 친구는 이미 당신이 온다고 믿고 준비를 시작했다.</p></div>', unsafe_allow_html=True)
    st.caption("정답이 없습니다. 가장 자연스럽게 떠오르는 것을 고르세요.")
    st.divider()

    # Q2-A
    q2a = st.radio(
        "**Q2-A. 이 상황을 어떻게 풀겠는가?**",
        options=["1","2","3","4"],
        format_func=lambda x: {
            "1": "① 약속했으니 친구를 돕는다",
            "2": "② 솔직하게 말하고 일정을 조율한다",
            "3": "③ 친구가 진짜 필요한 게 내 도움인지, 다른 방법도 있는지 먼저 확인한다",
            "4": "④ 약속의 의미와 상황 변화 중 무엇이 더 중요한지 따져본다",
        }[x],
        index=None, key="s3_q2a"
    )

    if q2a:
        st.divider()
        # 조건 변화 투입
        st.info("💡 **새로운 정보**: 알고 보니 친구가 부탁한 일은 당신이 아니어도 할 수 있는 일이었다.")

        q2b = st.radio(
            "**Q2-B. 이 사실을 알게 됐다. 생각이 바뀌는가?**",
            options=["1","2","3","4"],
            format_func=lambda x: {
                "1": "① 그래도 약속은 지킨다 — 내가 가기로 했으니까",
                "2": "② 다른 사람을 구해주는 방향으로 생각이 바뀐다",
                "3": "③ 처음부터 내가 꼭 가야 한다고 생각한 게 틀렸다",
                "4": "④ 이건 '내가 가느냐'의 문제가 아니라 '친구가 무엇을 필요로 하느냐'의 문제였다",
            }[x],
            index=None, key="s3_q2b"
        )

        if q2b:
            st.divider()
            # Q2-F — 전환 여부에 따라 질문 다르게
            if q2b == "1":
                f_question = "생각이 바뀌지 않는 이유는?"
                f_options = {
                    "a": "(a) 약속은 지키는 것이 원칙이다",
                    "b": "(b) 친구가 이미 준비를 시작했으니 지금 바꾸면 피해가 크다",
                    "c": "(c) 내가 가는 게 친구에게 더 중요한 의미일 수 있다",
                    "d": "(d) 솔직히 어떻게 해야 할지 모르겠다",
                    "e": "(e) 위에 해당하지 않는다 — 직접 입력",
                }
            else:
                f_question = "생각이 바뀐 이유는?"
                f_options = {
                    "a": "(a) 새 정보가 생겼으니 판단을 바꾸는 게 맞다",
                    "b": "(b) 내가 처음 판단할 때 빠뜨린 조건이 있었다",
                    "c": "(c) 문제를 다른 각도에서 보게 됐다",
                    "d": "(d) 사실 처음부터 확신이 없었는데 이유가 생긴 것뿐이다",
                    "e": "(e) 위에 해당하지 않는다 — 직접 입력",
                }

            q2f = st.radio(
                f"**Q2-F. {f_question}**",
                options=list(f_options.keys()),
                format_func=lambda x: f_options[x],
                index=None, key="s3_q2f"
            )

            q2f_text = ""
            if q2f == "e":
                q2f_text = st.text_area(
                    "직접 입력해주세요 (200자 이하)",
                    max_chars=200,
                    height=80,
                    key="s3_q2f_text"
                )

            if q2f and (q2f != "e" or q2f_text.strip()):
                st.divider()
                if st.button("결과 보기 →", type="primary", use_container_width=True):
                    # 점수 산출
                    qli = QLI_MATRIX.get((q1, q1f), 5) if (q1, q1f) in QLI_MATRIX else 5
                    mti_key = (q2b, q2f) if q2f != "e" else (q2b, "a")
                    mti = MTI_MATRIX.get(mti_key, 5)
                    aiq = compute_aiq(qli, mti)

                    st.session_state.q2a      = q2a
                    st.session_state.q2b      = q2b
                    st.session_state.q2f      = q2f
                    st.session_state.q2f_text = q2f_text.strip()
                    st.session_state.qli_score = qli
                    st.session_state.mti_score = mti
                    st.session_state.aiq_index = aiq
                    st.session_state.stage = 4
                    st.rerun()

    # (q1, q1f는 함수 첫 부분에서 복원 완료)

# ─────────────────────────────────────────────
# stage_4: 결과 리포트
# ─────────────────────────────────────────────
def stage_4():
    type1     = st.session_state.type1
    type2     = st.session_state.type2
    type_sc   = st.session_state.type_scores
    qli       = st.session_state.qli_score
    mti       = st.session_state.mti_score
    aiq       = st.session_state.aiq_index

    # 저장 — 1회만
    if not st.session_state.save_attempted:
        st.session_state.save_attempted = True
        serial = save_result(
            name=st.session_state.user_name,
            birth=st.session_state.user_birth,
            type1=type1, type2=type2,
            type_scores=type_sc,
            q1=st.session_state.q1,
            q1f=st.session_state.q1f,
            q2a=st.session_state.q2a,
            q2b=st.session_state.q2b,
            q2f=st.session_state.q2f,
            q2f_text=st.session_state.q2f_text,
            qli=qli, mti=mti, aiq=aiq,
            answers=st.session_state.answers
        )
        if serial:
            st.session_state.serial = serial
            st.session_state.saved  = True

    # 보고서 헤더
    name_disp  = st.session_state.user_name or "anonymous"
    birth_raw  = st.session_state.user_birth or ""
    birth_disp = f"{birth_raw[:4]}.{birth_raw[4:6]}.{birth_raw[6:]}" if len(birth_raw)==8 else ""
    serial_disp = st.session_state.serial or "—"
    diag_time  = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    st.markdown(f'''
    <div class="report-header">
      <p class="report-title">AIQ Diagnostic Report</p>
      <p class="report-name">AI 공생 지수 진단 결과</p>
      <div class="report-meta">
        <span><strong>응답자</strong>{name_disp} ({birth_disp})</span>
        <span><strong>시리얼</strong>{serial_disp}</span>
        <span><strong>진단일</strong>{diag_time}</span>
        <span><strong>버전</strong>AIQ v9.0 · 파일럿</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # AIQ 히어로
    st.markdown(f'''
    <div class="aiq-hero">
      <p class="aiq-label">AIQ</p>
      <p class="aiq-value">{aiq}</p>
      <div class="aiq-badge-row">
        <span class="aiq-badge">{type1}</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # 유형 설명
    axis = TYPE_AXIS.get(type1, ("",""))
    st.markdown(f'<h2 class="section-title">{type1}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="axis-tag">{axis[0]} · {axis[1]} — 1순위</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="type-quote">{TYPE_DESC.get(type1,"")}</p>', unsafe_allow_html=True)
    st.divider()

    # 유형 좌표
    st.markdown('<p class="sub-section-title">유형 좌표</p>', unsafe_allow_html=True)
    type_order = ["상상가형","설계자형","의존형","실행형"]
    labels_map = {"설계자":"설계자형","상상가":"상상가형","실행":"실행형","의존":"의존형"}
    html = '<div class="coord-grid">'
    for t in type_order:
        is_first  = (t == type1)
        is_second = (t == type2)
        rank = "1순위" if is_first else ("2순위" if is_second else "")
        cls  = "cc cc-hl" if is_first else "cc"
        html += f'<div class="{cls}"><span class="cc-name">{t}{" ✦" if is_first else ""}</span><span class="cc-tag">{rank}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    if type2:
        st.markdown(f'<p class="second-rank">2순위: {type2}</p>', unsafe_allow_html=True)
    st.divider()

    # 서브 지표
    st.markdown('<p class="sub-section-title">측정 근거</p>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="sub-metrics">
      <div class="sm"><p class="sm-lbl">질문 설계 QLI</p><p class="sm-val">{qli}</p></div>
      <div class="sm"><p class="sm-lbl">사고 전환 MTI</p><p class="sm-val">{mti}</p></div>
    </div>
    ''', unsafe_allow_html=True)
    st.divider()

    # 저장 상태
    if st.session_state.saved:
        st.caption(f"✅ 결과가 저장되었습니다.  ·  시리얼: {st.session_state.serial}")
    else:
        st.warning("저장 실패 — 운영자에게 알려주세요.")

    if st.button("처음으로 돌아가기"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────
stage = st.session_state.get("stage", 0)
if   stage == 0: stage_0()
elif stage == 1: stage_1()
elif stage == 2: stage_2()
elif stage == 3: stage_3()
elif stage == 4: stage_4()
