# -*- coding: utf-8 -*-
"""
AIQ 파일럿 진단 앱 (단일 파일) — 설계기준 v0.3 전체 반영
- 인증(사번10자리 + 이메일 6자리 코드) → [무작위 순서] {L1 설문 · L3 대화} → 채점 → 저장
- 점수는 보류-후-일괄(유효 200) 모델이라 응시 시점에는 표시하지 않음(유형·근거 코멘트만)
- 의존성: streamlit, openai, gspread, google-auth, pytz + 표준라이브러리(smtplib, secrets, email, datetime, json, statistics, random)
- 비밀값은 .streamlit/secrets.toml 의 EMAIL_ADDRESS / EMAIL_APP_PW / OPENAI_API_KEY / SHEET_ID / [gcp_service_account]
"""

import json
import random
import secrets as pysecrets
import smtplib
import statistics
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.utils import formataddr

import pytz
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG (구현명세서 §0 디폴트 · §1~§5)
# ──────────────────────────────────────────────────────────────────────────────
APP_VERSION = "AIQ v-next 0.3 · 파일럿"
KST = pytz.timezone("Asia/Seoul")

CODE_TTL_MIN = 10        # 인증 코드 유효(분)
EMP_ID_LEN = 10          # 사번 자릿수(변경 시 이 값만 수정)
CODE_COOLDOWN_SEC = 60   # 재발송 쿨다운(초)
CODE_MAX_ATTEMPTS = 3    # 실패 잠금
TYPE_GAP_T = 2           # 1·2순위 점수차 임계(미만이면 '복합')
K_PASSES = 3             # 채점 반복
SCORING_TEMP = 0.3       # 채점 온도
PARTNER_TEMP = 0.7       # 대화 파트너 온도
MAX_USER_TURNS = 7       # 사용자 발화 상한
PERTURB_AFTER = 2        # n번째 사용자 발화 직후 도전질문 삽입
NORM_THRESHOLD = 200     # 유효 응답 정규화 트리거

RESP_WS = "responses"
IDMAP_WS = "id_map"
META_WS = "meta"         # norm_locked 등 상태

# ── 토픽(일상 딜레마, 지식장벽 低·추론부하 高) ────────────────────────────────
TOPICS = {
    "T1": {
        "title": "이사",
        "dilemma": "직장과 가까운 비싼 도심 vs 멀지만 저렴하고 환경 좋은 외곽, 어디로 이사할지",
        "intro": "도심은 가깝지만 비싸고, 외곽은 멀지만 저렴하고 환경이 좋습니다. 어떤 점부터 따져볼까요?",
    },
    "T2": {
        "title": "진로",
        "dilemma": "안정적이지만 성장 느린 자리 vs 불안정하지만 성장 빠른 기회, 무엇을 택할지",
        "intro": "안정과 성장은 자주 부딪칩니다. 지금 가장 중요하게 보는 기준은 무엇인가요?",
    },
    "T3": {
        "title": "우선순위",
        "dilemma": "한정된 시간에 중요한 일 A와 급한 일 B가 충돌할 때 어떻게 배분할지",
        "intro": "중요한 일과 급한 일이 부딪칩니다. 지금 상황을 어떻게 보고 계신가요?",
    },
}

# ── L1 문항(성향별 5 = 순3·역2) + 일관성 점검 2 ───────────────────────────────
# key: F(정방향) / R(역방향)  ·  type: 설계/상상/실행/의존  ·  CK: 일관성(유형 미합산)
L1_ITEMS = [
    # 설계자형
    {"code": "D1", "type": "설계", "key": "F", "text": "질문을 보내기 전에 원하는 결과물의 형태나 관점을 먼저 정했다."},
    {"code": "D2", "type": "설계", "key": "F", "text": "AI에게 출력 형식·범위·조건을 구체적으로 지정했다."},
    {"code": "D3", "type": "설계", "key": "F", "text": "의도가 잘못 전달됐을 때 곧바로 교정 질문을 보냈다."},
    {"code": "D4", "type": "설계", "key": "R", "text": "무엇을 얻을지 정하지 않은 채 일단 AI에게 물어봤다."},
    {"code": "D5", "type": "설계", "key": "R", "text": "AI 답을 받은 뒤 내 기준으로 다시 따져보지 않고 그대로 진행했다."},
    # 상상가형
    {"code": "I1", "type": "상상", "key": "F", "text": "“만약 ~라면” 같은 가정을 던지며 대화를 확장했다."},
    {"code": "I2", "type": "상상", "key": "F", "text": "예상치 못한 연결이나 새로운 관점을 발견하려고 질문했다."},
    {"code": "I3", "type": "상상", "key": "F", "text": "정답보다 탐색 자체를 목적으로 AI와 대화했다."},
    {"code": "I4", "type": "상상", "key": "R", "text": "떠오른 아이디어를 더 펼치지 않고 한두 마디로 끝냈다."},
    {"code": "I5", "type": "상상", "key": "R", "text": "새로운 방향이 보여도 처음 정한 범위 안에서만 질문했다."},
    # 실행형
    {"code": "X1", "type": "실행", "key": "F", "text": "AI가 준 결과를 곧바로 적용하거나 실행해봤다."},
    {"code": "X2", "type": "실행", "key": "F", "text": "AI 결과를 내 상황에 맞게 고쳐서 사용했다."},
    {"code": "X3", "type": "실행", "key": "F", "text": "받은 답을 다른 도구·방법과 결합해 써먹었다."},
    {"code": "X4", "type": "실행", "key": "R", "text": "AI 결과를 실제로 써보지 않고 읽기만 하고 넘어갔다."},
    {"code": "X5", "type": "실행", "key": "R", "text": "결과를 고치거나 적용하기보다 그대로 두었다."},
    # 의존형
    {"code": "P1", "type": "의존", "key": "F", "text": "AI가 준 답을 별 의심 없이 그대로 받아들였다."},
    {"code": "P2", "type": "의존", "key": "F", "text": "AI 답이 마음에 안 들어도 다시 시도하지 않고 그냥 썼다."},
    {"code": "P3", "type": "의존", "key": "F", "text": "AI 없이 혼자 문제를 푸는 것이 부담스러웠다."},
    {"code": "P4", "type": "의존", "key": "R", "text": "AI 답이 내 상황과 다르다고 느껴 반론하거나 수정 요청을 했다."},
    {"code": "P5", "type": "의존", "key": "R", "text": "AI 답의 한계나 틀린 점을 스스로 짚어봤다."},
    # 일관성 점검(유형 미합산)
    {"code": "CK1", "type": "CK", "key": "F", "text": "AI에게 묻기 전에 목표를 분명히 해두었다."},   # D1 패러프레이즈
    {"code": "CK2", "type": "CK", "key": "F", "text": "AI 답을 검토 없이 신뢰하는 편이었다."},        # P1 패러프레이즈
]
L1_SCALE = {1: "거의 안 함", 2: "가끔", 3: "자주", 4: "거의 매번"}

# ── L2 자기효능감(태도 4점, 유형 미합산) ──────────────────────────────────────
L2_ITEMS = [
    {"code": "E1", "key": "F", "text": "나는 AI를 활용해 원하는 결과를 얻을 자신이 있다."},
    {"code": "E2", "key": "F", "text": "새로운 AI 도구도 금방 익혀 쓸 수 있다."},
    {"code": "E3", "key": "R", "text": "AI를 제대로 활용하지 못할까 봐 걱정된다."},
    {"code": "E4", "key": "R", "text": "AI를 쓸 때 내가 잘하고 있는지 확신이 안 선다."},
]
L2_SCALE = {1: "전혀 아니다", 2: "아니다", 3: "그렇다", 4: "매우 그렇다"}

# ── 유형 프로파일(드래프트 · SJ 검토 대상) ────────────────────────────────────
TYPE_PROFILES = {
    "설계": {
        "name": "설계자형",
        "axis": "질문력↑ · 재구성↑",
        "one_line": "AI를 도구로 부리는 사람. 질문 전에 목적을 정의하고, 결과를 받으면 재조립한다.",
        "sections": {
            "이런 사람입니다": "문제를 정의하는 일 자체에 공을 들인다. 첫 질문이 이미 구체적이고, AI 답을 그대로 쓰기보다 자기 기준으로 다시 짠다.",
            "가장 빛나는 순간": "복잡한 문제를 잘게 나눠 AI에 맡기고, 돌아온 조각을 엮어 결론을 만들 때. AI를 확장된 작업대처럼 쓴다.",
            "이 유형의 함정": "모든 걸 직접 설계하려다 속도가 느려질 수 있다. 위임해도 될 일까지 끌어안는다.",
            "다른 사람 눈에 비치는 당신": "방향이 분명하고 결과물의 완성도가 높은 사람. 다만 가끔 과하게 통제한다는 인상을 줄 수 있다.",
            "지금 필요한 한 가지": "충분히 좋은 초안은 AI에 더 맡겨보라. 설계와 실행의 균형이 생산성을 키운다.",
        },
    },
    "상상": {
        "name": "상상가형",
        "axis": "질문력↑ · 재구성↓",
        "one_line": "질문은 풍부하나 착지가 늦은 사람. 아이디어를 펼치는 데 강하다.",
        "sections": {
            "이런 사람입니다": "“만약 ~라면”을 즐긴다. 정답보다 탐색이 목적이고, 예상 밖의 연결을 잘 찾는다.",
            "가장 빛나는 순간": "아무도 묻지 않은 질문을 꺼낼 때. 개념 설계·아이디어 발산에서 특히 강하다.",
            "이 유형의 함정": "탐색이 즐거운 나머지 결론을 못 맺는다. 좋은 질문이 결과물 없이 쌓일 수 있다.",
            "다른 사람 눈에 비치는 당신": "시각이 넓고 독창적인 사람. 다만 결과물의 완성도가 들쭉날쭉해 보일 수 있다.",
            "지금 필요한 한 가지": "대화 시작 전에 “이 대화의 결과물은 ___이다”를 먼저 적어보라. 탐색이 착지로 바뀐다.",
        },
    },
    "실행": {
        "name": "실행형",
        "axis": "질문력↓ · 재구성↑",
        "one_line": "받으면 바로 적용·수정하는 사람. 실행과 변환에 강하다.",
        "sections": {
            "이런 사람입니다": "AI 답을 곧장 써보고, 자기 상황에 맞게 고친다. 머무르기보다 움직인다.",
            "가장 빛나는 순간": "받은 결과를 실제 과제에 즉시 적용하고 다른 도구와 결합해 끝을 볼 때.",
            "이 유형의 함정": "첫 질문 설계가 약해 엉뚱한 답을 받고도 일단 적용할 수 있다. 방향을 먼저 잡으면 효율이 크게 오른다.",
            "다른 사람 눈에 비치는 당신": "추진력 있고 결과를 내는 사람. 다만 검토 없이 빨리 간다는 인상을 줄 수 있다.",
            "지금 필요한 한 가지": "실행 전에 “무엇을 얻으려는가”를 한 줄 적어보라. 같은 속도로 더 정확해진다.",
        },
    },
    "의존": {
        "name": "의존형",
        "axis": "질문력↓ · 재구성↓",
        "one_line": "AI 답을 기준 삼아 판단하는 경향. 사고 역량을 키울 여지가 크다.",
        "sections": {
            "이런 사람입니다": "AI 답을 대체로 그대로 받아들이고, 마음에 안 들어도 다시 시도하기보다 그냥 쓰는 편이다.",
            "가장 빛나는 순간": "정형화된 일을 빠르게 처리할 때. AI가 좋은 출발점을 줄 때 효율이 좋다.",
            "이 유형의 함정": "AI 답의 한계를 스스로 짚지 못하면, 틀린 전제 위에서 일이 진행될 수 있다.",
            "다른 사람 눈에 비치는 당신": "AI를 적극 활용하는 사람. 다만 결과를 검증한다는 신뢰는 아직 덜 줄 수 있다.",
            "지금 필요한 한 가지": "AI 답을 받으면 “이 답이 틀렸다면 어디일까?”를 한 번만 물어보라. 작은 의심이 큰 차이를 만든다.",
        },
    },
}
TYPE_ORDER = ["설계", "상상", "실행", "의존"]  # 2x2 표시 순서

# ── 채점 앵커(프롬프트에 그대로 삽입, 구현 §1.2·1.3) ──────────────────────────
SCORING_PROMPT_HEAD = """당신은 심리측정 채점 전문가다. 아래는 한 사용자가 AI와 일상 주제를 논의한 대화 전문이다.
사용자의 발화만 근거로 7개 세부지표를 각각 1~10 정수로 채점한다.
대화 내용의 '옳고 그름'이나 '지식 수준'은 평가하지 않는다. 사고의 구조만 본다.

[QLI — '[개시구간]'만 본다] (각 1~10)
1 목적정향성: 1-2 막연 / 3-4 주제만 / 5-6 산출형태 일부 / 7-8 산출형태+맥락 / 9-10 산출·관점·기준 규정
2 맥락구조화: 1-2 없음 / 3-4 1요소 / 5-6 2요소 / 7-8 3요소+ / 9-10 제약·예외·우선순위까지
3 인지부하설계: 1-2 단순사실 / 3-4 단일설명 / 5-6 한단계추론 / 7-8 비교·인과·조건부 / 9-10 시나리오·반사실·트레이드오프
4 주도성: 1-2 완전위임 / 3-4 대체로위임 / 5-6 절반 / 7-8 사용자가틀+부분위임 / 9-10 전체설계

[MTI — '[전환구간]'만 본다] (각 1~10)
5 프레임전환: 1-2 무시/반복 / 3-4 방어 / 5-6 일부조정 / 7-8 전제 재검토 / 9-10 틀 자체 재설정
6 심화전환: 1-2 수용 / 3-4 표면질문 / 5-6 근거요구 / 7-8 허점·전제 지목 / 9-10 반례·조건으로 압박
7 통합전환: 1-2 단발 / 3-4 약한참조 / 5-6 연결시도 / 7-8 엮어 새관점 / 9-10 통합해 상위 질문
- '[전환구간]'의 사용자 발화가 2개 미만이면 MTI를 채점하지 말고 mti_status="undetermined".

[transition_turn] 도전 질문 직후 사용자 발화를 1로 세어, 프레임 또는 심화 전환이 처음 나타난 발화 번호. 없으면 0.

[few-shot 앵커 — 주제 '이사']
A(高): 개시 "재택2·출근3 기준 통근1h↑ vs 월40만원 절약, 3년 누적 비교해줘. 단 아이 학교는 외곽이 좋아" / 도전후 "통근을 손실로만 봤네, 전제를 다시 잡아야겠다" → purpose8 context9 load8 initiative8 frame8 deepen7 integrate7 tt1
B(中): 개시 "도심·외곽 이사 어디가 나아? 장단점 정리해줘" / 도전후 "그럼 외곽 단점 더 알려줘" → purpose5 context3 load4 initiative4 frame4 deepen5 integrate4 tt0
C(低/보류): 개시 "이사 어디로 갈까?" / 도전후 발화없음 → purpose2 context1 load1 initiative1 mti_status="undetermined" tt0

아래 JSON만 출력한다. 다른 텍스트 금지.
{"purpose":N,"context":N,"load":N,"initiative":N,"frame":N,"deepen":N,"integrate":N,"mti_status":"scored 또는 undetermined","transition_turn":N,"comment":"채점 근거 1~2문장(행동 인용)"}

[대화 전문]
"""

PARTNER_SYSTEM = """당신은 '{dilemma}'를 함께 고민하는 토론 파트너다.
- 한국어로 3~5문장으로 답한다.
- 사용자의 말에 충실히 답하되, 매 답변에 사용자가 미처 보지 못한 관점·정보를 정확히 1개 더한다.
- 단정하지 말고 근거와 함께 제시한다.
- 사용자에게 되묻지 않는다. 답변만 한다."""

PERTURB_PROMPT = """아래 대화에서 사용자가 지금까지 당연하게 전제한 가정 1개를 식별하라.
그 전제가 사실이 아닐 수 있음을 한 문장 질문으로 제시하라.
형식: "지금까지 ___를 당연한 전제로 두신 것 같은데, 만약 그렇지 않다면 어떻게 달라질까요?"
새로운 정보·조언을 주지 말 것. 전제는 정확히 1개만 건드릴 것. 질문 한 문장만 출력.

[대화]
"""


# ──────────────────────────────────────────────────────────────────────────────
# 외부 연결 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def openai_model():
    return st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")


@st.cache_resource(show_spinner=False)
def get_spreadsheet():
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=scopes
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(st.secrets["SHEET_ID"])


def get_ws(name, header=None):
    """워크시트 가져오기(없으면 생성). header 주면 1행에 보장."""
    sh = get_spreadsheet()
    try:
        ws = sh.worksheet(name)
    except Exception:
        ws = sh.add_worksheet(title=name, rows=1000, cols=60)
        if header:
            ws.append_row(header)
        return ws
    if header:
        first = ws.row_values(1)
        if not first:
            ws.append_row(header)
    return ws


def now_kst():
    return datetime.now(KST)


# ──────────────────────────────────────────────────────────────────────────────
# 인증 (사번 + 이메일 6자리 코드)
# ──────────────────────────────────────────────────────────────────────────────
def send_email_code(to_email, code):
    sender = st.secrets["EMAIL_ADDRESS"]
    pw = st.secrets["EMAIL_APP_PW"]
    body = f"AIQ 진단 본인확인 코드: {code}\n유효시간 {CODE_TTL_MIN}분.\n본 메일을 요청하지 않았다면 무시하세요."
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = "[AIQ] 본인확인 코드"
    msg["From"] = formataddr(("AIQ 진단", sender))
    msg["To"] = to_email
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as s:
        s.starttls()
        s.login(sender, pw)
        s.sendmail(sender, [to_email], msg.as_string())


def render_auth():
    st.subheader("본인 확인")
    st.caption("사번(10자리)과 이메일을 입력하면 6자리 코드를 메일로 보냅니다. 스팸함도 확인해 주세요.")
    st.caption("ℹ️ 입력하신 이메일은 **결과 발송을 위해 수집·보관**되며, 진단 결과 안내 외 용도로 사용하지 않습니다.")
    ss = st.session_state
    emp_id = st.text_input(f"사번 (숫자 {EMP_ID_LEN}자리)", value=ss.get("emp_id", ""), max_chars=EMP_ID_LEN)
    email = st.text_input("이메일", value=ss.get("email", ""))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("인증 코드 보내기", use_container_width=True):
            if not (emp_id.isdigit() and len(emp_id) == EMP_ID_LEN):
                st.error(f"사번은 숫자 {EMP_ID_LEN}자리여야 합니다.")
            elif "@" not in email or "." not in email:
                st.error("이메일 형식을 확인하세요.")
            else:
                last = ss.get("code_sent_at")
                if last and (now_kst() - last).total_seconds() < CODE_COOLDOWN_SEC:
                    wait = int(CODE_COOLDOWN_SEC - (now_kst() - last).total_seconds())
                    st.warning(f"재발송은 {wait}초 후 가능합니다.")
                else:
                    code = f"{pysecrets.randbelow(10**6):06d}"
                    try:
                        send_email_code(email, code)
                        ss.emp_id, ss.email = emp_id, email
                        ss.code = code
                        ss.code_expires = now_kst() + timedelta(minutes=CODE_TTL_MIN)
                        ss.code_attempts = 0
                        ss.code_sent_at = now_kst()
                        ss.code_sent = True
                        st.success("코드를 보냈습니다. 메일함(및 스팸함)을 확인하세요.")
                    except Exception as e:
                        st.error(f"메일 발송 실패: {e}")

    if ss.get("code_sent"):
        code_in = st.text_input("받은 6자리 코드", max_chars=6)
        with c2:
            if st.button("확인", use_container_width=True):
                if ss.get("code_attempts", 0) >= CODE_MAX_ATTEMPTS:
                    st.error("실패 횟수를 초과했습니다. 코드를 다시 받아주세요.")
                elif now_kst() > ss.get("code_expires", now_kst()):
                    st.error("코드가 만료되었습니다. 다시 받아주세요.")
                elif code_in.strip() == ss.get("code"):
                    ss.verified = True
                    ss.serial = "AIQ_" + now_kst().strftime("%m%d_%H%M%S")
                    ss.order = ["l1", "l3"] if random.random() < 0.5 else ["l3", "l1"]
                    ss.step = 0
                    ss.topic_id = random.choice(list(TOPICS.keys()))
                    ss.stage = "flow"
                    st.success("본인 확인 완료.")
                    st.rerun()
                else:
                    ss.code_attempts = ss.get("code_attempts", 0) + 1
                    left = CODE_MAX_ATTEMPTS - ss.code_attempts
                    st.error(f"코드가 일치하지 않습니다. (남은 시도 {left}회)")


# ──────────────────────────────────────────────────────────────────────────────
# L1 설문 + 채점
# ──────────────────────────────────────────────────────────────────────────────
def render_l1():
    st.subheader("1단계 · 스스로 답하기")
    st.caption("지난 1주간 AI를 사용하면서 다음 행동을 얼마나 했습니까? 정답은 없습니다 — 평소 모습 그대로 답하세요.")
    ss = st.session_state
    resp = ss.get("l1_resp", {})
    with st.form("l1_form"):
        for it in L1_ITEMS:
            resp[it["code"]] = st.radio(
                it["text"],
                options=[1, 2, 3, 4],
                format_func=lambda v: f"{v} · {L1_SCALE[v]}",
                horizontal=True,
                index=None,  # 기본 선택 없음(묵종·중앙 편향 방지)
                key="l1_" + it["code"],
            )
        st.markdown("---")
        st.caption("AI에 대한 생각")
        for it in L2_ITEMS:
            resp[it["code"]] = st.radio(
                it["text"],
                options=[1, 2, 3, 4],
                format_func=lambda v: f"{v} · {L2_SCALE[v]}",
                horizontal=True,
                index=None,
                key="l2_" + it["code"],
            )
        submitted = st.form_submit_button("다음", use_container_width=True)
    if submitted:
        missing = [it["code"] for it in (L1_ITEMS + L2_ITEMS) if resp.get(it["code"]) is None]
        if missing:
            st.error(f"모든 문항에 답해 주세요. (미응답 {len(missing)}개)")
        else:
            ss.l1_resp = resp
            ss.l1_result = score_l1(resp)
            ss.step += 1
            st.rerun()


def score_l1(resp):
    def val(item):
        v = resp.get(item["code"], 0)
        return (5 - v) if item["key"] == "R" else v

    sums = {t: 0 for t in TYPE_ORDER}
    for it in L1_ITEMS:
        if it["type"] in sums:
            sums[it["type"]] += val(it)

    ranked = sorted(TYPE_ORDER, key=lambda t: sums[t], reverse=True)
    t1, t2 = ranked[0], ranked[1]
    s1 = sums[t1]

    if s1 >= 17:
        level = "뚜렷"
    elif s1 >= 13:
        level = "보통"
    elif s1 >= 9:
        level = "복합"
    else:
        level = "미형성"

    # 표시 규칙(§3.5)
    if level == "미형성":
        display = "경향 약함 — 아직 뚜렷한 패턴이 나타나지 않았습니다."
        primary = None
    elif (s1 - sums[t2]) < TYPE_GAP_T:
        display = f"복합 (혼합 경향) — {TYPE_PROFILES[t1]['name']} · {TYPE_PROFILES[t2]['name']}"
        primary = t1
    else:
        display = f"{TYPE_PROFILES[t1]['name']} · {level} (1순위) / {TYPE_PROFILES[t2]['name']} (2순위)"
        primary = t1

    # 일관성/직선 플래그
    consistency = (
        abs(resp.get("D1", 0) - resp.get("CK1", 0)) >= 2
        or abs(resp.get("P1", 0) - resp.get("CK2", 0)) >= 2
    )
    answered = [resp.get(it["code"]) for it in L1_ITEMS]
    straightline = len(set(answered)) == 1

    eff = 0
    for it in L2_ITEMS:
        v = resp.get(it["code"], 0)
        eff += (5 - v) if it["key"] == "R" else v

    # 불일치용 좌표(§5.4)
    qli_tend = (sums["설계"] + sums["상상"]) - (sums["실행"] + sums["의존"])
    recon_tend = (sums["설계"] + sums["실행"]) - (sums["상상"] + sums["의존"])

    return {
        "sums": sums, "type1": t1, "type2": t2, "level1": level,
        "type_display": display, "primary": primary,
        "consistency_flag": consistency, "straightline_flag": straightline,
        "efficacy_score": eff, "L1_QLI성향": qli_tend, "L1_재구성성향": recon_tend,
    }


# ──────────────────────────────────────────────────────────────────────────────
# L3 대화 + 표준 도전 질문
# ──────────────────────────────────────────────────────────────────────────────
def ai_reply(messages, dilemma):
    client = get_openai_client()
    sys = PARTNER_SYSTEM.format(dilemma=dilemma)
    chat = [{"role": "system", "content": sys}] + messages
    r = client.chat.completions.create(model=openai_model(), messages=chat, temperature=PARTNER_TEMP)
    return r.choices[0].message.content.strip()


def make_perturbation(messages):
    client = get_openai_client()
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    r = client.chat.completions.create(
        model=openai_model(),
        messages=[{"role": "user", "content": PERTURB_PROMPT + convo}],
        temperature=SCORING_TEMP,
    )
    return r.choices[0].message.content.strip()


def render_l3():
    ss = st.session_state
    topic = TOPICS[ss.topic_id]
    st.subheader("2단계 · AI와 대화하기")
    st.info(f"**논의 주제 · {topic['title']}**\n\n{topic['dilemma']}에 대해 논의합니다. AI에게 가장 먼저 어떤 질문을 하시겠습니까?")
    st.caption(f"대화는 최대 {MAX_USER_TURNS}번까지 이어갈 수 있고, 충분하다고 느끼면 마칠 수 있습니다.")

    if "l3_msgs" not in ss:
        ss.l3_msgs = []           # [{role, content}]  (AI 도전 질문은 role=assistant, challenge=True)
        ss.l3_user_turns = 0
        ss.l3_perturbed = False
        ss.l3_perturb_user_index = None  # 도전 직전까지의 사용자 발화 수
        # AI 인트로 1회
        ss.l3_msgs.append({"role": "assistant", "content": topic["intro"], "challenge": False})

    # 렌더(시간순, 입력창은 항상 하단)
    for m in ss.l3_msgs:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            if m.get("challenge"):
                st.markdown(f"🔆 **(생각해볼 질문)** {m['content']}")
            else:
                st.markdown(m["content"])

    done_col = st.container()
    user_text = st.chat_input("메시지를 입력하세요" if ss.l3_user_turns < MAX_USER_TURNS else "최대 턴에 도달했습니다")

    if user_text and ss.l3_user_turns < MAX_USER_TURNS:
        ss.l3_msgs.append({"role": "user", "content": user_text, "challenge": False})
        ss.l3_user_turns += 1
        api_msgs = [{"role": m["role"], "content": m["content"]} for m in ss.l3_msgs]

        # 2번째 사용자 발화 직후 → 표준 도전 질문 1회
        if (not ss.l3_perturbed) and ss.l3_user_turns == PERTURB_AFTER:
            try:
                q = make_perturbation(api_msgs)
            except Exception:
                q = "지금까지 한 가지를 당연한 전제로 두신 것 같은데, 만약 그렇지 않다면 판단이 달라질까요?"
            ss.l3_msgs.append({"role": "assistant", "content": q, "challenge": True})
            ss.l3_perturbed = True
            ss.l3_perturb_user_index = ss.l3_user_turns
            ss.perturbation_text = q
        else:
            try:
                a = ai_reply(api_msgs, topic["dilemma"])
            except Exception as e:
                a = f"(응답 생성 오류: {e})"
            ss.l3_msgs.append({"role": "assistant", "content": a, "challenge": False})
        st.rerun()

    # 종료 버튼(발화 2회 이상부터)
    with done_col:
        if ss.l3_user_turns >= 2:
            if st.button("논의 마치고 다음으로 →", use_container_width=True, type="primary"):
                ss.l3_done = True
                ss.step += 1
                st.rerun()


def build_segments():
    """[개시구간]/[전환구간]/[AI]/[AI·도전] 태깅 전문 + 전환구간 사용자 발화 수."""
    ss = st.session_state
    lines, seg = [], "개시"
    post_user = 0
    for m in ss.l3_msgs:
        if m["role"] == "user":
            tag = "[개시구간]" if seg == "개시" else "[전환구간]"
            lines.append(f"{tag} 사용자: {m['content']}")
            if seg == "전환":
                post_user += 1
        else:
            if m.get("challenge"):
                lines.append(f"[AI·도전] {m['content']}")
                seg = "전환"
            else:
                lines.append(f"[AI] {m['content']}")
    return "\n".join(lines), post_user


# ──────────────────────────────────────────────────────────────────────────────
# 채점 (k=3 · 중앙값 · 판단보류)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_score(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    return json.loads(t[t.find("{"): t.rfind("}") + 1])


def score_dialogue(transcript, post_user_turns):
    client = get_openai_client()
    passes = []
    for _ in range(K_PASSES):
        try:
            r = client.chat.completions.create(
                model=openai_model(),
                messages=[{"role": "user", "content": SCORING_PROMPT_HEAD + transcript}],
                temperature=SCORING_TEMP,
            )
            passes.append(_parse_score(r.choices[0].message.content))
        except Exception:
            continue

    if len(passes) < 2:
        return {"aiq_raw": None, "mti_status": "error", "valid": False,
                "reason": "채점 실패", "qli_mean": None, "mti_mean": None,
                "transition_turn": 0, "comment": "", "pass_var_qli": None, "pass_var_mti": None,
                "qli": {}, "mti": {}, "passes": passes}

    def med(key):
        return statistics.median([p[key] for p in passes if key in p])

    qli_keys = ["purpose", "context", "load", "initiative"]
    mti_keys = ["frame", "deepen", "integrate"]
    qli = {k: med(k) for k in qli_keys}
    qli_mean = round(sum(qli.values()) / 4, 2)

    undetermined = (post_user_turns < 2) or any(p.get("mti_status") == "undetermined" for p in passes)
    if undetermined:
        return {"aiq_raw": None, "mti_status": "undetermined", "valid": False,
                "reason": "행동 변화 미관측(전환구간 발화 부족)",
                "qli_mean": qli_mean, "mti_mean": None,
                "transition_turn": int(med("transition_turn")) if any("transition_turn" in p for p in passes) else 0,
                "comment": passes[0].get("comment", ""),
                "pass_var_qli": round(statistics.pvariance([p["purpose"] for p in passes]), 2) if len(passes) > 1 else 0,
                "pass_var_mti": None, "qli": qli, "mti": {}, "passes": passes}

    mti = {k: med(k) for k in mti_keys}
    mti_mean = round(sum(mti.values()) / 3, 2)
    aiq_raw = max(0, min(200, round(100 + (qli_mean + mti_mean - 10) * 5)))
    return {
        "aiq_raw": aiq_raw, "mti_status": "scored", "valid": True, "reason": "",
        "qli_mean": qli_mean, "mti_mean": mti_mean,
        "transition_turn": int(med("transition_turn")),
        "comment": passes[0].get("comment", ""),
        "pass_var_qli": round(statistics.pvariance([sum(p[k] for k in qli_keys) / 4 for p in passes]), 2),
        "pass_var_mti": round(statistics.pvariance([sum(p[k] for k in mti_keys) / 3 for p in passes]), 2),
        "qli": qli, "mti": mti, "passes": passes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 저장 (responses + id_map 분리) · 유효성 판정
# ──────────────────────────────────────────────────────────────────────────────
RESP_HEADER = [
    "serial", "ts_kst", "verified", "order_mode", "topic_id",
    "type_설계", "type_상상", "type_실행", "type_의존",
    "type1", "type2", "level1", "type_display",
    "consistency_flag", "straightline_flag", "efficacy_score",
    "qli_purpose", "qli_context", "qli_load", "qli_initiative", "qli_mean",
    "mti_frame", "mti_deepen", "mti_integrate", "mti_mean", "mti_status",
    "transition_turn", "pass_var_qli", "pass_var_mti", "aiq_raw", "comment",
    "L1_QLI성향", "L1_재구성성향", "transcript", "perturbation_text",
    "valid_flag", "invalid_reason", "aiq_pct", "pct_label", "result_sent",
]
IDMAP_HEADER = ["serial", "emp_id", "email", "ts_kst"]


def save_response(transcript, sc, pct="", label="", result_sent=""):
    ss = st.session_state
    l1 = ss.l1_result

    # 유효성 판정(§5.3)
    invalid = []
    if not ss.get("verified"):
        invalid.append("미인증")
    if l1["straightline_flag"]:
        invalid.append("직선응답")
    if l1["consistency_flag"]:
        invalid.append("일관성의심")
    if ss.get("l3_user_turns", 0) < 2:
        invalid.append("발화부족")
    if sc["mti_status"] != "scored":
        invalid.append("전환력보류" if sc["mti_status"] == "undetermined" else "채점실패")
    valid = "유효" if not invalid else "무효"

    q = sc.get("qli", {})
    m = sc.get("mti", {})
    row = [
        ss.serial, now_kst().strftime("%Y-%m-%d %H:%M:%S"), True, "/".join(ss.order), ss.topic_id,
        l1["sums"]["설계"], l1["sums"]["상상"], l1["sums"]["실행"], l1["sums"]["의존"],
        l1["type1"], l1["type2"], l1["level1"], l1["type_display"],
        l1["consistency_flag"], l1["straightline_flag"], l1["efficacy_score"],
        q.get("purpose"), q.get("context"), q.get("load"), q.get("initiative"), sc.get("qli_mean"),
        m.get("frame"), m.get("deepen"), m.get("integrate"), sc.get("mti_mean"), sc.get("mti_status"),
        sc.get("transition_turn"), sc.get("pass_var_qli"), sc.get("pass_var_mti"),
        sc.get("aiq_raw"), sc.get("comment"),
        l1["L1_QLI성향"], l1["L1_재구성성향"], transcript, ss.get("perturbation_text", ""),
        valid, ";".join(invalid), pct, label, result_sent,
    ]
    get_ws(RESP_WS, RESP_HEADER).append_row(row, value_input_option="USER_ENTERED")
    get_ws(IDMAP_WS, IDMAP_HEADER).append_row(
        [ss.serial, ss.emp_id, ss.email, now_kst().strftime("%Y-%m-%d %H:%M:%S")]
    )
    return valid


def valid_count():
    try:
        ws = get_ws(RESP_WS, RESP_HEADER)
        col = ws.col_values(RESP_HEADER.index("valid_flag") + 1)[1:]
        return sum(1 for v in col if v == "유효")
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# 정규화 (유효 200 자동·1회 잠금) — 점수는 이후 단계에서만 표시
# ──────────────────────────────────────────────────────────────────────────────
def norm_locked():
    try:
        ws = get_ws(META_WS, ["key", "value"])
        for r in ws.get_all_values()[1:]:
            if r and r[0] == "norm_locked":
                return r[1] == "1"
    except Exception:
        pass
    return False


def maybe_normalize():
    """유효 200 도달 시 1회만: 분포→백분위 일괄 기입(batch)→잠금."""
    if norm_locked() or valid_count() < NORM_THRESHOLD:
        return
    from gspread.utils import rowcol_to_a1

    ws = get_ws(RESP_WS, RESP_HEADER)
    rows = ws.get_all_values()
    header, data = rows[0], rows[1:]
    if not data:
        return
    i_valid = header.index("valid_flag")
    i_raw = header.index("aiq_raw")
    i_pct = header.index("aiq_pct")
    i_lbl = header.index("pct_label")

    raws = [float(r[i_raw]) for r in data
            if len(r) > i_raw and r[i_valid] == "유효" and r[i_raw] not in ("", None)]
    if not raws:
        return
    raws_sorted = sorted(raws)
    n = len(raws_sorted)

    # 행 순서대로 백분위/문구 열 값 구성(무효 행은 공백 유지)
    pct_col, lbl_col = [], []
    for r in data:
        if len(r) > i_raw and r[i_valid] == "유효" and r[i_raw] not in ("", None):
            v = float(r[i_raw])
            le = sum(1 for x in raws_sorted if x <= v)
            pct = round(le / n * 100)
            top = max(1, round((100 - pct) / 10) * 10)  # 상위 N%(10%p 반올림)
            pct_col.append([pct])
            lbl_col.append([f"상위 약 {top}% 안에 포함됩니다"])
        else:
            pct_col.append([""])
            lbl_col.append([""])

    last_row = 1 + len(data)  # 데이터는 시트 2행부터
    rng_pct = f"{rowcol_to_a1(2, i_pct + 1)}:{rowcol_to_a1(last_row, i_pct + 1)}"
    rng_lbl = f"{rowcol_to_a1(2, i_lbl + 1)}:{rowcol_to_a1(last_row, i_lbl + 1)}"

    # 단일 batch 요청(호출 1~2회)
    ws.batch_update(
        [{"range": rng_pct, "values": pct_col},
         {"range": rng_lbl, "values": lbl_col}],
        value_input_option="USER_ENTERED",
    )

    mws = get_ws(META_WS, ["key", "value"])
    mws.append_row(["norm_locked", "1"])
    mws.append_row(["norm_n", str(n)])
    mws.append_row(["norm_dist", json.dumps(raws_sorted)])  # 운영모드 백분위 산출 기준
    mws.append_row(["norm_ts", now_kst().strftime("%Y-%m-%d %H:%M:%S")])


# ── 운영 모드(잠금 후): 임의 raw → 백분위 ───────────────────────────────────────
def get_meta_dict():
    try:
        ws = get_ws(META_WS, ["key", "value"])
        return {r[0]: r[1] for r in ws.get_all_values()[1:] if len(r) >= 2}
    except Exception:
        return {}


def get_norm_dist():
    d = get_meta_dict().get("norm_dist")
    try:
        return json.loads(d) if d else None
    except Exception:
        return None


def raw_to_pct(raw, dist):
    n = len(dist)
    le = sum(1 for x in dist if x <= raw)
    pct = round(le / n * 100)
    top = max(1, round((100 - pct) / 10) * 10)
    return pct, f"상위 약 {top}% 안에 포함됩니다"


# ── 결과 메일 ──────────────────────────────────────────────────────────────────
def send_result_email(to_email, pct_label, type_display, comment):
    sender = st.secrets["EMAIL_ADDRESS"]
    pw = st.secrets["EMAIL_APP_PW"]
    lines = [
        "AIQ 진단 결과 안내",
        "",
        f"· 상대 위치: {pct_label}",
        f"· 유형 경향: {type_display}",
    ]
    if comment:
        lines += ["", f"· 행동 근거: {comment}"]
    lines += ["", "※ 부산대 직원 표본 기준 상대 위치이며, 점수는 구간으로 해석하시기 바랍니다.",
              "※ 본 결과는 잠정(베타)입니다."]
    msg = MIMEText("\n".join(lines), _charset="utf-8")
    msg["Subject"] = "[AIQ] 진단 결과 안내"
    msg["From"] = formataddr(("AIQ 진단", sender))
    msg["To"] = to_email
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as s:
        s.starttls()
        s.login(sender, pw)
        s.sendmail(sender, [to_email], msg.as_string())


# ── 최초 200 일괄 발송(관리자 1클릭·중복방지·재실행 가능) ──────────────────────
def send_pending_results(progress=None):
    """result_sent != '1' 이고 백분위가 있는 유효 행을 id_map 이메일로 발송, 발송분 표시."""
    import time
    from gspread.utils import rowcol_to_a1

    rws = get_ws(RESP_WS, RESP_HEADER)
    rrows = rws.get_all_values()
    rhead, rdata = rrows[0], rrows[1:]
    i_serial = rhead.index("serial")
    i_valid = rhead.index("valid_flag")
    i_pct = rhead.index("aiq_pct")
    i_lbl = rhead.index("pct_label")
    i_type = rhead.index("type_display")
    i_cmt = rhead.index("comment")
    i_sent = rhead.index("result_sent")

    # serial → email
    iws = get_ws(IDMAP_WS, IDMAP_HEADER)
    email_of = {r[0]: r[2] for r in iws.get_all_values()[1:] if len(r) >= 3}

    targets = []  # (row_idx_in_data, serial, email, pct_label, type_display, comment)
    for idx, r in enumerate(rdata):
        if (r[i_valid] == "유효" and r[i_pct] not in ("", None)
                and r[i_sent] != "1" and email_of.get(r[i_serial])):
            targets.append((idx, r[i_serial], email_of[r[i_serial]],
                            r[i_lbl], r[i_type], r[i_cmt]))

    total = len(targets)
    sent_rows = []
    for k, (idx, serial, email, label, tdisp, cmt) in enumerate(targets):
        try:
            send_result_email(email, label, tdisp, cmt)
            sent_rows.append(idx)
        except Exception:
            pass  # 실패분은 result_sent 미표시 → 다음 실행에서 재시도
        # 20건마다 발송표시 flush(중단되어도 그만큼은 보존)
        if len(sent_rows) >= 20:
            _flush_sent(rws, i_sent, sent_rows, rowcol_to_a1)
            sent_rows = []
        if progress and total:
            progress.progress(min(1.0, (k + 1) / total), text=f"발송 {k+1}/{total}")
        time.sleep(0.4)  # Gmail 스로틀 회피
    if sent_rows:
        _flush_sent(rws, i_sent, sent_rows, rowcol_to_a1)
    return total


def _flush_sent(ws, i_sent, row_idxs, rowcol_to_a1):
    """data 인덱스 목록의 result_sent 셀을 '1'로 batch 표시."""
    reqs = []
    for idx in row_idxs:
        cell = rowcol_to_a1(idx + 2, i_sent + 1)
        reqs.append({"range": f"{cell}:{cell}", "values": [["1"]]})
    if reqs:
        ws.batch_update(reqs, value_input_option="USER_ENTERED")


# ──────────────────────────────────────────────────────────────────────────────
# 결과 화면 (콜드스타트: 유형·근거만 / 점수는 보류 안내)
# ──────────────────────────────────────────────────────────────────────────────
def render_done():
    ss = st.session_state
    sc = ss.scored
    l1 = ss.l1_result
    st.success(f"✅ 응답이 저장되었습니다 · 시리얼 {ss.serial}")
    st.markdown("### AI 활용 성향 진단 결과")

    # 유형 카드(경향)
    st.markdown(f"#### 유형 경향")
    st.markdown(f"**{l1['type_display']}**")
    st.caption("고정된 분류가 아니라 현재 경향이며, 반복할수록 이동을 볼 수 있습니다.")

    prof = TYPE_PROFILES.get(l1["primary"]) if l1.get("primary") else None
    if prof:
        st.markdown(f"> {prof['one_line']}")
        for h, body in prof["sections"].items():
            st.markdown(f"**{h}**")
            st.write(body)

    # 행동 근거 / 판단보류
    st.markdown("---")
    if sc["mti_status"] == "scored":
        if sc.get("comment"):
            st.markdown("**행동 근거**")
            st.info(sc["comment"])
    elif sc["mti_status"] == "undetermined":
        st.warning("행동 변화를 관측할 만큼 대화가 진행되지 않았습니다. 조금 더 길게 대화해 다시 측정해 보세요.")
        if sc.get("comment"):
            st.caption(sc["comment"])
    else:
        st.warning("채점을 완료하지 못했습니다. 다시 시도해 주세요.")

    # 점수: 잠금 후(백분위 있음)면 크게 표시, 콜드스타트면 보류 안내
    st.markdown("---")
    if ss.get("pct_label"):
        st.markdown("#### AIQ 결과 (상대 위치)")
        st.markdown(
            f"<div style='font-size:30px;font-weight:800;margin:6px 0'>{ss['pct_label']}</div>",
            unsafe_allow_html=True,
        )
        st.caption("부산대 직원 표본 기준 상대 위치 · 점수는 구간으로 해석하세요 · 잠정(베타)")
        st.caption("동일 내용을 입력하신 이메일로도 보냈습니다.")
    else:
        st.caption(
            "AIQ 점수는 부산대 직원 표본이 모두 모인 뒤(잠정·베타) 기준 분포를 만들어 "
            "이메일로 일괄 발송됩니다. 지금은 점수를 표시하지 않습니다."
        )

    if st.button("처음으로 돌아가기"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# 라우팅
# ──────────────────────────────────────────────────────────────────────────────
def render_flow():
    ss = st.session_state
    steps = ss.order  # 예: ["l1","l3"]
    if ss.step < len(steps):
        cur = steps[ss.step]
        if cur == "l1":
            render_l1()
        else:
            render_l3()
        return

    # 두 단계 완료 → 채점·저장(1회)
    if not ss.get("scored"):
        with st.spinner("응답을 채점하고 저장하는 중…"):
            transcript, post_user = build_segments()
            sc = score_dialogue(transcript, post_user)
            ss.scored = sc
            try:
                was_locked = norm_locked()  # 내 저장 전에 이미 잠겼나(=나는 201+)
                pct = label = ""
                sent = ""
                if was_locked and sc["valid"] and sc.get("aiq_raw") is not None:
                    dist = get_norm_dist()
                    if dist:
                        p, l = raw_to_pct(sc["aiq_raw"], dist)
                        ss.aiq_pct, ss.pct_label = p, l
                        pct, label = p, l
                        try:  # 201+ 본인 결과 메일 즉시 발송
                            send_result_email(ss.email, l, ss.l1_result["type_display"], sc.get("comment"))
                            sent = "1"
                        except Exception:
                            sent = ""
                save_response(transcript, sc, pct, label, sent)
                if not was_locked:
                    maybe_normalize()  # 내가 200번째면 여기서 잠금·일괄 백분위 기입
                    if norm_locked() and sc["valid"] and sc.get("aiq_raw") is not None:
                        dist = get_norm_dist()
                        if dist:
                            ss.aiq_pct, ss.pct_label = raw_to_pct(sc["aiq_raw"], dist)
            except Exception as e:
                st.error(f"저장 오류: {e}")
        st.rerun()

    render_done()


def render_admin():
    """사이드바 관리자 영역(secrets의 ADMIN_CODE 일치 시). 최초 200 일괄 발송."""
    admin_code = st.secrets.get("ADMIN_CODE", "")
    with st.sidebar:
        st.markdown("#### 관리자")
        if not admin_code:
            st.caption("ADMIN_CODE 미설정")
            return
        code_in = st.text_input("관리자 코드", type="password", key="admin_code_in")
        if code_in != admin_code:
            return
        try:
            vc = valid_count()
            locked = norm_locked()
        except Exception as e:
            st.error(f"시트 접근 오류: {e}")
            return
        st.metric("유효 응답", f"{vc} / {NORM_THRESHOLD}")
        st.caption(f"정규화 잠금: {'완료' if locked else '대기'}")
        if locked:
            if st.button("미발송 결과 일괄 발송", use_container_width=True):
                prog = st.progress(0.0, text="발송 준비…")
                try:
                    n = send_pending_results(progress=prog)
                    st.success(f"발송 처리 완료 (대상 {n}건). 실패분은 다시 눌러 재시도하세요.")
                except Exception as e:
                    st.error(f"발송 오류: {e} — 다시 눌러 재시도하세요.")
        else:
            st.caption("유효 200 도달 후 일괄 발송 버튼이 활성화됩니다.")


def main():
    st.set_page_config(page_title="AIQ 진단", page_icon="🧭", layout="centered")
    st.markdown(f"<div style='text-align:right;color:#888;font-size:12px'>{APP_VERSION}</div>", unsafe_allow_html=True)
    render_admin()
    ss = st.session_state
    stage = ss.get("stage", "auth")
    if stage == "auth" or not ss.get("verified"):
        render_auth()
    else:
        render_flow()


if __name__ == "__main__":
    main()


