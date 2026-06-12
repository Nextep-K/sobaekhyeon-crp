# =============================================================================
# AIQ 파일럿 — Streamlit 앱
# Version: v10.0 (2026.06) — 프로토타입 전면 개편
#
# 변경 이력:
#   v10.0 — 측정 구조 재설계 (프로토타입)
#          · 1단계: 12문항(유형별 순방향2+역방향1) — 유형 분류 전용, 점수 미산출
#          · 2단계: 시나리오 객관식 폐지 → AI와 열린 대화 (최대 7턴, 조기 종료 가능)
#          · 채점: 대화 종료 후 전체 로그 일괄 LLM 채점 (QLI 4지표 / MTI 3전환차원)
#          · AIQ = 100 + (QLI + MTI − 10) × 5  [공식 유지]
#          · 결과 화면: 점수 카드와 유형 카드 분리
#          · 1단계 무변별 응답(전부 동일값) 플래그 표시
#
#   v9.4 — 관리자 인증을 이메일 OTP 방식으로 교체
#          · ADMIN_EMAILS 허용 목록 + Gmail SMTP 6자리 코드 발송
#          · 코드 유효 10분 / 시도 5회 제한 / 재발송 60초 쿨다운
#          · OTP는 해시로만 세션에 보관 (hmac 상수시간 비교)
#          · 필요 secrets: ADMIN_EMAILS, SMTP_USER, SMTP_PASSWORD
#
#   v9.3 — 앱 내 문항 관리자 화면 추가 (?mode=admin)
#          · ADMIN_PASSWORD secret 인증 (v9.4에서 OTP로 대체)
#          · 문항 추가 / 활성·비활성 토글 / 삭제 (2단계 확인)
#          · 유형별 최소 5문항 제약 위반 시 비활성·삭제 차단
#          · 변경 즉시 캐시 무효화 — 사용자 화면 실시간 반영
#
#   v9.2 — L1 문항 풀 Google Sheets 외부화
#          · questions 탭에서 활성 문항 로드 (no/text/type/reverse/active)
#          · 유형별 5문항 랜덤 추출 → 20문항 세션 고정
#          · 탭 없으면 기존 20문항으로 자동 생성(bootstrap)
#          · 시트 오류·제약 미달 시 코드 내 QUESTIONS_FALLBACK으로 폴백
#
#   v9.1 — 유형 보고서 텍스트를 GitHub MD 파일에서 동적 호출
#          · load_content() — raw URL에서 MD 파일 로드, @st.cache_data(ttl=3600)
#          · parse_content() — MD 파싱 → {(type1,type2): 섹션 dict}
#          · 코드 재배포 없이 MD 파일만 수정하면 텍스트 즉시 반영
#          · 호출 실패 시 코드 내 TYPE_COMBOS/TYPE_DESC fallback 유지
#
#   v9.0 — 서사형 객관식 구조 + v7.4 연결 방식 + 단일 시트 저장
# =============================================================================

import streamlit as st
from openai import OpenAI
import re
import requests
from datetime import datetime
import pytz
import gspread
from google.oauth2.service_account import Credentials
import smtplib
import ssl
import time
import hashlib
import hmac
import secrets as _pysecrets
from email.mime.text import MIMEText

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
# AIQ 20문항 정의 (VF1 확정본) — 시트 로드 실패 시 폴백
# ─────────────────────────────────────────────
QUESTIONS_FALLBACK = [
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
    "상상가형": "AI와의 대화가 지적 유희인 사람. 질문의 수준은 높고 탐색의 폭은 넓지만, 그 풍요로운 대화가 결과물로 착지하지 못한다.",
    "실행형":   "일단 받고, 고치면서 나아가는 사람. 빠르고 실용적이지만, 더 좋은 질문이 더 좋은 결과를 만든다는 사실을 가끔 잊는다.",
    "의존형":   "AI가 주면 받는 사람. AI와 공생하는 게 아니라 AI에 기대고 있는 상태. AI를 깊이 신뢰하지만, 그 신뢰가 무비판적 수용으로 이어진다.",
}

# 4유형 × 4수준 코멘트 + 권고 (VF1 확정본)
# 수준 기준: 17~20=뚜렷 / 13~16=보통 / 9~12=복합 / 5~8=미형성
TYPE_COMMENTS = {
    "설계자형": {
        "뚜렷":  ("AI가 당신을 살짝 두려워할 것 같습니다. 질문하기 전에 이미 답의 윤곽이 머릿속에 있고, AI는 그것을 완성하는 도구에 가깝습니다. 이 관계에서 갑은 당신입니다.",
                  "한 AI에게 초안을, 다른 AI에게 반론을 맡겨보세요. 설계자의 다음 단계는 AI를 병렬로 운용하는 것입니다."),
        "보통":  ("제법 잘 다루고 있습니다. 가끔 AI 답을 그냥 쓰는 순간도 있지만 대체로 당신이 설계하고 AI가 실행합니다. 방향은 맞습니다.",
                  "AI 결과를 받은 후 '이걸 반박해봐'를 한 번 더 시도해보세요. 재구성의 깊이가 달라집니다."),
        "복합":  ("설계자의 DNA는 있습니다. 다만 AI 앞에서 가끔 주도권을 내어주는 경향이 있습니다. 30초가 성향을 바꿉니다.",
                  "질문을 보내기 전 딱 한 줄 — '내가 원하는 최종 형태'를 먼저 적어보세요. 그 한 줄이 설계의 시작입니다."),
        "미형성": ("설계자형과는 아직 거리가 있습니다. 괜찮습니다 — 모든 건축가도 처음엔 벽돌부터 배웠습니다.",
                   "AI에게 '내 질문이 좋은 질문인지 평가해줘'라고 물어보세요. 자기 질문을 의심하는 순간이 설계자의 출발점입니다."),
    },
    "상상가형": {
        "뚜렷":  ("대화창은 역대급으로 흥미롭고 역대급으로 미완성일 것입니다. 질문은 훌륭합니다 — 문제는 그 훌륭한 질문들이 결론 없이 탭을 닫히고 있다는 것입니다.",
                  "대화를 시작할 때 첫 줄에 '이 대화의 결과물은 ___이다'를 적어놓고 시작해보세요. 탐색이 착지로 바뀝니다."),
        "보통":  ("아이디어 생산력은 충분합니다. AI와 나눈 대화 중 절반만 결과물로 만들어도 꽤 생산적인 사람이 됩니다. 지금은 그 절반을 대화 안에 두고 있습니다.",
                  "오늘 AI와 나눈 대화 중 가장 좋은 아이디어 하나를 골라 50자로 요약해보세요. 요약이 되면 결과물이 됩니다."),
        "복합":  ("복잡한 질문을 즐기는 기질이 보입니다. 아직 그 즐거움이 실제 재구성으로 연결되지 않는 경우가 있습니다.",
                  "AI 대화창을 닫기 전에 '내가 오늘 얻은 것 한 가지'를 적는 습관을 만들어보세요. 한 줄이면 충분합니다."),
        "미형성": ("상상가형과는 거리가 있지만 발전 여지가 큽니다. 질문의 폭을 조금만 넓히면 AI가 생각보다 훨씬 흥미로운 상대임을 곧 발견하게 됩니다.",
                   "AI에게 '엉뚱한 질문' 하나를 던져보세요. 정답을 찾는 게 아니라 대화를 즐기는 것이 상상가의 출발점입니다."),
    },
    "실행형": {
        "뚜렷":  ("AI 시대의 장인입니다. 재료를 받으면 곧바로 손이 움직입니다. 입력보다 출력에 강한 유형 — 그 방식이 실제로 잘 작동하고 있습니다.",
                  "같은 주제로 AI에게 '다른 방식의 답'을 한 번 더 요청해보세요. 비교가 질문 설계를 자연스럽게 가르칩니다."),
        "보통":  ("AI 결과를 그냥 쓰지 않고 변환하는 감각이 있습니다. '더 좋은 질문을 했더라면 처음부터 덜 고쳤을 텐데' — 그 순간이 다음 단계의 입구입니다.",
                  "AI 결과를 수정하기 전에 '왜 이 부분이 맞지 않는가'를 한 줄 적어보세요. 수정의 이유가 명확해지면 다음 질문이 달라집니다."),
        "복합":  ("실행 성향이 있지만 AI 결과를 그대로 수용하는 경우도 혼재합니다. 변환 습관을 의식적으로 강화하면 실행형의 강점이 선명해집니다.",
                  "AI 결과를 받은 후 한 문장만 내 언어로 바꿔써보세요. 그 한 문장이 재구성의 시작입니다."),
        "미형성": ("아직 실행형과 거리가 있습니다. AI 결과를 내 상황에 맞게 바꿔보는 시도부터 시작해보세요.",
                   "AI가 준 답변 중 '내 상황과 다른 부분' 하나를 찾아보세요. 차이를 발견하는 순간 실행형이 시작됩니다."),
    },
    "의존형": {
        "뚜렷":  ("AI를 매우 신뢰하는 편입니다 — 어쩌면 지나치게. AI는 훌륭한 조수지만, 당신의 맥락은 당신만 압니다. AI가 모르는 것이 있습니다.",
                  "AI가 준 답변에 '이건 내 상황과 다르다'고 반응하는 연습을 오늘 딱 한 번만 해보세요. 한 번이면 됩니다."),
        "보통":  ("AI에 꽤 의존하는 편입니다. 당신만이 아는 맥락을 AI가 놓치고 있는 부분이 반드시 있습니다. 그 부분을 찾는 것이 시작입니다.",
                  "AI 결과를 붙여넣기 전에 '이게 정말 내가 원하는 말인가'를 3초만 물어보세요. 딱 3초입니다."),
        "복합":  ("의존 성향이 일부 있지만 다른 유형의 기질도 섞여 있습니다. 작은 습관 하나면 전환됩니다.",
                  "AI에게 '이 답변의 한계가 뭐야'라고 한 번 물어보세요. AI 스스로 약점을 말해줍니다 — 그때부터 비판적 사용이 시작됩니다."),
        "미형성": ("의존형 성향이 낮습니다. 이미 어딘가에서 AI를 비판적으로 쓰고 있는 것입니다. 2순위 유형을 확인해보세요.",
                   "그 비판적 순간을 의식적으로 더 자주 만들어보세요. 이미 하고 있는 것을 더 하면 됩니다."),
    },
}

def get_type_level(score: int) -> str:
    """유형 점수(5~20) → 수준 반환"""
    if score >= 17: return "뚜렷"
    elif score >= 13: return "보통"
    elif score >= 9:  return "복합"
    else:             return "미형성"

# 12개 조합 (1순위 × 2순위) — 캐릭터명 + 위트 한 줄
TYPE_COMBOS = {
    ("설계자형", "상상가형"): ("전략가",      "큰 그림을 그리고 새 프레임을 만든다"),
    ("설계자형", "실행형"):   ("지휘관",      "설계하고 직접 전장에서 실행한다"),
    ("설계자형", "의존형"):   ("기획가",      "설계는 내가, 실행은 맡긴다"),
    ("상상가형", "설계자형"): ("탐험가",      "아이디어가 넘치고 가끔 착지도 한다"),
    ("상상가형", "실행형"):   ("몽상 실행가",  "엉뚱한 아이디어를 일단 만들어본다"),
    ("상상가형", "의존형"):   ("브레인스토머", "아이디어는 AI와 함께, 마무리는 내일"),
    ("실행형",   "설계자형"): ("장인",        "만들면서 설계를 배운다"),
    ("실행형",   "상상가형"): ("실험가",      "일단 만들고 엉뚱한 방향으로 튼다"),
    ("실행형",   "의존형"):   ("스프린터",    "빠르게 받고 빠르게 쓴다"),
    ("의존형",   "설계자형"): ("견습생",      "AI를 믿지만 설계 본능이 깨어나는 중"),
    ("의존형",   "상상가형"): ("구경꾼",      "AI가 신기하고 재밌는데 아직 주도권이 없다"),
    ("의존형",   "실행형"):   ("복사기",      "받아서 바로 쓰는데 나름 손은 빠르다"),
}

# ─────────────────────────────────────────────
# 점수 매트릭스 (선택값 → 점수)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 2단계 — AI 열린 대화 설정 (v10.0)
# ─────────────────────────────────────────────
DIALOGUE_TOPIC = "AI 시대의 미래에 일어날 경제적 양극화 문제"
MAX_USER_TURNS = 7   # 사용자 발화 상한
MIN_TURNS_TO_FINISH = 2  # 종료 버튼 활성화 최소 사용자 발화 수

DIALOGUE_SYSTEM_PROMPT = f"""당신은 '{DIALOGUE_TOPIC}'를 함께 논의하는 토론 파트너다.
규칙:
- 한국어로, 3~5문장으로 답한다.
- 사용자의 질문에 충실히 답하되, 매 답변에 사용자가 미처 생각하지 못했을 관점이나 정보를 정확히 1개 포함한다.
- 단정하지 말고 근거와 함께 관점을 제시한다.
- 사용자에게 되묻지 않는다. 답변만 한다."""

SCORING_PROMPT = """당신은 심리측정 전문가다. 아래는 한 사용자가 AI와 '{topic}'를 논의한 대화 전문이다.
사용자의 발화만을 근거로 두 지표를 1~10 정수로 채점하라.

[QLI — 질문 구성력] 사용자의 첫 질문(필요시 1~2번째 발화)을 다음 4개 지표로 평가해 종합:
1. 목적 정향성: 무엇을 얻으려는지 명시했는가 (막연한 질문=低, 원하는 산출·관점 규정=高)
2. 맥락 구조화: 시점·범위·대상·조건을 제공했는가
3. 인지적 부하 설계: AI에게 비교·인과·시나리오 등 구조적 사고를 요구했는가 (단순 정보 요구=低)
4. 주도성: 대화 방향을 사용자가 설계하는가, AI에 판단을 위임하는가

[MTI — 메타인지 전환] 2번째 발화부터, AI 응답이 사용자의 사고를 움직였는가:
- 프레임 전환: AI 답을 받아 질문의 전제·관점 자체를 재설정
- 심화 전환: AI 답의 허점·전제를 파고들어 구체화
- 통합 전환: 이전 답들을 엮어 새로운 질문 구성
- 전환이 빠를수록(적은 턴), 깊을수록 높게. 같은 수준 반복·재진술만 있으면 1~3.
- 사용자 발화가 1개뿐이면 MTI=3 (전환 기회 자체가 관측되지 않음).

반드시 아래 JSON만 출력하라. 다른 텍스트 금지.
{{"qli": <1-10 정수>, "mti": <1-10 정수>, "transition_turn": <전환이 처음 나타난 사용자 발화 번호, 없으면 0>, "comment": "<채점 근거 한두 문장, 행동 기반 언어>"}}

[대화 전문]
{transcript}
"""


def generate_ai_reply(dialogue: list) -> str:
    """대화 이력 기반 AI 응답 생성. 실패 시 안내 문구."""
    try:
        msgs = [{"role": "system", "content": DIALOGUE_SYSTEM_PROMPT}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in dialogue]
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=msgs,
            max_tokens=500, temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "(AI 응답 생성에 실패했습니다. 계속 질문을 이어가거나 논의를 마쳐주세요.)"


def score_dialogue(dialogue: list) -> tuple:
    """
    대화 종료 후 일괄 채점. 반환: (qli, mti, transition_turn, comment)
    실패 시 (5, 5, 0, "") — 중립값 폴백.
    """
    lines, uturn = [], 0
    for m in dialogue:
        if m["role"] == "user":
            uturn += 1
            lines.append(f"[사용자 발화 {uturn}] {m['content']}")
        else:
            lines.append(f"[AI 응답] {m['content']}")
    transcript = "\n".join(lines)
    prompt = SCORING_PROMPT.format(topic=DIALOGUE_TOPIC, transcript=transcript)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(json)?|```", "", raw).strip()
        import json as _json
        data = _json.loads(raw)
        qli = max(1, min(10, int(data.get("qli", 5))))
        mti = max(1, min(10, int(data.get("mti", 5))))
        tt  = max(0, int(data.get("transition_turn", 0)))
        cm  = str(data.get("comment", ""))[:300]
        return qli, mti, tt, cm
    except Exception:
        return 5, 5, 0, ""

def compute_aiq(qli: int, mti: int) -> int:
    """AIQ 지수 산출 — 100 기준, 범위 약 70~150"""
    return max(0, min(200, 100 + (qli + mti - 10) * 5))

# ─────────────────────────────────────────────
# 유형 산출
# ─────────────────────────────────────────────
def compute_type_scores(answers: dict, questions: list) -> dict:
    scores = {"설계자": 0, "상상가": 0, "실행": 0, "의존": 0}
    for (no, text, typ, reverse) in questions:
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
# 유형 보고서 콘텐츠 — GitHub MD 동적 호출
# ─────────────────────────────────────────────
CONTENT_URL = (
    "https://raw.githubusercontent.com/Nextep-K/sobaekhyeon-crp/main/aiq_content_v1.md"
)

@st.cache_data(ttl=3600)
def load_content() -> dict:
    """
    GitHub raw URL에서 MD 파일을 로드하고 파싱하여 반환한다.
    반환값: {(type1, type2): {"name","tagline","quote","quote_attr",
                              "intro","strength","trap","perception","advice"}}
    실패 시 빈 dict — 코드 내 TYPE_COMBOS/TYPE_DESC가 fallback으로 작동.
    """
    try:
        r = requests.get(CONTENT_URL, timeout=5)
        if r.status_code != 200:
            return {}
        return parse_content(r.text)
    except Exception:
        return {}


def parse_content(md: str) -> dict:
    """
    MD 파일 텍스트 → {(type1, type2): sections_dict}

    섹션 헤더 형식: ## 01. 지휘관 · 설계자형 + 실행형
    섹션 블록 형식: **이런 사람입니다** (줄 시작)
    """
    result = {}
    # 각 유형 블록을 ## 헤더로 분리
    blocks = re.split(r"\n## \d+\.", md)
    for block in blocks[1:]:  # 첫 번째는 파일 헤더
        lines = block.strip().split("\n")
        header = lines[0].strip()  # "지휘관 · 설계자형 + 실행형"

        # 캐릭터명 + 유형 조합 파싱
        m = re.match(r"(.+?)\s*·\s*(\S+형)\s*\+\s*(\S+형)", header)
        if not m:
            continue
        char_name = m.group(1).strip()
        type1     = m.group(2).strip()
        type2     = m.group(3).strip()

        body = "\n".join(lines[1:])

        # 위트(tagline) — 첫 번째 > ** 블록
        tagline_m = re.search(r'>\s*\*\*"(.+?)"\*\*', body)
        tagline   = tagline_m.group(1) if tagline_m else ""

        # 명언 — 두 번째 > * 블록
        quote_m   = re.search(r'>\s*\*"(.+?)"\*', body)
        quote     = quote_m.group(1) if quote_m else ""
        # 명언 출처
        quote_attr_m = re.search(r'—\s*(.+?)$', quote, re.M) if quote else None
        if "—" in quote:
            parts      = quote.split("—", 1)
            quote      = parts[0].strip()
            quote_attr = "— " + parts[1].strip()
        else:
            quote_attr = ""

        def extract_section(label: str) -> str:
            """**label** 이후 다음 ** 블록 전까지 텍스트 추출"""
            pattern = rf"\*\*{re.escape(label)}\*\*\n(.*?)(?=\n\*\*|\Z)"
            sm = re.search(pattern, body, re.S)
            return sm.group(1).strip() if sm else ""

        result[(type1, type2)] = {
            "name":        char_name,
            "tagline":     tagline,
            "quote":       quote,
            "quote_attr":  quote_attr,
            "intro":       extract_section("이런 사람입니다"),
            "strength":    extract_section("가장 빛나는 순간"),
            "trap":        extract_section("이 유형의 함정"),
            "perception":  extract_section("다른 사람 눈에 비치는 당신"),
            "advice":      extract_section("지금 필요한 한 가지"),
        }

    return result


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


# ─────────────────────────────────────────────
# L1 문항 풀 — Google Sheets 외부화 (v9.2)
# 탭: questions  ·  열: no / text / type / reverse / active
# ─────────────────────────────────────────────
import random

VALID_TYPES = ("설계자", "상상가", "실행", "의존")
MIN_FWD_PER_TYPE = 2  # 유형별 순방향 최소 (v10.0: 출제 2문항)
MIN_REV_PER_TYPE = 1  # 유형별 역방향 최소 (v10.0: 출제 1문항)
# 세션 출제 = 유형별 (순방향 2 + 역방향 1) = 총 12문항. 유형 분류 전용, 점수 미산출.


def _to_bool(v) -> bool:
    """시트 셀 값을 bool로 정규화 (TRUE/1/y/yes 허용)"""
    return str(v).strip().upper() in ("TRUE", "1", "Y", "YES")


def _bootstrap_questions_sheet(ss) -> None:
    """questions 탭이 없으면 폴백 20문항으로 생성한다."""
    ws = ss.add_worksheet(title="questions", rows=200, cols=5)
    ws.append_row(["no", "text", "type", "reverse", "active"])
    rows = [[no, text, typ, str(rev).upper(), "TRUE"]
            for (no, text, typ, rev) in QUESTIONS_FALLBACK]
    ws.append_rows(rows)


@st.cache_data(ttl=300)
def load_question_pool() -> list | None:
    """
    questions 탭에서 활성 문항을 로드한다.
    반환: [(no:int, text:str, type:str, reverse:bool), ...] 또는 실패 시 None
    - 탭이 없으면 폴백 문항으로 자동 생성 후 그것을 반환
    - no 중복 시 첫 번째만 유지
    - 캐시 ttl=300 — 시트 수정 후 최대 5분 내 반영
    """
    try:
        gc = get_gsheet_client()
        ss = gc.open(SHEET_NAME)
        try:
            ws = ss.worksheet("questions")
        except gspread.WorksheetNotFound:
            _bootstrap_questions_sheet(ss)
            ws = ss.worksheet("questions")

        records = ws.get_all_records()
        pool, seen = [], set()
        for r in records:
            if not _to_bool(r.get("active", "")):
                continue
            try:
                no = int(r.get("no", 0))
            except (TypeError, ValueError):
                continue
            text = str(r.get("text", "")).strip()
            typ  = str(r.get("type", "")).strip()
            if no <= 0 or no in seen or not text or typ not in VALID_TYPES:
                continue
            seen.add(no)
            pool.append((no, text, typ, _to_bool(r.get("reverse", ""))))
        return pool if pool else None
    except Exception:
        return None


def validate_pool(pool: list) -> bool:
    """유형별 순방향 ≥2, 역방향 ≥1 확보 여부 검증 (v10.0)"""
    if not pool:
        return False
    fwd = {t: 0 for t in VALID_TYPES}
    rev = {t: 0 for t in VALID_TYPES}
    for (_, _, typ, reverse) in pool:
        (rev if reverse else fwd)[typ] += 1
    return all(fwd[t] >= MIN_FWD_PER_TYPE and rev[t] >= MIN_REV_PER_TYPE
               for t in VALID_TYPES)


def build_question_set() -> list:
    """
    문항 풀에서 유형별 순방향 2 + 역방향 1 랜덤 추출 → 셔플하여 12문항 반환.
    풀 로드 실패 또는 제약 미달 시 QUESTIONS_FALLBACK 반환.
    호출 측에서 session_state에 1회 저장하여 세션 내 고정해야 한다.
    """
    pool = load_question_pool()
    if not pool or not validate_pool(pool):
        pool = list(QUESTIONS_FALLBACK)
    fwd = {t: [] for t in VALID_TYPES}
    rev = {t: [] for t in VALID_TYPES}
    for q in pool:
        (rev if q[3] else fwd)[q[2]].append(q)
    selected = []
    for t in VALID_TYPES:
        selected.extend(random.sample(fwd[t], MIN_FWD_PER_TYPE))
        selected.extend(random.sample(rev[t], MIN_REV_PER_TYPE))
    random.shuffle(selected)
    return selected


# ─────────────────────────────────────────────
# 문항 관리자 — 시트 쓰기 함수 (v9.3)
# 관리자 화면은 캐시를 거치지 않고 항상 시트를 직접 읽는다.
# ─────────────────────────────────────────────
def _questions_ws():
    gc = get_gsheet_client()
    ss = gc.open(SHEET_NAME)
    try:
        return ss.worksheet("questions")
    except gspread.WorksheetNotFound:
        _bootstrap_questions_sheet(ss)
        return ss.worksheet("questions")


def admin_fetch_records() -> list:
    """관리자용 — 비캐시 전체 행 조회. 반환 행 순서 = 시트 행 순서."""
    return _questions_ws().get_all_records()


def _active_count_by_type(records: list) -> dict:
    """유형별 (순방향, 역방향) 활성 문항 수"""
    counts = {t: [0, 0] for t in VALID_TYPES}  # [fwd, rev]
    for r in records:
        typ = str(r.get("type", "")).strip()
        if typ in VALID_TYPES and _to_bool(r.get("active", "")):
            idx = 1 if _to_bool(r.get("reverse", "")) else 0
            counts[typ][idx] += 1
    return counts


def can_remove_from_active(records: list, no) -> tuple:
    """
    활성 문항을 비활성/삭제해도 방향별 최소 제약(순방향 2·역방향 1)이 유지되는지 검사.
    반환: (가능 여부, 사유 메시지)
    비활성 문항은 항상 제거 가능.
    """
    target = next((r for r in records if str(r.get("no")) == str(no)), None)
    if target is None:
        return False, "해당 번호의 문항이 없습니다."
    if not _to_bool(target.get("active", "")):
        return True, ""
    typ = str(target.get("type", "")).strip()
    is_rev = _to_bool(target.get("reverse", ""))
    counts = _active_count_by_type(records)
    fwd, rev = counts.get(typ, [0, 0])
    if is_rev and rev <= MIN_REV_PER_TYPE:
        return False, f"'{typ}' 유형 역방향 활성 문항이 최소치({MIN_REV_PER_TYPE}개)입니다. 먼저 같은 유형의 역방향 문항을 추가하세요."
    if (not is_rev) and fwd <= MIN_FWD_PER_TYPE:
        return False, f"'{typ}' 유형 순방향 활성 문항이 최소치({MIN_FWD_PER_TYPE}개)입니다. 먼저 같은 유형의 순방향 문항을 추가하세요."
    return True, ""


def admin_add_question(text: str, typ: str, reverse: bool) -> int:
    """새 문항 추가. no는 자동 채번(최대값+1). 반환: 부여된 no."""
    ws = _questions_ws()
    records = ws.get_all_records()
    nos = []
    for r in records:
        try:
            nos.append(int(r.get("no", 0)))
        except (TypeError, ValueError):
            continue
    new_no = (max(nos) + 1) if nos else 1
    ws.append_row([new_no, text.strip(), typ, "TRUE" if reverse else "FALSE", "TRUE"])
    st.cache_data.clear()
    return new_no


def admin_set_active(no, active: bool) -> bool:
    """활성/비활성 토글. active 열은 5번째 열."""
    ws = _questions_ws()
    records = ws.get_all_records()
    for i, r in enumerate(records):
        if str(r.get("no")) == str(no):
            ws.update_cell(i + 2, 5, "TRUE" if active else "FALSE")
            st.cache_data.clear()
            return True
    return False


def admin_delete_question(no) -> bool:
    """문항 행 삭제. 헤더가 1행이므로 레코드 i번째 = 시트 i+2행."""
    ws = _questions_ws()
    records = ws.get_all_records()
    for i, r in enumerate(records):
        if str(r.get("no")) == str(no):
            ws.delete_rows(i + 2)
            st.cache_data.clear()
            return True
    return False


# ─────────────────────────────────────────────
# 관리자 이메일 OTP 인증 (v9.4)
# secrets: ADMIN_EMAILS(쉼표 구분 허용 목록), SMTP_USER, SMTP_PASSWORD
# ─────────────────────────────────────────────
OTP_TTL_SEC      = 600   # 코드 유효 10분
OTP_MAX_ATTEMPTS = 5     # 검증 시도 한도
OTP_RESEND_SEC   = 60    # 재발송 쿨다운


def _admin_email_allowlist() -> list:
    raw = st.secrets.get("ADMIN_EMAILS", "")
    return [e.strip().lower() for e in str(raw).split(",") if e.strip()]


def _send_otp_email(to_email: str, code: str) -> bool:
    """Gmail SMTP(587/STARTTLS)로 6자리 코드 발송. 실패 시 False."""
    user = st.secrets.get("SMTP_USER", "")
    pw   = st.secrets.get("SMTP_PASSWORD", "")
    if not user or not pw:
        return False
    body = (
        f"AIQ 관리자 인증 코드: {code}\n\n"
        f"10분 내에 입력하세요.\n"
        f"본인이 요청하지 않았다면 이 메일을 무시하세요."
    )
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = "AIQ 관리자 인증 코드"
    msg["From"]    = user
    msg["To"]      = to_email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(user, pw)
            s.send_message(msg)
        return True
    except Exception:
        return False


def _store_otp(email: str, code: str) -> None:
    """OTP는 원문이 아닌 SHA-256 해시로만 세션에 보관."""
    st.session_state.otp_hash      = hashlib.sha256(code.encode()).hexdigest()
    st.session_state.otp_email     = email
    st.session_state.otp_expires   = time.time() + OTP_TTL_SEC
    st.session_state.otp_attempts  = 0
    st.session_state.otp_last_sent = time.time()


def _clear_otp() -> None:
    for k in ("otp_hash", "otp_email", "otp_expires", "otp_attempts"):
        st.session_state.pop(k, None)


def _verify_otp(entered: str) -> tuple:
    """반환: (성공 여부, 실패 사유)"""
    if not st.session_state.get("otp_hash"):
        return False, "발송된 코드가 없습니다. 먼저 코드를 발송하세요."
    if time.time() > st.session_state.get("otp_expires", 0):
        _clear_otp()
        return False, "코드가 만료되었습니다(10분). 다시 발송하세요."
    if st.session_state.get("otp_attempts", 0) >= OTP_MAX_ATTEMPTS:
        _clear_otp()
        return False, "시도 횟수를 초과했습니다. 다시 발송하세요."
    st.session_state.otp_attempts = st.session_state.get("otp_attempts", 0) + 1
    h = hashlib.sha256(entered.strip().encode()).hexdigest()
    if hmac.compare_digest(h, st.session_state.otp_hash):
        _clear_otp()
        return True, ""
    remain = OTP_MAX_ATTEMPTS - st.session_state.otp_attempts
    return False, f"코드가 일치하지 않습니다. (남은 시도 {remain}회)"


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
    .aiq-value { font-size:clamp(96px, 24vw, 160px); font-weight:700; color:#1D6FA8;
                 line-height:1; margin:0; letter-spacing:-4px; }
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
        "questions":     [],    # 세션 고정 문항 세트 (v9.2)
        "answers":       {},    # 12문항 응답
        "type_scores":   {},
        "type1":         "",
        "type2":         "",
        # 2단계 대화 (v10.0)
        "dialogue":        [],
        "transition_turn": 0,
        "score_comment":   "",
        "low_variance":    False,
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
    steps = [(1,"유형 진단"), (2,"AI 대화"), (3,"결과")]
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

- **12개 행동 문항**으로 AI 협업 사고 유형을 분류합니다
- **AI와의 열린 대화**(최대 7턴)로 질문 설계력과 사고 전환 점수를 산출합니다
- 소요 시간: 약 7~10분
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
        st.session_state.questions     = build_question_set()  # 세션 고정 (v9.2)
        st.session_state.stage = 1
        st.rerun()

# ─────────────────────────────────────────────
# stage_1: 20문항 유형 진단
# ─────────────────────────────────────────────
def stage_1():
    step_bar(1)
    st.markdown("### AI와 나는 어떻게 함께 사고하는가")
    st.caption("12개 행동 문항에 응답해주세요. 정답이 없으며 평소 습관을 기준으로 선택하세요.")
    st.divider()

    SCALE = ["① 거의 안 그렇다", "② 가끔 그렇다", "③ 자주 그렇다", "④ 항상 그렇다"]

    # 세션 고정 문항 (직접 URL 진입 등으로 비어 있으면 즉시 구성)
    if not st.session_state.questions:
        st.session_state.questions = build_question_set()
    questions = st.session_state.questions

    with st.form("q_form"):
        answers = {}
        for (no, text, typ, reverse) in questions:
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
        st.caption(f"{answered} / {len(questions)} 문항 완료")
        submitted = st.form_submit_button("✅ 완료 — AI 대화로 이동", use_container_width=True)

    if submitted:
        missing = [no for no, v in answers.items() if v is None]
        if missing:
            st.warning(f"아직 응답하지 않은 문항: Q{', Q'.join(f'{n:02d}' for n in missing)}")
        else:
            st.session_state.answers      = answers
            st.session_state.type_scores  = compute_type_scores(answers, questions)
            st.session_state.low_variance = (len(set(answers.values())) == 1)
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
    st.markdown("### AI와의 열린 대화")
    st.markdown(f"""
<div class="scn-box"><p class="scn-title">논의 주제</p>
<p class="scn-text">AI와 함께 <strong>{DIALOGUE_TOPIC}</strong>에 대해 논의합니다.<br>
AI에게 가장 먼저 어떤 질문을 하시겠습니까? 자유롭게 작성해 주세요.</p></div>
""", unsafe_allow_html=True)
    st.caption(f"대화는 최대 {MAX_USER_TURNS}번까지 이어갈 수 있고, 충분하다고 느끼면 언제든 마칠 수 있습니다.")
    st.divider()

    dialogue = st.session_state.dialogue
    user_turns = sum(1 for m in dialogue if m["role"] == "user")

    # 대화 이력 표시
    for m in dialogue:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    # 입력 (상한 도달 전까지)
    if user_turns < MAX_USER_TURNS:
        user_msg = st.chat_input(
            "첫 질문을 입력하세요" if user_turns == 0 else "후속 질문 또는 의견을 입력하세요"
        )
        if user_msg and user_msg.strip():
            dialogue.append({"role": "user", "content": user_msg.strip()})
            with st.chat_message("user"):
                st.markdown(user_msg.strip())
            with st.spinner("AI가 답변 중..."):
                reply = generate_ai_reply(dialogue)
            dialogue.append({"role": "assistant", "content": reply})
            st.session_state.dialogue = dialogue
            st.rerun()
    else:
        st.info(f"최대 {MAX_USER_TURNS}턴에 도달했습니다. 논의를 마치고 결과를 확인하세요.")

    # 종료 버튼 — 최소 발화 수 충족 시
    if user_turns >= MIN_TURNS_TO_FINISH:
        st.divider()
        if st.button("논의 마치고 결과 보기 →", type="primary", use_container_width=True):
            with st.spinner("응답을 분석 중입니다..."):
                qli, mti, tt, cm = score_dialogue(st.session_state.dialogue)
            st.session_state.qli_score       = qli
            st.session_state.mti_score       = mti
            st.session_state.aiq_index       = compute_aiq(qli, mti)
            st.session_state.transition_turn = tt
            st.session_state.score_comment   = cm
            st.session_state.stage = 3
            st.rerun()
    elif user_turns == 1:
        st.caption("AI의 답변을 보고 한 번 이상 더 이어가면 논의를 마칠 수 있습니다.")

# ─────────────────────────────────────────────
# stage_3: 결과 리포트
# ─────────────────────────────────────────────
def stage_3():
    type1     = st.session_state.type1
    type2     = st.session_state.type2
    type_sc   = st.session_state.type_scores
    qli       = st.session_state.qli_score
    mti       = st.session_state.mti_score
    aiq       = st.session_state.aiq_index

    # 저장 — 1회만
    if not st.session_state.save_attempted:
        st.session_state.save_attempted = True
        dlg = st.session_state.dialogue
        first_q  = next((m["content"] for m in dlg if m["role"] == "user"), "")[:500]
        u_turns  = sum(1 for m in dlg if m["role"] == "user")
        import json as _json
        transcript_json = _json.dumps(dlg, ensure_ascii=False)[:40000]
        # 시트 열 재활용: q1=첫 질문 / q1f=발화 수 / q2a=전환 턴 / q2f_text=대화 전문(JSON)
        serial = save_result(
            name=st.session_state.user_name,
            birth=st.session_state.user_birth,
            type1=type1, type2=type2,
            type_scores=type_sc,
            q1=first_q,
            q1f=str(u_turns),
            q2a=str(st.session_state.transition_turn),
            q2b="",
            q2f="",
            q2f_text=transcript_json,
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
        <span><strong>버전</strong>AIQ v10.0 · 프로토타입</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # MD 콘텐츠 로드 (캐시됨, 실패 시 빈 dict)
    content = load_content()
    combo_data = content.get((type1, type2), None)

    # 캐릭터명 + 위트 — MD 우선, fallback은 TYPE_COMBOS
    combo_fallback = TYPE_COMBOS.get((type1, type2), None)
    if combo_data:
        combo_name = combo_data["name"]
        combo_desc = combo_data["tagline"]
    elif combo_fallback:
        combo_name = combo_fallback[0]
        combo_desc = combo_fallback[1]
    else:
        combo_name = type1.replace("형", "")
        combo_desc = TYPE_DESC.get(type1, "")

    # 점수 카드 — 점수만 단독·대형 표시 (v10.0)
    st.markdown(f'''
    <div class="aiq-hero">
      <p class="aiq-label">AIQ</p>
      <p class="aiq-value">{aiq}</p>
    </div>
    ''', unsafe_allow_html=True)

    # 유형 카드 — 점수와 분리 (v10.0)
    st.markdown(f'''
    <div style="text-align:center; padding:1.25rem 1rem; background:#FFFFFF;
                border:1px solid #E5E7EB; border-radius:12px; margin:0 0 1.5rem;">
      <span class="aiq-badge" style="font-size:14px; padding:5px 16px;">{combo_name}</span>
      <p style="margin:.7rem 0 0; font-size:13px; color:#6B7280;">"{combo_desc}"</p>
    </div>
    ''', unsafe_allow_html=True)

    # 무변별 응답 경고 (v10.0)
    if st.session_state.get("low_variance", False):
        st.warning("모든 문항에 동일한 응답을 하셨습니다. 유형 판정의 신뢰도가 낮으므로 참고용으로만 활용하세요.")

    # 유형 설명
    axis = TYPE_AXIS.get(type1, ("",""))
    st.markdown(f'<h2 class="section-title">{combo_name}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="axis-tag">{type1} · {axis[0]} {axis[1]} — 1순위 &nbsp;|&nbsp; {type2} — 2순위</p>', unsafe_allow_html=True)
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

    if combo_data:
        # ─── MD 콘텐츠 풀버전 ───
        # 명언 블록
        if combo_data.get("quote"):
            st.markdown(f'''
            <div style="background:var(--color-background-secondary);
                        border-left:3px solid var(--color-border-info);
                        border-radius:0;padding:.85rem 1rem;margin:0 0 1rem;">
              <p style="font-size:14px;font-weight:500;color:var(--color-text-primary);
                        margin:0 0 .25rem;line-height:1.5;">"{combo_data["quote"]}"</p>
              <p style="font-size:12px;color:var(--color-text-secondary);
                        margin:0;font-style:italic;">{combo_data.get("quote_attr","")}</p>
            </div>
            ''', unsafe_allow_html=True)

        # 5개 섹션
        sections = [
            ("이런 사람입니다",        "intro"),
            ("가장 빛나는 순간",       "strength"),
            ("이 유형의 함정",         "trap"),
            ("다른 사람 눈에 비치는 당신", "perception"),
        ]
        for title, key in sections:
            text = combo_data.get(key, "")
            if text:
                st.markdown(f'''
                <div style="margin:0 0 1rem;">
                  <p style="font-size:13px;font-weight:500;
                            color:var(--color-text-primary);margin:0 0 .3rem;">{title}</p>
                  <p style="font-size:13px;color:var(--color-text-secondary);
                            line-height:1.75;margin:0;">{text}</p>
                </div>
                ''', unsafe_allow_html=True)

        # 지금 필요한 한 가지 — 파란 박스
        advice = combo_data.get("advice", "")
        if advice:
            st.markdown(f'''
            <div style="background:var(--color-background-info);
                        border-radius:var(--border-radius-md);
                        padding:.85rem 1rem;margin:0 0 1rem;">
              <p style="font-size:11px;font-weight:500;color:var(--color-text-info);
                        letter-spacing:.06em;text-transform:uppercase;margin:0 0 .3rem;">
                💡 지금 필요한 한 가지</p>
              <p style="font-size:13px;color:var(--color-text-info);
                        line-height:1.7;margin:0;">{advice}</p>
            </div>
            ''', unsafe_allow_html=True)

    else:
        # ─── fallback: 기존 TYPE_COMMENTS 코멘트 박스 ───
        type1_score = st.session_state.type_scores.get(
            next((k for k, v in TYPE_LABELS.items() if v == type1), ""), 0
        )
        level = get_type_level(type1_score)
        if type1 in TYPE_COMMENTS and level in TYPE_COMMENTS[type1]:
            comment, advice = TYPE_COMMENTS[type1][level]
            st.markdown(f'''
            <div style="background:var(--color-background-secondary);
                        border:0.5px solid var(--color-border-tertiary);
                        border-radius:var(--border-radius-md);
                        padding:.9rem 1rem;margin-bottom:.75rem;
                        font-size:13px;color:var(--color-text-secondary);line-height:1.7;">
                {comment}
            </div>
            <div style="background:var(--color-background-info);
                        border-radius:var(--border-radius-md);
                        padding:.75rem 1rem;font-size:13px;
                        color:var(--color-text-info);line-height:1.7;">
                💡 {advice}
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
# 관리자 화면 (v9.3) — URL: ?mode=admin
# ─────────────────────────────────────────────
def render_admin():
    st.markdown("## AIQ 문항 관리 (관리자)")

    allow = _admin_email_allowlist()
    if not allow:
        st.error("ADMIN_EMAILS secret이 설정되지 않았습니다. (쉼표 구분 이메일 목록)")
        st.stop()
    if not (st.secrets.get("SMTP_USER", "") and st.secrets.get("SMTP_PASSWORD", "")):
        st.error("SMTP_USER / SMTP_PASSWORD secret이 설정되지 않았습니다.")
        st.stop()

    # 인증 — 이메일 OTP (v9.4)
    if not st.session_state.get("admin_authed", False):
        st.markdown("#### 관리자 이메일 인증")
        email = st.text_input("관리자 이메일", key="adm_email_in",
                              placeholder="등록된 관리자 이메일 입력")

        c1, c2 = st.columns([1, 1])
        if c1.button("인증 코드 발송", type="primary"):
            em = email.strip().lower()
            if em not in allow:
                st.error("등록되지 않은 관리자 이메일입니다.")
            elif time.time() - st.session_state.get("otp_last_sent", 0) < OTP_RESEND_SEC:
                wait = int(OTP_RESEND_SEC - (time.time() - st.session_state.get("otp_last_sent", 0)))
                st.warning(f"재발송은 {wait}초 후에 가능합니다.")
            else:
                code = f"{_pysecrets.randbelow(1000000):06d}"
                if _send_otp_email(em, code):
                    _store_otp(em, code)
                    st.success("인증 코드를 발송했습니다. 메일함(스팸함 포함)을 확인하세요.")
                else:
                    st.error("메일 발송 실패 — SMTP_USER / SMTP_PASSWORD 설정을 확인하세요.")

        if st.session_state.get("otp_hash"):
            entered = st.text_input("6자리 인증 코드", max_chars=6, key="adm_otp_in")
            if c2.button("코드 확인"):
                ok, msg = _verify_otp(entered)
                if ok:
                    st.session_state.admin_authed = True
                    st.rerun()
                else:
                    st.error(msg)
        st.stop()

    # 데이터 로드 (비캐시)
    try:
        records = admin_fetch_records()
    except Exception as e:
        st.error(f"시트 연결 실패: {e}")
        st.stop()

    # 유형별 활성 현황
    counts = _active_count_by_type(records)
    cols = st.columns(4)
    for i, t in enumerate(VALID_TYPES):
        fwd, rev = counts[t]
        warn = " ⚠️" if (fwd <= MIN_FWD_PER_TYPE or rev <= MIN_REV_PER_TYPE) else ""
        cols[i].metric(t, f"순{fwd}·역{rev}{warn}")
    if any(counts[t][0] < MIN_FWD_PER_TYPE or counts[t][1] < MIN_REV_PER_TYPE for t in VALID_TYPES):
        st.warning(f"유형별 순방향 {MIN_FWD_PER_TYPE}·역방향 {MIN_REV_PER_TYPE} 미만이면 풀이 사용되지 않고 기본 문항으로 대체됩니다.")
    st.divider()

    # 새 문항 추가
    with st.expander("➕ 새 문항 추가", expanded=False):
        new_text = st.text_input("문항 텍스트", key="adm_new_text")
        c1, c2 = st.columns([1, 1])
        new_type = c1.selectbox("유형", VALID_TYPES, key="adm_new_type")
        new_rev  = c2.checkbox("역코딩 문항", key="adm_new_rev")
        if st.button("추가", key="adm_add_btn"):
            if not new_text.strip():
                st.error("문항 텍스트를 입력하세요.")
            else:
                no = admin_add_question(new_text, new_type, new_rev)
                st.success(f"문항 추가 완료 (no={no})")
                st.rerun()

    st.divider()
    st.markdown(f"**전체 문항 {len(records)}개** · 변경은 사용자 화면에 즉시 반영됩니다.")

    # 문항 목록
    for r in records:
        no_  = r.get("no", "")
        text = str(r.get("text", ""))
        typ  = str(r.get("type", "")).strip()
        rev  = _to_bool(r.get("reverse", ""))
        act  = _to_bool(r.get("active", ""))

        c1, c2, c3, c4, c5 = st.columns([0.6, 5, 1.2, 1.4, 1.2])
        c1.markdown(f"`{no_}`")
        style = "" if act else "color:gray;text-decoration:line-through;"
        rev_tag = " · 역코딩" if rev else ""
        c2.markdown(f"<span style='{style}'>{text}</span><br><small>{typ}{rev_tag}</small>",
                    unsafe_allow_html=True)

        # 활성/비활성 토글
        if act:
            if c3.button("비활성", key=f"adm_off_{no_}"):
                ok, msg = can_remove_from_active(records, no_)
                if ok:
                    admin_set_active(no_, False)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            if c3.button("활성", key=f"adm_on_{no_}"):
                admin_set_active(no_, True)
                st.rerun()

        # 삭제 — 2단계 확인
        pending = st.session_state.get("adm_del_pending")
        if pending == no_:
            if c4.button("⚠️ 확인 삭제", key=f"adm_delc_{no_}", type="primary"):
                ok, msg = can_remove_from_active(records, no_)
                if ok:
                    admin_delete_question(no_)
                    st.session_state.adm_del_pending = None
                    st.rerun()
                else:
                    st.session_state.adm_del_pending = None
                    st.error(msg)
            if c5.button("취소", key=f"adm_delx_{no_}"):
                st.session_state.adm_del_pending = None
                st.rerun()
        else:
            if c4.button("삭제", key=f"adm_del_{no_}"):
                st.session_state.adm_del_pending = no_
                st.rerun()

    st.divider()
    if st.button("로그아웃"):
        st.session_state.admin_authed = False
        st.rerun()


# ─────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────
if st.query_params.get("mode") == "admin":
    render_admin()
else:
    stage = st.session_state.get("stage", 0)
    if   stage == 0: stage_0()
    elif stage == 1: stage_1()
    elif stage == 2: stage_2()
    else:            stage_3()
