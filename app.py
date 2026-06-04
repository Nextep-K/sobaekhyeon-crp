# =============================================================================
# AIQ 파일럿 — Streamlit 앱
# Version: v8.5  (2026.06)
#
# 변경 이력:
#   v8.5 — 저장 채번 로직을 gspread 버전 독립적으로 수정
#          · v8.3의 append_row 반환값 파싱 방식은 gspread 버전마다 다름
#            (일부는 dict, 일부는 requests.Response 반환) → "저장 오류: <Response [200]>"
#          · 수정: append_row 결과 무시, 직후 ws.col_values(1)로 행 수 조회 →
#            마지막 행 번호로 serial 생성 → 첫 셀에 update_cell
#          · 동시성 안전성 유지 (Google Sheets API의 append는 원자적)
#
#   v8.4 — 마지막 턴 LLM 응답 생략, 자동 다음 단계 이동
#   v8.3 — 응답자 식별 체계 전면 개편 (자동 채번, 이름·생년월일, 동의)
#   v8.2 — 9가지 정합성 이슈 수정
#   v8.1 — stage_4 결과 화면 개편
#   v8.0 — v7.4(탭 구조)에서 4단계 화면 전환 구조로 전면 개편
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
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="AIQ 진단", layout="centered", initial_sidebar_state="collapsed")

ENSEMBLE_N           = 3
SHEET_NAME           = "AIQ_Pilot"
KST                  = pytz.timezone("Asia/Seoul")
DIVERGENCE_THRESHOLD = 1.5  # 앙상블 std_dev 임계값 — 초과 시 low_reliability 플래그

# ─────────────────────────────────────────────
# AIQ 20문항 정의 (VF1 확정본)
# ─────────────────────────────────────────────
QUESTIONS = [
    # (번호, 텍스트, 유형, 역방향여부)
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

TYPE_LABELS = {
    "설계자": "설계자형",
    "상상가": "상상가형",
    "실행":   "실행형",
    "의존":   "의존형",
}

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
# Topic 1 · 2 시나리오 정의
# ─────────────────────────────────────────────
TOPIC1_SCENARIO = """당신은 어떤 일을 시작할지 말지 고민 중이다. 믿을 만한 두 사람에게 의견을 구했더니, 한 사람은 "지금이 적기다, 바로 해라"라고 하고, 다른 한 사람은 "지금은 때가 아니다, 기다려라"라고 한다. 두 사람 다 당신을 잘 알고, 둘 다 진심으로 조언하고 있다. 어느 쪽 말도 무시하기 어렵다.

이 상황에서 무엇을 먼저 따져보고 싶은지, AI와 대화하며 정리해 보세요."""

TOPIC2_SCENARIO = """친구가 다음 주 주말에 자기 일을 좀 도와달라고 부탁했고, 당신은 그러기로 했다. 그런데 약속 후에 알게 된 사실이 있다. 그 주말은 당신에게도 중요한 일이 있는 날이었다. 친구는 당신이 도와줄 거라 믿고 이미 다른 준비를 시작했다.

이 상황을 어떻게 풀어갈지, AI와 대화하며 생각을 전개해 보세요."""

TOPIC1_AI_FIRST = "안녕하세요. 이 상황에 대해 가장 먼저 떠오르는 것을 편하게 말해보세요."
TOPIC2_AI_FIRST = "안녕하세요. 이 상황, 어떻게 풀면 좋을지 편하게 말해보세요."

TOPIC2_CONDITION_INJECT = "그런데 친구가 부탁한 그 일이, 사실 당신이 아니어도 할 수 있는 일이라면 어떨까요? 친구가 굳이 당신에게 부탁한 이유가 따로 있을 수도 있는데요."

# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def normalize_score(v: float) -> float:
    if v > 100: v = v / 10
    elif v > 10: v = v / 10
    return max(1.0, min(10.0, round(v, 2)))


def compute_type_scores(answers: dict) -> dict:
    """20문항 응답 → 유형별 점수 산출"""
    scores = {"설계자": 0, "상상가": 0, "실행": 0, "의존": 0}
    for (no, text, typ, reverse) in QUESTIONS:
        val = answers.get(no, 2)
        if reverse:
            val = 5 - val  # 역산
        scores[typ] += val
    return scores


def compute_top_types(type_scores: dict) -> tuple[str, str]:
    """1순위·2순위 유형 반환"""
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    t1 = TYPE_LABELS[sorted_types[0][0]]
    t2 = TYPE_LABELS[sorted_types[1][0]]
    return t1, t2


def compute_aiq_index(qli: float, recon: float) -> int:
    """AIQ 지수 = 100 기준 0~200 (기하평균 잠정 공식)"""
    combined = (qli * recon) ** 0.5
    index = 100 + (combined - 5.0) * 20
    return max(0, min(200, round(index)))


# ─────────────────────────────────────────────
# Google Sheets 연동
# ─────────────────────────────────────────────
#
# ▼ 개발노트 ──────────────────────────────────
#
# 스프레드시트 이름: "AIQ_Pilot" (st.secrets["gcp_service_account"] 서비스 계정 공유 필요)
#
# ─── 2개 워크시트 분리 구조 (v8.3~) ───
#
# 식별 정보와 진단 결과를 시트로 분리한다. 두 시트는 'serial' 컬럼으로 조인된다.
# 분리 목적: 식별 정보 노출 최소화, 향후 데이터 분석 시 익명 데이터 분리 추출 용이.
#
# ─── 시트 1: "participants" — 식별 정보 (컬럼 5개) ───
#
#  1. serial           — #2026_000001 형식의 자동 채번 ID
#  2. timestamp        — KST 응답 완료 시각 (YYYY-MM-DD HH:MM:SS KST)
#  3. name             — 응답자 이름 (2~20자, 한글/영문)
#  4. birth            — 생년월일 YYYYMMDD 8자리
#  5. consent          — 동의 여부 (Y, 미동의 시 진행 불가하므로 항상 Y)
#
# ─── 시트 2: "responses" — 진단 결과 (컬럼 15개) ───
#
#  1. serial           — participants 시트의 serial과 매칭
#  2. timestamp        — KST 응답 완료 시각 (위와 동일)
#
#  ─── 1단계: 유형 진단 (20문항) ───
#  3. type1            — 1순위 유형 (설계자형/상상가형/실행형/의존형)
#  4. type2            — 2순위 유형
#  5. score_designer   — 설계자형 점수 (5~20, 역방향 Q08 역산 적용)
#  6. score_imaginer   — 상상가형 점수 (5~20, 역방향 Q03 역산 적용)
#  7. score_executor   — 실행형 점수   (5~20, 역방향 Q13 역산 적용)
#  8. score_follower   — 의존형 점수   (5~20, 역방향 Q18 역산 적용)
#
#  ─── 2단계: 시나리오 측정 (앙상블 3회 평균) ───
#  9. QLI              — 질문 설계력 (LP·BF·AE 기하평균, 1~10)
# 10. Recon            — 재구성력   (1~10)
# 11. MTI              — 사고 전환  (3-Layer 가중합, 1~10)
# 12. AIQ_index        — AIQ 지수 (100 기준 0~200, 잠정 환산)
# 13. class_s          — Class S(문제 재정의) 발생 여부 (Y/N)
#
#  ─── 원자료 (분석 재현용) ───
# 14. q_answers        — 20문항 응답 dict 문자열 {1:2, 2:3, ...}
# 15. log_topic1       — Topic 1 대화 로그 (role·content 리스트 문자열)
# 16. log_topic2       — Topic 2 대화 로그 (role·content 리스트 문자열)
#
# ─── 자동 채번 메커니즘 (v8.5) ───
#
# • 채번 시점: 진단 완료 시(save_result 호출 시점) — 중도 이탈자는 번호 미부여
# • 채번 방식:
#   1) participants 시트에 빈 serial로 행을 먼저 append
#   2) 직후 ws.col_values(1)로 첫 컬럼의 전체 길이를 조회 → 마지막 행 번호 획득
#   3) 헤더 행을 제외한 (행번호 - 1)을 6자리 시리얼로 변환
#   4) 그 행의 첫 셀에 update_cell로 시리얼 값을 기록
#   Google Sheets API의 append가 원자적이므로 동시 진입자가 있어도
#   각 호출이 서로 다른 행을 차지한다.
# • 형식: #2026_000001 ~ #2026_999999 (6자리 패딩)
#
# ─── 마이그레이션 주의사항 ───
#
# • v8.2에서 v8.3으로 올라온 경우: 기존 responses 시트 컬럼 구조가 다르므로
#   자동 호환되지 않는다. 다음 중 하나 필요:
#   (a) 기존 데이터 백업 후 responses·participants 시트 모두 삭제
#   (b) 새 스프레드시트로 시작
#
# • 새 컬럼 추가 시: 헤더 리스트와 append_row 데이터 순서 일치시킬 것
#
# • AIQ_index 환산 공식이 바뀌면(파일럿 캘리브레이션 후): compute_aiq_index() 수정.
#   기존 저장값은 잠정 공식 결과이므로 재계산이 필요할 수 있다.
#
# • 채점 LLM 응답이 파싱 실패하면 run_scoring()이 fallback 5.0을 반환하므로,
#   QLI/Recon/MTI가 정확히 5.00·5.00·5.00이고 scoring_failed=Y이면 파싱 실패.
#
# ────────────────────────────────────────────

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


def save_result(name: str, birth: str, type1: str, type2: str, type_scores: dict,
                qli: float, recon: float, mti: float, aiq: int,
                has_class_s: bool, answers: dict,
                log_t1: list, log_t2: list,
                scoring_failed: bool = False) -> str | None:
    """
    진단 결과를 Google Sheets에 저장하고 자동 채번된 serial을 반환한다.

    채번 방식 (v8.5):
      1) participants 시트에 빈 serial로 행 append
      2) ws.col_values(1)로 1번 컬럼 전체 길이 조회 → 마지막 행 번호 획득
      3) (행번호 - 1)을 6자리 패딩한 "#2026_NNNNNN" 형식의 serial 생성
      4) 첫 셀(serial 컬럼)에 update_cell로 값 기록

      Google Sheets API의 append가 원자적이므로 동시 진입자가 있어도
      각 호출이 서로 다른 행을 차지한다. col_values는 append 직후 실행되어
      자신이 방금 추가한 행까지 포함한 정확한 행 수를 반환한다.

    반환값:
      성공: "#2026_000001" 같은 serial 문자열
      실패: None (저장 오류 시)
    """
    try:
        gc = get_gsheet_client()
        ss = gc.open(SHEET_NAME)
        ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")

        # ─── 1) participants 시트에 append (채번) ───
        try:
            ws_p = ss.worksheet("participants")
        except gspread.WorksheetNotFound:
            ws_p = ss.add_worksheet(title="participants", rows=5000, cols=10)
            ws_p.append_row(["serial", "timestamp", "name", "birth", "consent"])

        # 빈 serial로 먼저 append (반환값에 의존하지 않음 — gspread 버전마다 다름)
        ws_p.append_row(
            ["", ts, name, birth, "Y"],
            value_input_option="RAW"
        )

        # append 직후 마지막 행 번호 조회
        # col_values(1)은 1번 컬럼(serial 열)의 모든 값을 가져온다 (빈 값 포함, 마지막 비어있는 행까지)
        col_a = ws_p.col_values(1)
        row_num = len(col_a)  # 1-based: 헤더 포함 마지막 행 번호

        serial_num = row_num - 1  # 헤더 행(1행) 제외
        serial = f"#2026_{serial_num:06d}"

        # 빈 serial 셀에 실제 값 기록
        ws_p.update_cell(row_num, 1, serial)

        # ─── 2) responses 시트에 append ───
        try:
            ws_r = ss.worksheet("responses")
        except gspread.WorksheetNotFound:
            ws_r = ss.add_worksheet(title="responses", rows=5000, cols=20)
            ws_r.append_row([
                "serial", "timestamp",
                "type1", "type2",
                "score_designer", "score_imaginer", "score_executor", "score_follower",
                "QLI", "Recon", "MTI", "AIQ_index", "class_s",
                "q_answers", "log_topic1", "log_topic2"
            ])
        ws_r.append_row([
            serial, ts,
            type1, type2,
            type_scores.get("설계자", 0), type_scores.get("상상가", 0),
            type_scores.get("실행", 0), type_scores.get("의존", 0),
            round(qli, 2), round(recon, 2), round(mti, 2), aiq,
            "Y" if has_class_s else "N",
            str(answers),
            str(log_t1),
            str(log_t2)
        ])
        return serial
    except Exception as e:
        st.warning(f"저장 오류: {e}")
        return None


# ─────────────────────────────────────────────
# AI 대화 응답 생성
# ─────────────────────────────────────────────
def get_ai_response(messages: list, topic: int) -> str:
    """Topic 1·2 대화 진행 AI 응답"""
    system = (
        "당신은 AIQ 진단 시스템의 대화 진행자입니다. "
        "응답자의 생각을 끌어내되 답을 유도하거나 평가하지 마십시오. "
        "짧고 자연스럽게 이어지도록 하십시오(2~3문장 이하). "
        "정답이 있는 것처럼 느끼게 하지 마십시오."
    )
    # Topic 2 조건 투입 — 응답자가 판단을 세운 뒤 1회만 (t2_injected 플래그 활용)
    inject_condition = (
        topic == 2
        and len(messages) >= 3
        and not st.session_state.get("t2_injected", False)
    )
    if inject_condition:
        system += (
            "\n\n[중요] 응답자가 어떤 입장이나 판단을 세웠다면, "
            "이번 응답에 다음 조건을 자연스러운 흐름으로 한 번만 삽입하십시오: "
            f"'{TOPIC2_CONDITION_INJECT}'"
        )
        st.session_state.t2_injected = True
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.3,
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# QLI·MTI·Recon 채점 (앙상블 3회)
# ─────────────────────────────────────────────
RUBRIC_SYSTEM = """You are an AIQ diagnostic scoring system. Score the following conversation logs.

SCORING RUBRIC:

QLI (Question Logic Index) = geometric mean of LP · BF · AE (each 1-10):
  LP (Logical Precision — 논리 정밀도):
    9-10: Explicitly states conditions, premises, constraints
    7-8: Mostly precise but some vagueness
    5-6: Moderate structure
    3-4: Loose, informal
    1-2: Simple information request only
  BF (Breadth Factor — 범위 확장도):
    9-10: Multiple paths, diverse sub-questions explored
    7-8: Some branching attempted
    5-6: Moderate exploration
    3-4: Minimal branching
    1-2: No expansion
  AE (Abstraction Elevation — 추상도 수준):
    9-10: Questions underlying principles, challenges premises
    7-8: Moves toward abstraction partially
    5-6: Mixed concrete and abstract
    3-4: Mostly concrete
    1-2: Purely factual/confirmatory

MTI (Meta-cognitive Tension Index) — 1-10 score (NOT a grade):
  Uses 3-Layer weighted formula: L1×0.60 + L2×0.25 + L3×0.15
  L1 weights: self-negation(A)×3 + strategy revision(B)×2 + reflection(C)×1, normalized 1-10
  Apply speed bonus: multiply by (1 + 1/turns_to_transition) if Class A detected
  L2: quality of monitoring (1-10)
  L3: implicit contextual reasoning (1-10)
  Class S detected (problem reframing "this is not X but Y"): add flag

Recon (Reconfiguration) — 1-10:
  9-10: Creates new logical structure, extends beyond AI output
  7-8: Integrates and attempts synthesis
  5-6: Lists ideas without integration
  3-4: Repeats pattern without reassembly
  1-2: Copies AI output directly

Output ONLY this format:
[SCORES]
LP: <1-10>
BF: <1-10>
AE: <1-10>
QLI: <geometric mean, 1-10>
MTI: <1-10>
Recon: <1-10>
ClassS: <YES/NO>
[INSIGHT_KO]
<3문장 이내. 질문 구조·사고 전환·재구성 관점에서 관찰된 사실만 서술. 권고·평가 금지.>
"""

def parse_aiq_scores(text: str) -> dict | None:
    """채점 응답 파싱"""
    try:
        lp    = float(re.search(r"LP\s*:\s*([\d.]+)", text, re.I).group(1))
        bf    = float(re.search(r"BF\s*:\s*([\d.]+)", text, re.I).group(1))
        ae    = float(re.search(r"AE\s*:\s*([\d.]+)", text, re.I).group(1))
        qli   = float(re.search(r"QLI\s*:\s*([\d.]+)", text, re.I).group(1))
        mti   = float(re.search(r"MTI\s*:\s*([\d.]+)", text, re.I).group(1))
        recon = float(re.search(r"Recon\s*:\s*([\d.]+)", text, re.I).group(1))
        cs    = bool(re.search(r"ClassS\s*:\s*YES", text, re.I))
        ko_m  = re.search(r"\[INSIGHT_KO\](.*?)$", text, re.S | re.I)
        ko    = ko_m.group(1).strip() if ko_m else ""
        return {
            "LP": normalize_score(lp), "BF": normalize_score(bf),
            "AE": normalize_score(ae), "QLI": normalize_score(qli),
            "MTI": normalize_score(mti), "Recon": normalize_score(recon),
            "ClassS": cs, "insight_ko": ko
        }
    except Exception:
        return None


def run_scoring(log_t1: list, log_t2: list) -> dict:
    """앙상블 3회 채점 → 평균 반환"""
    log_t1_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in log_t1])
    log_t2_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in log_t2])
    user_prompt = (
        f"[Topic 1 — 질문 구조 측정]\n{log_t1_text}\n\n"
        f"[Topic 2 — 사고 전환 측정]\n{log_t2_text}"
    )
    results = []
    for _ in range(ENSEMBLE_N):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": RUBRIC_SYSTEM},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            parsed = parse_aiq_scores(resp.choices[0].message.content)
            if parsed:
                results.append(parsed)
        except Exception:
            pass

    if not results:
        # 채점 전면 실패 — 신뢰도 0, fallback 5.0
        return {
            "QLI": 5.0, "MTI": 5.0, "Recon": 5.0,
            "ClassS": False,
            "insight_ko": "채점 오류 — LLM 응답 파싱 실패. 로그를 검토 후 재채점이 필요합니다.",
            "low_reliability": True,
            "scoring_failed": True
        }

    avg_qli   = round(sum(r["QLI"]   for r in results) / len(results), 2)
    avg_mti   = round(sum(r["MTI"]   for r in results) / len(results), 2)
    avg_recon = round(sum(r["Recon"] for r in results) / len(results), 2)
    has_cs    = any(r["ClassS"] for r in results)
    insight   = results[0]["insight_ko"]

    std_mti = (sum((r["MTI"] - avg_mti) ** 2 for r in results) / len(results)) ** 0.5
    low_reliability = std_mti > DIVERGENCE_THRESHOLD

    return {
        "QLI": avg_qli, "MTI": avg_mti, "Recon": avg_recon,
        "ClassS": has_cs, "insight_ko": insight,
        "low_reliability": low_reliability,
        "scoring_failed": False
    }


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 780px; margin: 0 auto; }
    .step-bar { display: flex; align-items: center; gap: 0;
                padding: 0.6rem 0; margin-bottom: 1.5rem;
                border-bottom: 1px solid #e5e7eb; }
    .step-item { display: flex; align-items: center; gap: 6px; flex: 1; }
    .step-num  { width: 22px; height: 22px; border-radius: 50%;
                 display: flex; align-items: center; justify-content: center;
                 font-size: 11px; font-weight: 600; }
    .step-on   { background: #EBF4FF; color: #1D6FA8; border: 1.5px solid #93C5FD; }
    .step-done { background: #ECFDF5; color: #065F46; border: 1.5px solid #6EE7B7; }
    .step-off  { background: #F9FAFB; color: #9CA3AF; border: 1px solid #E5E7EB; }
    .step-lbl-on   { font-size: 12px; font-weight: 600; color: #111827; }
    .step-lbl-off  { font-size: 12px; color: #9CA3AF; }
    .step-div  { flex: none; width: 20px; height: 1px; background: #E5E7EB; margin: 0 2px; }
    .scn-box   { background: #F0F7FF; border-left: 3px solid #3B82F6;
                 border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    .scn-title { font-size: 11px; font-weight: 600; color: #6B7280;
                 text-transform: uppercase; letter-spacing: .06em; margin-bottom: .4rem; }
    .scn-text  { font-size: 14px; color: #111827; line-height: 1.7; margin: 0; }
    .topic-tag { display: inline-block; font-size: 11px; font-weight: 600;
                 padding: 2px 9px; border-radius: 4px; margin-bottom: 0.6rem; }
    .t1-tag    { background: #ECFDF5; color: #065F46; border: 1px solid #6EE7B7; }
    .t2-tag    { background: #FFFBEB; color: #92400E; border: 1px solid #FCD34D; }
    .coord-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 0.75rem 0; }
    .cc        { border: 1px solid #E5E7EB; border-radius: 6px;
                 padding: 8px 10px; font-size: 12px; background: #F9FAFB; }
    .cc-hl     { background: #EBF4FF; border-color: #93C5FD; }
    .cc-name   { font-weight: 600; font-size: 13px; display: block; color: #111827; }
    .cc-tag    { font-size: 11px; color: #6B7280; }
    .cc-hl .cc-name { color: #1D6FA8; }
    .interp    { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;
                 padding: 0.75rem 1rem; font-size: 13px; color: #374151; line-height: 1.6; }

    /* ─── 결과 리포트 (v8.1) ─── */
    .report-header { border-bottom: 2px solid #1F3864; padding-bottom: 0.85rem; margin-bottom: 1.5rem; }
    .report-title  { font-size: 11px; font-weight: 600; color: #1D6FA8;
                     letter-spacing: 0.15em; text-transform: uppercase; margin: 0 0 0.25rem; }
    .report-name   { font-size: 22px; font-weight: 500; color: #111827; margin: 0 0 0.5rem; }
    .report-meta   { display: flex; gap: 1.5rem; font-size: 12px; color: #6B7280; flex-wrap: wrap; }
    .report-meta span strong { color: #374151; font-weight: 500; margin-right: 0.25rem; }

    .aiq-hero      { text-align: center; padding: 2.5rem 1rem 2rem;
                     background: linear-gradient(180deg, #F0F7FF 0%, #FFFFFF 100%);
                     border-radius: 12px; margin: 0 0 1.5rem; border: 1px solid #DBEAFE; }
    .aiq-label     { font-size: 13px; font-weight: 500; color: #6B7280;
                     letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 0.5rem; }
    .aiq-value     { font-size: 96px; font-weight: 300; color: #1D6FA8;
                     line-height: 1; margin: 0; letter-spacing: -2px; }
    .aiq-badge-row { margin-top: 0.75rem; display: flex; justify-content: center;
                     gap: 8px; align-items: center; flex-wrap: wrap; }
    .aiq-badge     { font-size: 11px; font-weight: 500; color: #1D6FA8;
                     background: #DBEAFE; padding: 3px 10px; border-radius: 4px; }

    .section-title { font-size: 24px; font-weight: 500; color: #111827; margin: 1rem 0 0.25rem; }
    .axis-tag      { font-size: 13px; color: #6B7280; margin: 0 0 0.75rem; }
    .type-quote    { border-left: 3px solid #D1D5DB; padding: 0 0 0 1rem;
                     margin: 0.5rem 0 1.5rem; color: #4B5563; font-size: 13px; line-height: 1.7; }

    .sub-section-title { font-size: 13px; font-weight: 600; color: #6B7280;
                         margin: 0 0 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; }

    .sub-metrics { display: grid; grid-template-columns: repeat(3, 1fr);
                   gap: 0; margin: 0.75rem 0; }
    .sm          { padding: 0.5rem 0.7rem; }
    .sm-divider  { border-right: 1px solid #E5E7EB; }
    .sm-lbl      { font-size: 10px; color: #9CA3AF; margin: 0 0 2px; letter-spacing: 0.03em; }
    .sm-val      { font-size: 14px; font-weight: 500; color: #4B5563; margin: 0; }

    .second-rank { font-size: 12px; color: #6B7280; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# session_state 초기화
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "stage":          0,       # 0=진입 1=유형진단 2=Topic1 3=Topic2 4=결과
        "answers":        {},      # {문항번호: 점수}
        "type_scores":    {},
        "type1":          "",
        "type2":          "",
        "chat_t1":        [],
        "chat_t2":        [],
        "t1_turns":       0,
        "t2_turns":       0,
        "t2_injected":    False,   # Topic 2 조건 투입 여부 (1회 제한)
        "scores":         {},      # 채점 결과
        "aiq_index":      0,
        # ─── v8.3 — 응답자 식별 ───
        "user_name":      "",      # 이름
        "user_birth":     "",      # 생년월일 YYYYMMDD 8자리
        "consent_given":  False,   # 개인정보 수집 동의
        "user_serial":    "",      # 자동 채번된 #2026_NNNNNN (진단 완료 후 부여)
        "saved":          False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# 단계 표시 바
# ─────────────────────────────────────────────
def step_bar(current: int):
    steps = [
        (1, "유형 진단"),
        (2, "시나리오 A"),
        (3, "시나리오 B"),
        (4, "결과"),
    ]
    html = '<div class="step-bar">'
    for i, (num, label) in enumerate(steps):
        if num < current:
            cls_n, cls_l = "step-num step-done", "step-lbl-off"
            icon = "✓"
        elif num == current:
            cls_n, cls_l = "step-num step-on", "step-lbl-on"
            icon = str(num)
        else:
            cls_n, cls_l = "step-num step-off", "step-lbl-off"
            icon = str(num)
        html += f'<div class="step-item"><div class="{cls_n}">{icon}</div><span class="{cls_l}">{label}</span></div>'
        if i < len(steps) - 1:
            html += '<div class="step-div"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 1단계: 유형 진단 (20문항)
# ─────────────────────────────────────────────
def stage_1():
    step_bar(1)
    st.markdown("### AI와 나는 어떻게 함께 사고하는가")
    st.caption("20개 행동 문항에 응답해주세요. 정답이 없으며 평소 습관을 기준으로 선택하세요.")
    st.divider()

    SCALE_LABELS = ["① 거의 안 한다", "② 가끔 한다", "③ 자주 한다", "④ 항상 한다"]

    with st.form("q_form"):
        answers = {}
        for (no, text, typ, reverse) in QUESTIONS:
            st.markdown(f"**Q{no:02d}.** {text}")
            # 이전 응답이 있으면 그 값을, 없으면 None(미선택)으로 시작
            prev = st.session_state.answers.get(no)
            prev_index = (prev - 1) if prev else None
            val = st.radio(
                label=f"q{no}",
                options=[1, 2, 3, 4],
                format_func=lambda x: SCALE_LABELS[x - 1],
                index=prev_index,                 # None이면 미선택 상태
                horizontal=True,
                label_visibility="collapsed",
                key=f"q{no}"
            )
            answers[no] = val                      # val이 None이면 미응답
            st.markdown("")

        # 미응답 정확히 감지 (None 제외)
        answered = len([v for v in answers.values() if v is not None])
        st.caption(f"{answered} / 20 문항 완료")
        submitted = st.form_submit_button("✅ 완료 — 시나리오로 이동", use_container_width=True)

    if submitted:
        # 미응답 문항이 하나라도 있으면 진행 차단
        missing = [no for no, v in answers.items() if v is None]
        if missing:
            st.warning(f"아직 응답하지 않은 문항이 있습니다: Q{', Q'.join(f'{n:02d}' for n in missing)}")
        else:
            st.session_state.answers     = answers
            st.session_state.type_scores = compute_type_scores(answers)
            t1, t2 = compute_top_types(st.session_state.type_scores)
            st.session_state.type1 = t1
            st.session_state.type2 = t2
            st.session_state.stage = 2
            st.rerun()


# ─────────────────────────────────────────────
# 2단계-A: Topic 1 시나리오
# ─────────────────────────────────────────────
def stage_2():
    step_bar(2)
    st.markdown('<span class="topic-tag t1-tag">Topic 1 — 질문 수준 측정 · 최대 5턴</span>', unsafe_allow_html=True)
    scn_html = TOPIC1_SCENARIO.replace("\n\n", "<br><br>").replace("\n", "<br>")
    st.markdown(f'<div class="scn-box"><p class="scn-title">두 사람의 조언</p><p class="scn-text">{scn_html}</p></div>', unsafe_allow_html=True)
    st.caption("정답이 없습니다. 어떻게 질문하고 판단을 전개하는지가 측정됩니다.")
    st.divider()

    # 첫 AI 발화 초기화
    if not st.session_state.chat_t1:
        st.session_state.chat_t1 = [
            {"role": "assistant", "content": TOPIC1_AI_FIRST}
        ]

    # 대화 표시
    for msg in st.session_state.chat_t1:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 입력 (5턴 이하)
    MAX_TURNS_T1 = 5
    if st.session_state.t1_turns < MAX_TURNS_T1:
        user_input = st.chat_input(
            placeholder="AI에게 질문하거나 생각을 입력하세요... (200자 이하)",
            max_chars=200,
            key="t1_input"
        )
        if user_input:
            st.session_state.chat_t1.append({"role": "user", "content": user_input})
            st.session_state.t1_turns += 1
            # 마지막 턴이면 LLM 응답 생략 + 자동 다음 단계 이동
            if st.session_state.t1_turns >= MAX_TURNS_T1:
                st.session_state.stage = 3
                st.rerun()
            else:
                ai_resp = get_ai_response(st.session_state.chat_t1, topic=1)
                st.session_state.chat_t1.append({"role": "assistant", "content": ai_resp})
                st.rerun()

    st.divider()
    st.caption(f"현재 {st.session_state.t1_turns} / {MAX_TURNS_T1} 턴 완료")

    # 조기 종료 옵션 — 5턴 미만에서도 다음 단계로 갈 수 있게
    if 0 < st.session_state.t1_turns < MAX_TURNS_T1:
        if st.button("다음 시나리오로 →", use_container_width=True, type="primary"):
            st.session_state.stage = 3
            st.rerun()


# ─────────────────────────────────────────────
# 2단계-B: Topic 2 시나리오
# ─────────────────────────────────────────────
def stage_3():
    step_bar(3)
    st.markdown('<span class="topic-tag t2-tag">Topic 2 — 사고 전환 측정 · 최대 7턴</span>', unsafe_allow_html=True)
    scn_html = TOPIC2_SCENARIO.replace("\n\n", "<br><br>").replace("\n", "<br>")
    st.markdown(f'<div class="scn-box"><p class="scn-title">친구의 부탁</p><p class="scn-text">{scn_html}</p></div>', unsafe_allow_html=True)
    st.caption("정답이 없습니다. 판단을 어떻게 전개하고 수정하는지가 측정됩니다.")
    st.divider()

    # 첫 AI 발화 초기화
    if not st.session_state.chat_t2:
        st.session_state.chat_t2 = [
            {"role": "assistant", "content": TOPIC2_AI_FIRST}
        ]

    # 대화 표시
    for msg in st.session_state.chat_t2:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def _run_scoring_and_advance():
        """채점 실행 후 stage_4로 이동 — 마지막 턴 자동 / 조기 종료 버튼 공통 사용"""
        with st.spinner("채점 중... (앙상블 3회, 약 20~30초 소요)"):
            scores = run_scoring(st.session_state.chat_t1, st.session_state.chat_t2)
            st.session_state.scores    = scores
            st.session_state.aiq_index = compute_aiq_index(scores["QLI"], scores["Recon"])
        st.session_state.stage = 4
        st.rerun()

    # 입력 (7턴 이하)
    MAX_TURNS_T2 = 7
    if st.session_state.t2_turns < MAX_TURNS_T2:
        user_input = st.chat_input(
            placeholder="AI에게 질문하거나 생각을 입력하세요... (200자 이하)",
            max_chars=200,
            key="t2_input"
        )
        if user_input:
            st.session_state.chat_t2.append({"role": "user", "content": user_input})
            st.session_state.t2_turns += 1
            # 마지막 턴이면 LLM 응답 생략 + 바로 채점·결과로 이동
            if st.session_state.t2_turns >= MAX_TURNS_T2:
                _run_scoring_and_advance()
            else:
                ai_resp = get_ai_response(st.session_state.chat_t2, topic=2)
                st.session_state.chat_t2.append({"role": "assistant", "content": ai_resp})
                st.rerun()

    st.divider()
    st.caption(f"현재 {st.session_state.t2_turns} / {MAX_TURNS_T2} 턴 완료")

    # 조기 종료 옵션 — 7턴 미만에서 분석 요청 가능
    if 0 < st.session_state.t2_turns < MAX_TURNS_T2:
        if st.button("분석 요청 →", use_container_width=True, type="primary"):
            _run_scoring_and_advance()


# ─────────────────────────────────────────────
# 3단계: 결과 리포트
# ─────────────────────────────────────────────
def stage_4():
    # v8.1 — 스텝바 제거, 보고서 헤더로 대체
    scores    = st.session_state.scores
    type1     = st.session_state.type1
    type2     = st.session_state.type2
    ts        = st.session_state.type_scores
    aiq       = st.session_state.aiq_index
    qli       = scores.get("QLI",   5.0)
    mti       = scores.get("MTI",   5.0)
    recon     = scores.get("Recon", 5.0)
    has_cs    = scores.get("ClassS", False)
    insight   = scores.get("insight_ko", "")
    low_rel        = scores.get("low_reliability", False)
    scoring_failed = scores.get("scoring_failed", False)

    # ─── 저장 (시리얼 채번) — 헤더 표시보다 먼저 실행 ───
    if not st.session_state.saved:
        serial = save_result(
            name=st.session_state.user_name,
            birth=st.session_state.user_birth,
            type1=type1, type2=type2,
            type_scores=ts, qli=qli, recon=recon,
            mti=mti, aiq=aiq, has_class_s=has_cs,
            answers=st.session_state.answers,
            log_t1=st.session_state.chat_t1,
            log_t2=st.session_state.chat_t2,
            scoring_failed=scoring_failed
        )
        if serial:
            st.session_state.user_serial = serial
            st.session_state.saved = True

    # ─── 보고서 헤더 ───
    name_display  = st.session_state.user_name or "anonymous"
    birth_raw     = st.session_state.user_birth or ""
    # 19950315 → 1995.03.15 표시
    birth_display = f"{birth_raw[:4]}.{birth_raw[4:6]}.{birth_raw[6:]}" if len(birth_raw) == 8 else ""
    serial_disp   = st.session_state.user_serial or "진단 중"
    diag_time     = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    st.markdown(f'''
    <div class="report-header">
      <p class="report-title">AIQ Diagnostic Report</p>
      <p class="report-name">AI 공생 지수 진단 결과</p>
      <div class="report-meta">
        <span><strong>응답자</strong>{name_display} ({birth_display})</span>
        <span><strong>시리얼</strong>{serial_disp}</span>
        <span><strong>진단일</strong>{diag_time}</span>
        <span><strong>버전</strong>AIQ v0.6 · 파일럿</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # ─── AIQ 단일 지수 — 화면 주인공 ───
    badge_class_s = '<span class="aiq-badge">✦ 문제재정의</span>' if has_cs else ''
    st.markdown(f'''
    <div class="aiq-hero">
      <p class="aiq-label">AIQ</p>
      <p class="aiq-value">{aiq}</p>
      <div class="aiq-badge-row">
        <span class="aiq-badge">{type1}</span>
        {badge_class_s}
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # ─── 유형 설명 ───
    axis = TYPE_AXIS.get(type1, ("", ""))
    st.markdown(f'<h2 class="section-title">{type1}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="axis-tag">{axis[0]} · {axis[1]} — 1순위</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="type-quote">{TYPE_DESC.get(type1, "")}</p>', unsafe_allow_html=True)
    st.divider()

    # ─── 4유형 좌표 ───
    st.markdown('<p class="sub-section-title">유형 좌표</p>', unsafe_allow_html=True)
    type_order = ["상상가형", "설계자형", "의존형", "실행형"]
    labels_map = {"설계자": "설계자형", "상상가": "상상가형", "실행": "실행형", "의존": "의존형"}

    html_coord = '<div class="coord-grid">'
    for t in type_order:
        raw_key = next((k for k, v in labels_map.items() if v == t), "")
        score_val = ts.get(raw_key, 0)
        is_first  = (t == type1)
        is_second = (t == type2)
        suffix    = " ✦" if is_first else ""
        rank_note = " — 1순위" if is_first else (" — 2순위" if is_second else "")
        cls       = "cc cc-hl" if is_first else "cc"
        html_coord += (
            f'<div class="{cls}">'
            f'<span class="cc-name">{t}{suffix}</span>'
            f'<span class="cc-tag">{score_val}점{rank_note}</span>'
            f'</div>'
        )
    html_coord += '</div>'
    st.markdown(html_coord, unsafe_allow_html=True)

    if type2:
        st.markdown(f'<p class="second-rank">2순위: {type2}</p>', unsafe_allow_html=True)

    st.divider()

    # ─── 측정 근거 (서브 지표, 작게) ───
    st.markdown('<p class="sub-section-title">측정 근거</p>', unsafe_allow_html=True)
    mti_display = f"{mti:.1f}" + (" ✦" if has_cs else "")
    st.markdown(f'''
    <div class="sub-metrics">
      <div class="sm sm-divider">
        <p class="sm-lbl">질문 설계 QLI</p>
        <p class="sm-val">{qli:.1f}</p>
      </div>
      <div class="sm sm-divider">
        <p class="sm-lbl">재구성력 Recon</p>
        <p class="sm-val">{recon:.1f}</p>
      </div>
      <div class="sm">
        <p class="sm-lbl">사고 전환 MTI</p>
        <p class="sm-val">{mti_display}</p>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    if scoring_failed:
        st.error("⚠️ 채점에 실패했습니다. 표시된 점수는 기본값(5.0)이며 실제 측정 결과가 아닙니다. 운영자에게 문의해주세요.")
    elif low_rel:
        st.warning(f"⚠️ MTI 신뢰도 낮음 — 앙상블 편차 > {DIVERGENCE_THRESHOLD}. 참고용으로만 활용하세요.")

    if insight:
        st.markdown(f'<div class="interp">{insight}</div>', unsafe_allow_html=True)

    st.divider()

    # ─── 저장 확인 표시 (저장은 보고서 헤더 직전에 이미 수행됨) ───
    if st.session_state.saved and st.session_state.user_serial:
        st.caption(f"✅ 결과가 저장되었습니다.  ·  시리얼: {st.session_state.user_serial}")
    elif not st.session_state.saved:
        st.warning("저장 실패 — 결과가 Google Sheets에 기록되지 않았습니다. 운영자에게 알려주세요.")

    if st.button("처음으로 돌아가기", use_container_width=False):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────
# 진입 화면 (응답자 정보 + 동의)
# ─────────────────────────────────────────────
def validate_name(name: str) -> bool:
    """이름: 2~20자, 한글/영문/공백"""
    if not name or not name.strip():
        return False
    return bool(re.match(r"^[가-힣A-Za-z\s]{2,20}$", name.strip()))


def validate_birth(birth: str) -> bool:
    """생년월일: YYYYMMDD 8자리 숫자, 유효한 날짜, 1900~현재"""
    if not birth or len(birth) != 8 or not birth.isdigit():
        return False
    try:
        d = datetime.strptime(birth, "%Y%m%d")
        return 1900 <= d.year <= datetime.now().year
    except ValueError:
        return False


def stage_0():
    st.markdown("## AIQ 파일럿 진단")
    st.markdown("""
AI와 나는 어떻게 함께 사고하는가를 측정합니다.

- **20개 행동 문항**으로 AI 협업 사고 유형을 분류합니다
- **시나리오 2개**로 질문 설계력·사고 전환 점수를 산출합니다
- 소요 시간: 약 10분
    """)
    st.divider()

    st.markdown("### 응답자 정보")
    col1, col2 = st.columns([1, 1])
    with col1:
        name = st.text_input(
            "이름",
            value=st.session_state.get("user_name", ""),
            max_chars=20,
            placeholder="예: 홍길동"
        )
    with col2:
        birth = st.text_input(
            "생년월일 (YYYYMMDD)",
            value=st.session_state.get("user_birth", ""),
            max_chars=8,
            placeholder="예: 19950315"
        )

    st.caption("※ 결과 보고와 향후 재진단 시 식별을 위해 사용됩니다.")
    st.divider()

    # 동의
    consent = st.checkbox(
        "응답자 식별 및 향후 재진단 시 결과 추적을 위해 이름과 생년월일을 수집·이용하는 것에 동의합니다. "
        "수집된 정보는 진단 결과 보고와 데이터 분석 외 용도로 사용되지 않으며, 요청 시 삭제할 수 있습니다.",
        value=st.session_state.get("consent_given", False)
    )

    st.markdown("")

    # 검증
    name_valid    = validate_name(name)
    birth_valid   = validate_birth(birth)
    ready         = name_valid and birth_valid and consent

    # 검증 안내
    if name and not name_valid:
        st.warning("이름은 2~20자의 한글/영문으로 입력해주세요.")
    if birth and not birth_valid:
        st.warning("생년월일은 YYYYMMDD 8자리 형식이어야 합니다. (예: 19950315)")

    if st.button("진단 시작 →", use_container_width=True, type="primary", disabled=not ready):
        st.session_state.user_name     = name.strip()
        st.session_state.user_birth    = birth.strip()
        st.session_state.consent_given = True
        st.session_state.stage = 1
        st.rerun()

    if not consent:
        st.caption("진단을 시작하려면 위 동의에 체크해주세요.")


# ─────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────
stage = st.session_state.get("stage", 0)

if stage == 0:
    stage_0()
elif stage == 1:
    stage_1()
elif stage == 2:
    stage_2()
elif stage == 3:
    stage_3()
elif stage == 4:
    stage_4()
