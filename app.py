import streamlit as st
from openai import OpenAI
import pandas as pd
import re

# 1. AI 열쇠 설정 (박사님의 API 키를 여기에 넣으세요)
client = OpenAI(api_key="sk-xxxx...") 

st.set_page_config(page_title="CRP Analysis System", layout="wide")

# 헤더 부분
st.title("🧠 소백현: CRP 인지 변화 분석 리포터")
st.markdown("박사님의 CRP 설계도에 따라 학습자의 '날카로움'을 측정하고 전문가 해설을 제공합니다.")
st.divider()

# --- 화면 분할 (좌측: 입력 / 우측: 결과) ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 대화 로그 입력")
    log_input = st.text_area("분석할 [Conversation Log]를 여기에 붙여넣으세요.", height=600, placeholder="대화 내용을 입력하고 아래 버튼을 누르세요...")
    analyze_btn = st.button("🚀 CRP 분석 엔진 가동", use_container_width=True)

with col2:
    st.subheader("📋 분석 리포트 및 전문가 해설")
    
    if analyze_btn:
        if log_input:
            # --- 진행 표시 (Spinner) ---
            with st.spinner("AI가 학습자의 사고 구조를 정밀 분석 중입니다. 잠시만 기다려 주세요..."):
                try:
                    # [업그레이드 완료] 새로운 설계를 반영한 시스템 프롬프트
                    CRP_SYSTEM_PROMPT = """
너는 CRP(Cognitive Re-configuration Protocol) 기반 학습 분석 시스템이며, 교육공학 전문가의 관점에서 통찰력 있는 해설을 제공해야 한다.
제공된 로그를 분석하여 다음 형식을 엄격히 지켜 출력하라.

### 1. 📊 정량적 지표 분석 (엄격 모드)
[수치 산출 지침]
- 모든 점수는 반드시 숫자로만 표기한다. (Low, Medium, High 등 형용사 사용 금지)
- SRI 지표는 1~10 사이의 정수로 표기한다.
- SAI 지표(SRR, TTR, POI)는 0.0~1.0 사이의 실수(비율)로 산출한다.

- MTI Stage: (1~6 숫자)
- SRI Recognition: (1~10 정수)
- SRI Reconfiguration: (1~10 정수)
- SRI Orchestration: (1~10 정수)
- SAI SRR (Self-Reference): (0.0~1.0 실수)
- SAI TTR (Lexical Diversity): (0.0~1.0 실수)
- SAI POI (Objectivity): (0.0~1.0 실수)
- SAI Level: (High/Medium/Low - 수치에 근거하여 표기)

### 2. 💡 이번 분석 리포트의 핵심 관전 포인트 (Insight)
이 섹션에서는 수치가 의미하는 바를 다음의 '날카로운' 관점에서 해설하라.

[MTI 판정 특례]
- Stage 6 (Recursive structuring): 학습자가 분석 시스템의 오류 가능성을 지적하거나, AI의 논리적 허점을 설계적 관점에서 비판할 경우 반드시 Stage 6를 부여한다. (단순 의문은 Stage 2)

- MTI 해설: 학습자의 인지 구조가 현재 '도약' 단계인지, 아니면 시스템 자체를 조망하는 '재귀적' 단계인지 명확히 판정하라.
- SRI 해설: 인식(Recognition)과 조합(Orchestration)의 간극을 통해 학습자의 강점과 보완점을 지적하라.
- SAI 해설: 산출된 수치(SRR, TTR, POI)를 근거로 학습자의 메타인지적 주체성을 평가하라.

Overall cognitive phase: (한 문장 요약)
"""

                    # API 호출
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": CRP_SYSTEM_PROMPT},
                            {"role": "user", "content": f"[Conversation Log] :\n{log_input}"}
                        ],
                        temperature=0.3
                    )
                    
                    full_result = response.choices[0].message.content

                    # --- 막대 그래프 시각화 (숫자 추출 시도) ---
                    try:
                        # 정규표현식으로 SRI 점수 추출
                        scores = re.findall(r"SRI \w+: (\d+)", full_result)
                        if len(scores) >= 3:
                            sri_data = pd.DataFrame({
                                '지표': ['Recognition', 'Reconfiguration', 'Orchestration'],
                                '점수': [int(scores[0]), int(scores[1]), int(scores[2])]
                            })
                            st.bar_chart(sri_data.set_index('지표'))
                    except:
                        st.caption("그래프 생성 중... (텍스트 분석 결과는 아래에 표시됩니다)")

                    # 최종 결과 출력
                    st.markdown(full_result)
                    
                except Exception as e:
                    st.error(f"오류 발생: {e}")
        else:
            st.warning("분석할 로그 데이터를 먼저 입력해 주세요.")
    else:
        st.info("왼쪽에 로그를 입력하고 '분석 엔진 가동' 버튼을 누르면 리포트가 여기에 표시됩니다.")

st.divider()
st.caption("© 2026 소백현 프로젝트 - SJ KIM CRP Analysis System")