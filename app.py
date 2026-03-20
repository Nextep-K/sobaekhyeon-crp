
import streamlit as st
from openai import OpenAI
import pandas as pd
import re
import base64

# 1. AI 열쇠 및 기본 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="소백현: CRP Analysis System", layout="wide")

# 2. [디자인 지문] 나눔고딕 폰트 적용 로직
def get_base64_font(font_path):
    with open(font_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 폰트 파일 경로 (작가님이 설정하신 assets 구조)
font_path = "assets/fonts/NanumGothic-Regular.ttf"

try:
    base64_font = get_base64_font(font_path)
    font_style = f"""
    <style>
    @font-face {{
        font-family: 'SobaekhyunFont';
        src: url(data:font/ttf;base64,{base64_font}) format('truetype');
    }}
    html, body, [class*="css"], .stMarkdown {{
        font-family: 'SobaekhyunFont', sans-serif !important;
    }}
    </style>
    """
    st.markdown(font_style, unsafe_allow_html=True)
except Exception:
    st.caption("시스템 폰트를 로드 중입니다...")

# 3. [보안/위장] 용어 매핑 정의
# 내부용(CRP/MTI) -> 대외용(S-Core/Alpha/Beta)
SECURITY_DICT = {
    "MTI": "S-Core",
    "Recognition": "Alpha-Density",
    "Reconfiguration": "Beta-Density",
    "Orchestration": "Gamma-Density",
    "SAI": "V-Orbit"
}

# 4. 헤더 부분
st.title("🛡️ 소백현: CRP 인지 변화 분석 리포터")
st.markdown(f"**{SECURITY_DICT['MTI']}** 설계도에 따라 학습자의 '날카로움'을 측정하고 전문가 해설을 제공합니다.")
st.divider()

# --- 화면 분할 ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 대화 로그 입력")
    log_input = st.text_area("분석할 [Conversation Log]를 입력하세요.", height=550)
    analyze_btn = st.button("🚀 분석 엔진 가동 (Secure Mode)", use_container_width=True)

with col2:
    st.subheader("📋 분석 리포트 (Secure Insights)")
    
    if analyze_btn and log_input:
        with st.spinner("인지 지문 분석 및 위장 보안 프로토콜 가동 중..."):
            try:
                # [동적 분석 엔진] 보안 위장 용어가 적용된 시스템 프롬프트
                SYSTEM_PROMPT = f"""
                너는 '소백현' 인지 분석 엔진이다. 다음 지침에 따라 로그를 분석하라.
                
                [보안 용어 강제 적용]
                - MTI 대신 '{SECURITY_DICT['MTI']}'를 사용하라.
                - SRI Recognition 대신 '{SECURITY_DICT['ALPHA']}'를 사용하라.
                - SRI Reconfiguration 대신 '{SECURITY_DICT['BETA']}'를 사용하라.
                - SRI Orchestration 대신 '{SECURITY_DICT['GAMMA']}'를 사용하라.
                
                [분석 가이드]
                1. 정량 지표: {SECURITY_DICT['MTI']}는 1-6, 나머지는 1-10 점수로 산출.
                2. 인사이트: 
                   - '{SECURITY_DICT['BETA']}'가 9점 이상이면 '설계자적 퀀텀 점프'로 명명.
                   - '{SECURITY_DICT['MTI']}' 6단계 판정 시 반드시 시스템 구조 비판 여부를 확인하라.
                """
                
                # OpenAI API 호출
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": log_input}
                    ],
                    temperature=0.2 # 일관성 있는 분석을 위해 낮게 설정
                )
                
                full_result = response.choices[0].message.content

                # 시각화 (정규표현식으로 위장된 점수 추출)
                try:
                    scores = re.findall(r": (\d+)", full_result)
                    if len(scores) >= 4:
                        chart_data = pd.DataFrame({
                            'Metric': ['Alpha', 'Beta', 'Gamma'],
                            'Score': [int(scores[1]), int(scores[2]), int(scores[3])]
                        })
                        st.bar_chart(chart_data.set_index('Metric'))
                except:
                    pass

                # 최종 리포트 출력
                st.markdown(full_result)
                
            except Exception as e:
                st.error(f"보안 엔진 오류: {e}")

st.divider()
st.sidebar.markdown(f"""
### 🛡️ Gatekeeper Status
- **Font**: NanumGothic Applied
- **Security**: {SECURITY_DICT['MTI']} Masking Active
- **Protocol**: v1.5 Stable
""")
st.caption("© 2026 소백현 프로젝트 - Secure Cognitive Analysis System")
