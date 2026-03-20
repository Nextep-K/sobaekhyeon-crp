import streamlit as st
from openai import OpenAI
import pandas as pd
import re
from datetime import datetime
from fpdf import FPDF
import io

# 1. AI 열쇠 설정 (Streamlit Secrets 활용)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

st.set_page_config(page_title="CRP Analysis System", layout="wide")

# 헤더 부분
st.title("🧠 소백현: CRP 인지 변화 분석 리포터")
st.markdown("학습자의 ID별로 인지 궤적을 추적하고 **알고리듬**으로 증명하는 전문가용 시스템입니다.")
st.divider()

# --- 화면 분할 (좌측: 입력 및 설정 / 우측: 결과 출력) ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("👤 학습자 정보 설정")
    # [수정 사항] 영문 7자 이내 아이디 입력 칸 추가
    user_id = st.text_input(
        "학습자 ID (영문 7자 이하)", 
        max_chars=7, 
        placeholder="예: SJKIM7",
        help="데이터베이스 식별을 위한 고유 ID입니다. 이후 시계열 분석의 기준이 됩니다."
    )
    
    st.subheader("📥 대화 로그 데이터")
    log_input = st.text_area(
        "분석할 [Conversation Log]를 입력하세요.", 
        height=500,
        placeholder="학습자와 AI 간의 대화 내용을 붙여넣으세요..."
    )
    
    # 분석 실행 버튼
    analyze_btn = st.button("🚀 CRP 분석 엔진 가동", use_container_width=True)

with col2:
    st.subheader("📋 분석 리포트 및 전문가 해설")
    
    if analyze_btn:
        if not user_id:
            st.warning("학습자 ID를 먼저 설정해 주세요 (영문 7자 이내).")
        elif not log_input:
            st.warning("분석할 로그 데이터를 입력해 주세요.")
        else:
            # 분석 시각 및 데이터 생성
            analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with st.spinner(f"[{user_id}] 학습자의 인지 구조를 알고리듬으로 분석 중입니다..."):
                try:
                    # 소백현 고유 알고리듬 프롬프트 (보안 코드네임 제거 버전)
                    CRP_SYSTEM_PROMPT = """
너는 CRP(Cognitive Re-configuration Protocol) 기반 학습 분석 시스템이다.
제공된 로그를 분석하여 학습자의 인지적 진실을 데이터로 증명하라.

### 1. 📊 정량적 지표 분석
- MTI Stage: (1~6 숫자)
- SRI Recognition: (1~10 정수)
- SRI Reconfiguration: (1~10 정수)
- SRI Orchestration: (1~10 정수)
- SAI SRR (Self-Reference): (0.0~1.0 실수)
- SAI TTR (Lexical Diversity): (0.0~1.0 실수)
- SAI POI (Objectivity): (0.0~1.0 실수)
- SAI Level: (High/Medium/Low)

### 2. 💡 분석 리포트 핵심 관전 포인트 (Insight)
- MTI 해설: 학습자의 인지 구조가 시스템을 조망하는 Stage 6에 도달했는지 판정하라.
- SRI 해설: 인식(Recognition)과 재구성(Reconfiguration)의 알고리듬적 조화를 분석하라.
- SAI 해설: 수치에 기반하여 학습자의 메타인지적 주체성을 평가하라.

Overall cognitive phase: (한 문장 요약)
"""

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": CRP_SYSTEM_PROMPT},
                            {"role": "user", "content": f"ID: {user_id}\nTime: {analysis_time}\n[Log]:\n{log_input}"}
                        ],
                        temperature=0.3
                    )
                    full_result = response.choices[0].message.content
                    
                    # 분석 헤더 정보 시각화
                    st.success(f"✅ 분석 완료 - ID: {user_id} | 일시: {analysis_time}")

                    # --- SRI 지표 막대 그래프 ---
                    scores = re.findall(r"SRI (?:Recognition|Reconfiguration|Orchestration): (\d+)", full_result)
                    if len(scores) >= 3:
                        sri_data = pd.DataFrame({
                            '지표': ['Recognition', 'Reconfiguration', 'Orchestration'],
                            '점수': [int(scores[0]), int(scores[1]), int(scores[2])]
                        })
                        st.bar_chart(sri_data.set_index('지표'))

                    # 최종 결과 출력
                    st.markdown(full_result)
                    
                    # --- PDF 생성 및 다운로드 ---
                    st.divider()
                    
                    def create_pdf(content, u_id, a_time):
                        pdf = FPDF()
                        pdf.add_page()
                        # 한글 폰트 설정 (저장소 내 NanumGothic.ttf 필수)
                        try:
                            pdf.add_font('Nanum', '', 'NanumGothic.ttf')
                            pdf.set_font('Nanum', size=11)
                        except:
                            pdf.set_font("Helvetica", size=11)
                        
                        pdf.cell(0, 10, f"SKIM CRP Analysis Report - ID: {u_id}", ln=True, align='C')
                        pdf.cell(0, 10, f"Analysis Time: {a_time}", ln=True, align='C')
                        pdf.ln(5)
                        clean_text = content.replace('#', '').replace('*', '')
                        pdf.multi_cell(0, 8, txt=clean_text)
                        return pdf.output()

                    pdf_bytes = create_pdf(full_result, user_id, analysis_time)
                    
                    st.download_button(
                        label=f"📄 {user_id}의 분석 리포트 PDF 출력 [y]",
                        data=pdf_bytes,
                        file_name=f"SKIM_{user_id}_{datetime.now().strftime('%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"알고리듬 실행 중 오류가 발생했습니다: {e}")

st.divider()
st.caption("© 2026 소백현 프로젝트 - SJ KIM CRP Analysis System | 설계자의 시선으로 인지를 관찰합니다.")