import streamlit as st
from openai import OpenAI
import pandas as pd
import re
from datetime import datetime
from fpdf import FPDF
import io

# 1. AI 열쇠 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

st.set_page_config(page_title="CRP Analysis System", layout="wide")

st.title("🧠 소백현: CRP 인지 변화 분석 리포터")
st.markdown("학습자의 인지 궤적을 **알고리듬**으로 관찰하고 기록합니다.")
st.divider()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("👤 학습자 정보 설정")
    user_id = st.text_input("학습자 ID (영문 7자 이하)", max_chars=7, placeholder="예: SJKIM7")
    
    st.subheader("📥 대화 로그 데이터")
    log_input = st.text_area("분석할 [Conversation Log] 입력", height=500)
    analyze_btn = st.button("🚀 CRP 분석 엔진 가동", use_container_width=True)

with col2:
    st.subheader("📋 분석 리포트 및 전문가 해설")
    
    if analyze_btn:
        if not user_id or not log_input:
            st.warning("ID와 로그 데이터를 모두 입력해 주세요.")
        else:
            analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with st.spinner(f"[{user_id}]의 인지 구조 분석 중..."):
                try:
                    CRP_SYSTEM_PROMPT = """
너는 CRP(Cognitive Re-configuration Protocol) 기반 학습 분석 시스템이다.
다음 형식을 엄격히 지켜 출력하라.

### 1. 📊 정량적 지표 분석
- MTI Stage: (1~6 숫자)
- SRI Recognition: (1~10 정수)
- SRI Reconfiguration: (1~10 정수)
- SRI Orchestration: (1~10 정수)
- SAI SRR (Self-Reference): (0.0~1.0 실수)
- SAI TTR (Lexical Diversity): (0.0~1.0 실수)
- SAI POI (Objectivity): (0.0~1.0 실수)
- SAI Level: (High/Medium/Low)

### 2. 💡 분석 리포트 핵심 관전 포인트
(해설 내용 작성)

Overall cognitive phase: (요약)
"""
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": CRP_SYSTEM_PROMPT},
                            {"role": "user", "content": f"ID: {user_id}\nLog:\n{log_input}"}
                        ],
                        temperature=0.3
                    )
                    full_result = response.choices[0].message.content
                    
                    st.success(f"✅ 분석 완료: {user_id}")

                    # --- [복구] SRI 지표 그래프 생성 로직 ---
                    # 결과 텍스트에서 숫자 추출
                    sri_scores = re.findall(r"SRI (?:Recognition|Reconfiguration|Orchestration): (\d+)", full_result)
                    
                    if len(sri_scores) >= 3:
                        chart_data = pd.DataFrame({
                            '지표': ['Recognition', 'Reconfiguration', 'Orchestration'],
                            '점수': [int(sri_scores[0]), int(sri_scores[1]), int(sri_scores[2])]
                        })
                        st.bar_chart(chart_data.set_index('지표'))
                    else:
                        st.info("📊 지표 데이터를 추출 중입니다. 잠시 후 리포트 본문을 확인하세요.")

                    # 리포트 본문 출력
                    st.markdown(full_result)
                    
                    # --- [교정] PDF 생성 및 다운로드 로직 ---
                    def create_pdf(content, u_id, a_time):
                        pdf = FPDF()
                        pdf.add_page()
                        try:
                            pdf.add_font('Nanum', '', 'NanumGothic.ttf')
                            pdf.set_font('Nanum', size=11)
                        except:
                            pdf.set_font("Helvetica", size=11)
                        
                        pdf.cell(0, 10, f"CRP Report - ID: {u_id}", ln=True, align='C')
                        pdf.cell(0, 10, f"Date: {a_time}", ln=True, align='C')
                        pdf.ln(5)
                        # 특수문자 제거 및 텍스트 삽입
                        clean_text = content.replace('#', '').replace('*', '')
                        pdf.multi_cell(0, 8, txt=clean_text)
                        
                        # fpdf2의 output은 bytearray이므로 bytes로 변환하여 반환
                        return bytes(pdf.output())

                    pdf_data = create_pdf(full_result, user_id, analysis_time)
                    
                    st.divider()
                    st.download_button(
                        label=f"📄 {user_id}의 리포트 PDF 다운로드",
                        data=pdf_data,
                        file_name=f"SKIM_{user_id}_{datetime.now().strftime('%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"알고리듬 실행 중 오류가 발생했습니다: {e}")

st.divider()
st.caption("© 2026 소백현 프로젝트 - SJ KIM CRP Analysis System")