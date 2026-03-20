import streamlit as st
from openai import OpenAI
import pandas as pd
import re
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

# 1. API 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

st.set_page_config(page_title="SKIM Ensemble System", layout="wide")

st.title("🧠 소백현: 앙상블 CRP 분석 엔진")
st.markdown("교차 검증 **알고리듬**을 통해 인지적 편차를 최소화하고 시각적 리포트를 생성합니다.")
st.divider()

# --- [교정] PDF 생성 함수: 폰트 등록 및 수치 포맷 고정 ---
def create_ensemble_pdf(content, u_id, time, chart_img, avg_data):
    pdf = FPDF()
    pdf.add_page()
    
    try:
        # 깃허브에 NanumGothic.ttf와 NanumGothicBold.ttf가 있어야 합니다.
        pdf.add_font('Nanum', '', 'NanumGothic.ttf')
        pdf.add_font('Nanum', 'B', 'NanumGothicBold.ttf') 
        pdf.set_font('Nanum', size=11)
        has_nanum = True
    except:
        pdf.set_font("Helvetica", size=11)
        has_nanum = False

    # 리포트 헤더 (Bold 적용)
    pdf.set_font('Nanum', 'B', 16) if has_nanum else pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 15, f"SKIM Ensemble Cognitive Report", ln=True, align='C')
    pdf.set_font('Nanum', '', 10) if has_nanum else pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 5, f"ID: {u_id} | Date: {time}", ln=True, align='C')
    pdf.ln(10)

    # 그래프 삽입
    chart_img.seek(0)
    pdf.image(chart_img, x=15, y=45, w=140)
    pdf.ln(95) 

    # [교정] 앙상블 평균 수치 요약 (소수점 2자리 고정으로 수치 왜곡 방지)
    pdf.set_font('Nanum', 'B', 12) if has_nanum else pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, " [ Ensemble Analytics Summary ]", ln=True)
    pdf.set_font('Nanum', '', 11) if has_nanum else pdf.set_font("Helvetica", '', 11)
    
    pdf.cell(0, 8, f"• Average MTI Stage: {avg_data['MTI']:.2f}", ln=True)
    pdf.cell(0, 8, f"• SRI Recognition (Avg): {avg_data['Rec']:.2f}", ln=True)
    pdf.cell(0, 8, f"• SRI Reconfiguration (Avg): {avg_data['Recon']:.2f}", ln=True)
    pdf.cell(0, 8, f"• SRI Orchestration (Avg): {avg_data['Orc']:.2f}", ln=True)
    pdf.ln(5)

    # 전문가 종합 해설
    pdf.set_font('Nanum', 'B', 12) if has_nanum else pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, " [ Expert Insight ]", ln=True)
    pdf.set_font('Nanum', '', 11) if has_nanum else pdf.set_font("Helvetica", '', 11)
    
    clean_text = content.replace('#', '').replace('*', '').strip()
    pdf.multi_cell(0, 8, txt=clean_text)

    return bytes(pdf.output())

# --- 메인 UI 및 분석 로직 ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("👤 학습자 정보")
    user_id = st.text_input("학습자 ID", max_chars=10, placeholder="예: sjkim")
    log_input = st.text_area("분석할 [Conversation Log] 입력", height=500)
    analyze_btn = st.button("🚀 앙상블 분석 시작 (3회 교차 검증)", use_container_width=True)

if analyze_btn and user_id and log_input:
    with col2:
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = []
        progress_bar = st.progress(0)
        
        try:
            for i in range(3):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "너는 CRP 분석 시스템이다. 반드시 [DATA] 섹션을 만들고 '지표명: 숫자' 형식으로 7개 지표를 써라. 그 후 [INSIGHT] 섹션에 해설을 작성하라. 지표명은 MTI, REC, RECON, ORC, SRR, TTR, POI만 사용한다."},
                        {"role": "user", "content": f"ID: {user_id}\nLog:\n{log_input}"}
                    ],
                    temperature=0.1
                )
                text = response.choices[0].message.content
                
                # [교정] 엄격한 데이터 추출: 지표 키워드와 결합된 숫자만 추출
                data_part = re.search(r"\[DATA\](.*?)\[INSIGHT\]", text, re.S)
                if data_part:
                    section_text = data_part.group(1)
                    m_values = re.findall(r"(?:MTI|REC|RECON|ORC|SRR|TTR|POI):\s*(\d+\.?\d*)", section_text, re.I)
                    if len(m_values) >= 7:
                        results.append([float(v) for v in m_values[:7]] + [text])
                
                progress_bar.progress((i + 1) * 33)
            
            if results:
                df = pd.DataFrame([r[:7] for r in results], columns=['MTI', 'Rec', 'Recon', 'Orc', 'SRR', 'TTR', 'POI'])
                avg = df.mean()

                st.success(f"✅ 앙상블 분석 완료 (ID: {user_id})")
                
                # 그래프 시각화
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Rec', 'Recon', 'Orc'], [avg['Rec'], avg['Recon'], avg['Orc']], color=['#3498db', '#e74c3c', '#2ecc71'])
                ax.set_title(f"SRI Cognitive Structure (Avg) - {user_id}")
                ax.set_ylim(0, 10)
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png')
                st.image(img_buf)

                # 해설 출력 및 PDF 생성
                final_insight = results[-1][-1].split("[INSIGHT]")[-1].strip()
                st.markdown("### 💡 전문가 종합 해설")
                st.write(final_insight)

                pdf_bytes = create_ensemble_pdf(final_insight, user_id, analysis_time, img_buf, avg)
                st.download_button(label="📄 PDF 리포트 다운로드", data=pdf_bytes, file_name=f"SKIM_{user_id}.pdf", mime="application/pdf", use_container_width=True)

        except Exception as e:
            st.error(f"실행 중 오류가 발생했습니다: {e}")