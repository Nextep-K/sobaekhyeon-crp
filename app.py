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

st.title("🧠 소백현: 앙상블 CRP 분석 엔진 (Lite)")
st.markdown("2회 교차 검증 **알고리듬**을 통해 속도를 개선하고 인지 분석 리포트를 생성합니다.")
st.divider()

# --- PDF 생성 함수 (바이트 변환 및 폰트 고정) ---
def create_ensemble_pdf(content, u_id, time, chart_img, avg_data):
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font('Nanum', '', 'NanumGothic.ttf')
        pdf.add_font('Nanum', 'B', 'NanumGothicBold.ttf') 
        pdf.set_font('Nanum', size=11)
        has_font = True
    except:
        pdf.set_font("Helvetica", size=11)
        has_font = False

    # 리포트 헤더
    pdf.set_font('Nanum', 'B', 16) if has_font else pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 15, "SKIM Ensemble Cognitive Report", ln=True, align='C')
    pdf.set_font('Nanum', '', 10) if has_font else pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 5, f"ID: {u_id} | Date: {time}", ln=True, align='C')
    pdf.ln(10)

    # 그래프 삽입
    chart_img.seek(0)
    pdf.image(chart_img, x=15, y=45, w=140)
    pdf.ln(95) 

    # 수치 요약
    pdf.set_font('Nanum', 'B', 12) if has_font else pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, " [ Ensemble Analytics Summary ]", ln=True)
    pdf.set_font('Nanum', '', 11) if has_font else pdf.set_font("Helvetica", '', 11)
    for k, v in avg_data.items():
        if k in ['MTI', 'Rec', 'Recon', 'Orc']:
            pdf.cell(0, 8, f"• Average {k}: {v:.2f} / 10.0", ln=True)
    pdf.ln(5)

    # 해설
    pdf.set_font('Nanum', 'B', 12) if has_font else pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, " [ Expert Insight ]", ln=True)
    pdf.set_font('Nanum', '', 11) if has_font else pdf.set_font("Helvetica", '', 11)
    pdf.multi_cell(0, 8, txt=content.replace('#', '').replace('*', '').strip())

    return bytes(pdf.output())

# --- 메인 로직 (2회 반복으로 수정) ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("👤 학습자 정보")
    user_id = st.text_input("학습자 ID", max_chars=10, value="sj")
    log_input = st.text_area("분석할 [Conversation Log] 입력", height=500)
    analyze_btn = st.button("🚀 빠른 앙상블 분석 (2회 교차 검증)", use_container_width=True)

if analyze_btn and user_id and log_input:
    with col2:
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = []
        progress_bar = st.progress(0)
        
        try:
            # [수정] 반복 횟수를 2회로 단축
            for i in range(2):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "너는 CRP 분석 시스템이다. [DATA] 섹션에 '지표명: 숫자' 형식으로 0-10 사이 점수만 써라. 지표: MTI, REC, RECON, ORC, SRR, TTR, POI. 그 후 [INSIGHT] 섹션에 해설을 써라."},
                        {"role": "user", "content": f"ID: {user_id}\nLog:\n{log_input}"}
                    ],
                    temperature=0.1
                )
                text = response.choices[0].message.content
                
                # 정밀 추출 및 10점 만점 정규화
                data_match = re.search(r"\[DATA\](.*?)\[INSIGHT\]", text, re.S)
                if data_match:
                    vals = re.findall(r"(?:MTI|REC|RECON|ORC|SRR|TTR|POI):\s*(\d+\.?\d*)", data_match.group(1), re.I)
                    if len(vals) >= 7:
                        num_vals = [float(v)/10 if float(v) > 10 else float(v) for v in vals[:7]]
                        results.append(num_vals + [text])
                progress_bar.progress((i + 1) * 50)
            
            if results:
                df = pd.DataFrame([r[:7] for r in results], columns=['MTI', 'Rec', 'Recon', 'Orc', 'SRR', 'TTR', 'POI'])
                avg = df.mean()

                st.success("✅ 분석 완료")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Rec', 'Recon', 'Orc'], [avg['Rec'], avg['Recon'], avg['Orc']], color=['#3498db', '#e74c3c', '#2ecc71'])
                ax.set_ylim(0, 10)
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png')
                plt.close(fig)
                st.image(img_buf)

                final_insight = results[-1][-1].split("[INSIGHT]")[-1].strip()
                st.markdown("### 💡 전문가 종합 해설")
                st.write(final_insight)

                pdf_bytes = create_ensemble_pdf(final_insight, user_id, analysis_time, img_buf, avg)
                st.download_button("📄 PDF 리포트 다운로드", data=pdf_bytes, file_name=f"SKIM_{user_id}.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"오류 발생: {e}")