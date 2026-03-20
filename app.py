import streamlit as st
from openai import OpenAI
import pandas as pd
import re
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

# 1. AI 열쇠 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 

st.set_page_config(page_title="SKIM Ensemble System", layout="wide")

st.title("🧠 소백현: 앙상블 CRP 분석 엔진")
st.markdown("교차 검증 **알고리듬**을 통해 인지적 편차를 최소화하고 시각적 리포트를 생성합니다.")
st.divider()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("👤 학습자 정보")
    user_id = st.text_input("학습자 ID (영문 7자 이하)", max_chars=7, placeholder="예: SJKIM")
    log_input = st.text_area("분석할 [Conversation Log] 입력", height=500)
    analyze_btn = st.button("🚀 앙상블 분석 시작 (3회 교차 검증)", use_container_width=True)

if analyze_btn and user_id and log_input:
    with col2:
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = []
        progress_bar = st.progress(0)
        
        try:
            for i in range(3):
                st.caption(f"🔄 {i+1}차 알고리듬 가동 중...")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "너는 CRP 기반 학습 분석 시스템이다. 모든 지표는 '지표명: 숫자' 형식으로 엄격히 출력하라."},
                        {"role": "user", "content": f"ID: {user_id}\n[Log]:\n{log_input}"}
                    ],
                    temperature=0.1
                )
                text = response.choices[0].message.content
                # 수치 추출 (MTI, SRI 3종, SAI 3종 순서)
                nums = re.findall(r": (\d+\.?\d*)", text)
                if len(nums) >= 7:
                    results.append([float(n) for n in nums[:7]] + [text])
                progress_bar.progress((i + 1) * 33)

            if results:
                # 데이터 프레임 생성 및 평균 계산
                df = pd.DataFrame([r[:7] for r in results], columns=['MTI', 'Rec', 'Recon', 'Orc', 'SRR', 'TTR', 'POI'])
                avg = df.mean()

                st.success(f"✅ 앙상블 분석 완료 (ID: {user_id})")
                
                # --- 시각화 (Matplotlib를 이용해 PDF 삽입용 이미지 생성) ---
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Recognition', 'Reconfiguration', 'Orchestration']
                values = [avg['Rec'], avg['Recon'], avg['Orc']]
                ax.bar(labels, values, color=['#3498db', '#e74c3c', '#2ecc71'])
                ax.set_title(f"SRI Cognitive Structure (Avg) - {user_id}")
                ax.set_ylim(0, 10)
                
                # 메모리에 그래프 저장
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png')
                st.image(img_buf) # 화면에 표시

                # 리포트 본문 (마지막 분석 기준)
                st.markdown("### 💡 전문가 종합 해설")
                st.write(results[-1][-1])

                # --- PDF 생성 (그래프 포함) ---
                def create_ensemble_pdf(content, u_id, time, chart_img, avg_data):
                    pdf = FPDF()
                    pdf.add_page()
                    try:
                        pdf.add_font('Nanum', '', 'NanumGothic.ttf')
                        pdf.set_font('Nanum', size=11)
                    except:
                        pdf.set_font("Helvetica", size=11)
                    
                    pdf.cell(0, 10, f"SKIM Ensemble Report: {u_id}", ln=True, align='C')
                    pdf.cell(0, 10, f"Date: {time}", ln=True, align='C')
                    pdf.ln(5)
                    
                    # 그래프 삽입
                    chart_img.seek(0)
                    pdf.image(chart_img, x=10, y=30, w=100)
                    pdf.ln(70) # 그래프 공간 확보
                    
                    # 평균 수치 요약
                    pdf.cell(0, 10, f"Average MTI Stage: {avg_data['MTI']:.1f}", ln=True)
                    pdf.ln(5)
                    
                    clean_text = content.replace('#', '').replace('*', '')
                    pdf.multi_cell(0, 8, txt=clean_text)
                    return bytes(pdf.output())

                pdf_bytes = create_ensemble_pdf(results[-1][-1], user_id, analysis_time, img_buf, avg)
                
                st.download_button(
                    label="📄 그래프 포함 리포트 PDF 다운로드",
                    data=pdf_bytes,
                    file_name=f"SKIM_Ensemble_{user_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"앙상블 알고리듬 실행 오류: {e}")