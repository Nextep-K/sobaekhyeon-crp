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
st.title("🧠 소백현: CRP 분석 엔진 (V 1.8.0)")
st.markdown("3회 교차 검증 **앙상블**을 통해 신뢰도를 높이고 인지 분석 리포트를 생성합니다.")
st.divider()

INDICATORS = ['MTI', 'REC', 'RECON', 'ORC', 'SRR', 'TTR', 'POI']
ENSEMBLE_N = 3  # 2회보다 통계적 의미 있고, 5회보다 토큰 효율적


# --- [개선 1] 정규화: 0~10 범위 강제 클램핑 ---
def normalize_score(v: float) -> float:
    """어떤 척도로 오더라도 0~10으로 안전하게 정규화"""
    if v > 100:
        v = v / 10  # 1000점 척도 방어
    elif v > 10:
        v = v / 10  # 100점 척도
    return max(0.0, min(10.0, round(v, 2)))


# --- [개선 2] 파싱: 실패 시 명시적 예외 발생 ---
def parse_response(text: str) -> tuple[list[float], str]:
    """
    GPT 응답에서 점수와 인사이트를 추출.
    형식 불일치 시 ValueError를 raise해 호출부에서 처리하게 함.
    """
    data_match = re.search(r"\[DATA\](.*?)\[INSIGHT\]", text, re.S | re.I)
    if not data_match:
        raise ValueError("응답에서 [DATA]...[INSIGHT] 구조를 찾을 수 없습니다.")

    raw_vals = re.findall(
        r"(?:MTI|REC|RECON|ORC|SRR|TTR|POI)\s*:\s*(\d+\.?\d*)",
        data_match.group(1),
        re.I
    )
    if len(raw_vals) < len(INDICATORS):
        raise ValueError(
            f"지표 {len(INDICATORS)}개 필요, {len(raw_vals)}개만 파싱됨. 응답:\n{text[:300]}"
        )

    scores = [normalize_score(float(v)) for v in raw_vals[:len(INDICATORS)]]
    insight = text.split("[INSIGHT]", 1)[-1].strip()

    if not insight:
        raise ValueError("INSIGHT 섹션이 비어 있습니다.")

    return scores, insight


# --- PDF 생성 ---
def create_ensemble_pdf(content: str, u_id: str, time: str,
                        chart_img: io.BytesIO, avg: pd.Series) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font('Nanum', '', 'NanumGothic.ttf')
        pdf.add_font('Nanum', 'B', 'NanumGothicBold.ttf')
        tf = ('Nanum', '', 11)
        bf = ('Nanum', 'B', 16)
        sf = ('Nanum', 'B', 12)
    except Exception:
        tf = ("Helvetica", '', 11)
        bf = ("Helvetica", 'B', 16)
        sf = ("Helvetica", 'B', 12)

    pdf.set_font(*bf)
    pdf.cell(0, 15, "SKIM Ensemble Cognitive Report", ln=True, align='C')
    pdf.set_font(*tf)
    pdf.cell(0, 5, f"ID: {u_id} | Date: {time} | Ensemble N={ENSEMBLE_N}", ln=True, align='C')
    pdf.ln(10)

    # [개선 3] img_buf seek(0) 명시적 보장
    chart_img.seek(0)
    pdf.image(chart_img, x=15, y=45, w=140)
    pdf.ln(95)

    pdf.set_font(*sf)
    pdf.cell(0, 10, " [ Ensemble Analytics Summary ]", ln=True)
    pdf.set_font(*tf)
    for k in ['MTI', 'Rec', 'Recon', 'Orc']:
        if k in avg:
            pdf.cell(0, 8, f"• Average {k}: {avg[k]:.2f} / 10.0", ln=True)
    pdf.ln(5)

    pdf.set_font(*sf)
    pdf.cell(0, 10, " [ Expert Insight ]", ln=True)
    pdf.set_font(*tf)
    pdf.multi_cell(0, 8, txt=content.replace('#', '').replace('*', '').strip())

    pdf_out = pdf.output(dest='S')
    return bytes(pdf_out) if isinstance(pdf_out, (str, bytearray)) else pdf_out


# --- 메인 UI ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("👤 학습자 정보")
    user_id = st.text_input("학습자 ID", max_chars=10, value="sj")
    log_input = st.text_area("분석할 [Conversation Log] 입력", height=500)
    analyze_btn = st.button(
        f"🚀 앙상블 분석 ({ENSEMBLE_N}회 교차 검증)",
        use_container_width=True
    )

if analyze_btn and user_id and log_input:
    with col2:
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scores_list: list[list[float]] = []
        last_insight = ""
        errors: list[str] = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(ENSEMBLE_N):
            status_text.text(f"분석 중... ({i+1}/{ENSEMBLE_N}회)")
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "너는 CRP 분석 시스템이다.\n"
                                "반드시 아래 형식만 사용하라:\n"
                                "[DATA]\n"
                                "MTI: <0-10>\nREC: <0-10>\nRECON: <0-10>\n"
                                "ORC: <0-10>\nSRR: <0-10>\nTTR: <0-10>\nPOI: <0-10>\n"
                                "[INSIGHT]\n<한국어 전문가 해설>"
                            )
                        },
                        {"role": "user", "content": f"ID: {user_id}\nLog:\n{log_input}"}
                    ],
                    temperature=0.1
                )
                text = response.choices[0].message.content

                # [개선 2] 파싱 실패를 명시적으로 포착
                scores, insight = parse_response(text)
                scores_list.append(scores)
                last_insight = insight

            except ValueError as ve:
                errors.append(f"회차 {i+1} 파싱 오류: {ve}")
            except Exception as e:
                errors.append(f"회차 {i+1} API 오류: {e}")

            progress_bar.progress(int((i + 1) / ENSEMBLE_N * 100))

        status_text.empty()

        # 오류 요약 표시
        if errors:
            with st.expander(f"⚠️ 오류 {len(errors)}건 발생 (클릭하여 확인)"):
                for err in errors:
                    st.warning(err)

        # 유효 결과가 하나라도 있으면 진행
        if scores_list:
            df = pd.DataFrame(scores_list, columns=INDICATORS)
            # 컬럼명 소문자 첫글자 통일 (PDF avg_data 호환)
            df.columns = ['MTI', 'Rec', 'Recon', 'Orc', 'SRR', 'TTR', 'POI']
            avg = df.mean()

            st.success(f"✅ {len(scores_list)}/{ENSEMBLE_N}회 분석 완료 (유효 앙상블)")

            # 표준편차 신뢰도 표시 (앙상블 의미 강화)
            if len(scores_list) > 1:
                std = df.std()
                st.caption(
                    "📊 앙상블 표준편차 — "
                    + " | ".join(f"{c}: ±{std[c]:.2f}" for c in ['Rec', 'Recon', 'Orc'])
                )

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                ['Rec', 'Recon', 'Orc'],
                [avg['Rec'], avg['Recon'], avg['Orc']],
                color=['#3498db', '#e74c3c', '#2ecc71'],
                yerr=[std['Rec'], std['Recon'], std['Orc']] if len(scores_list) > 1 else None,
                capsize=5
            )
            ax.set_ylim(0, 10)
            ax.set_title("Ensemble Average (with std)")

            # [개선 3] img_buf를 차트 저장 직후 seek(0)으로 초기화
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            plt.close(fig)
            img_buf.seek(0)  # ← 명시적 seek
            st.image(img_buf)

            st.markdown("### 💡 전문가 종합 해설")
            st.write(last_insight)

            pdf_data = create_ensemble_pdf(
                last_insight, user_id, analysis_time, img_buf, avg
            )

            if pdf_data:
                st.download_button(
                    label="📄 PDF 리포트 다운로드",
                    data=pdf_data,
                    file_name=f"SKIM_{user_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.error("❌ 모든 회차에서 분석 실패. 로그 형식 또는 API 상태를 확인하세요.")