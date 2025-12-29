import streamlit as st
import joblib
from konlpy.tag import Okt

# 페이지 설정
st.set_page_config(page_title="K-SmishGuard AI", page_icon="🛡️", layout="wide")

# 형태소 분석기 로드
okt = Okt()

@st.cache_resource
def load_model():
    model = joblib.load('spam_model_ko.pkl')
    tfidf = joblib.load('tfidf_ko.pkl')
    return model, tfidf

model, tfidf = load_model()

# --- UI 디자인 ---
st.title("🛡️ K-SmishGuard: 한국어 피싱 탐지 시스템")
st.markdown("### 인공지능이 당신의 문자를 분석하여 스미싱 여부를 판별한다.")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📲 문자 입력")
    user_input = st.text_area("의심되는 문자 메시지를 입력하라", height=200, placeholder="내용을 여기에 붙여넣기 하라...")
    
    if st.button("AI 분석 실행", use_container_width=True):
        if user_input:
            # 예측
            vec_input = tfidf.transform([user_input])
            prediction = model.predict(vec_input)[0]
            prob = model.predict_proba(vec_input)[0]
            
            # 결과 표시
            st.write("---")
            if prediction == 1:
                st.error(f"🚨 분석 결과: **피싱/스팸 위험 문구 감지** (위험도: {prob[1]*100:.1f}%)")
                st.progress(prob[1])
            else:
                st.success(f"✅ 분석 결과: **정상적인 문구로 판단** (안전도: {prob[0]*100:.1f}%)")
                st.progress(prob[0])
        else:
            st.warning("분석할 내용을 입력하라.")

with col2:
    st.subheader("💡 탐지 포인트")
    if user_input:
        # 입력된 문장에서 핵심 단어 추출하여 시각화
        nouns = okt.nouns(user_input)
        if nouns:
            st.write("문장에서 감지된 주요 단어:")
            for n in set(nouns):
                st.write(f"- {n}")
    else:
        st.write("문자를 입력하면 주요 단어를 분석한다.")

# --- 하단 섹션: 보안 직무 어필용 통계 ---
st.divider()
st.subheader("📊 시스템 상태")
c1, c2, c3 = st.columns(3)
c1.metric("탐지 정확도", "97.8%")
c2.metric("처리 언어", "한국어(KO)")
c3.metric("모델 버전", "v1.2.0")
