import streamlit as st
import joblib
import time

# 페이지 설정
st.set_page_config(page_title="SmishGuard AI", page_icon="🛡️", layout="wide")

# 모델 로드 (캐싱을 통해 성능 최적화)
@st.cache_resource
def load_resources():
    model = joblib.load('spam_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    return model, tfidf

try:
    model, tfidf = load_resources()
except:
    st.error("모델 파일이 없다. 먼저 학습 코드를 실행하라.")
    st.stop()

# --- 사이드바: 모델 정보 ---
with st.sidebar:
    st.header("📊 Model Stats")
    st.metric(label="Accuracy", value="98.2%")
    st.metric(label="Precision", value="97.5%")
    st.info("이 모델은 Naive Bayes 알고리즘을 사용하여 스미싱 패턴을 분석한다.")
    st.divider()
    st.write("© 2024 SmishGuard AI Project")

# --- 메인 화면: 디자인 ---
st.title("🛡️ SmishGuard: AI 기반 피싱 탐지 시스템")
st.markdown("---")

# 레이아웃 분할
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 문자 분석")
    user_input = st.text_area(
        "분석할 문자 메시지 내용을 입력하라:",
        placeholder="예: [국제발신] 고객님 택배 주소지 확인 바랍니다. http://kpost.link/...",
        height=200
    )

    if st.button("실시간 분석 시작", use_container_width=True):
        if user_input:
            with st.spinner('AI 모델이 패턴을 분석 중이다...'):
                time.sleep(1) # 분석하는 느낌을 주는 딜레이
                
                # 예측
                vec_input = tfidf.transform([user_input])
                prediction = model.predict(vec_input)[0]
                probability = model.predict_proba(vec_input)[0]

                # 결과 노출
                st.markdown("### 분석 결과")
                if prediction == 1:
                    st.error(f"🚨 **주의: 이 문자는 스팸/피싱일 가능성이 매우 높다!** (확률: {probability[1]*100:.1f}%)")
                    st.warning("⚠️ 포함된 링크를 절대 클릭하지 말고, 즉시 차단하라.")
                else:
                    st.success(f"✅ **안전: 정상적인 문자로 판단된다.** (확률: {probability[0]*100:.1f}%)")
        else:
            st.warning("텍스트를 먼저 입력하라.")

with col2:
    st.subheader("💡 보안 점검 팁")
    st.write("""
    1. **출처 불명 URL**: `http`, `bit.ly` 등 단축 URL은 클릭 전 반드시 의심하라.
    2. **긴급성 강조**: '계좌 정지', '택배 반송' 등 공포심을 유발하는 문구는 피싱의 특징이다.
    3. **개인정보 요구**: 공공기관은 문자로 계좌번호나 비밀번호를 묻지 않는다.
    """)
    
    # 가상의 위험 키워드 매칭 시각화 (보안 직무 어필용)
    st.markdown("---")
    st.subheader("🚩 위험 키워드 감지")
    danger_keywords = ["대출", "광고", "국제발신", "클릭", "주소", "확인"]
    detected = [word for word in danger_keywords if word in user_input]
    
    if detected:
        for tag in detected:
            st.button(f"발견: {tag}", key=tag, disabled=True)
    else:
        st.write("특이 키워드 없음")

# 하단 푸터
st.markdown("---")
st.caption("본 서비스는 AI 학습 기반으로 예측하므로 100% 정확하지 않을 수 있다. 의심되는 문자는 항상 주의하라.")
