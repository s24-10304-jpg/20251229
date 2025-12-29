import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 페이지 설정
st.set_page_config(page_title="스팸 문자 분석기", layout="wide")

@st.cache_data
def load_and_train_model(file_path):
    # 데이터 로드 (인코딩 문제 해결을 위해 latin-1 사용)
    df = pd.read_csv(file_path, encoding='latin-1')
    # 필요한 컬럼만 추출 및 이름 변경
    df = df[['v1', 'v2']]
    df.columns = ['target', 'text']
    
    # 타겟 라벨링 (ham: 0, spam: 1)
    df['label'] = df['target'].map({'ham': 0, 'spam': 1})
    
    # TF-IDF 벡터화
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X = tfidf.fit_transform(df['text'])
    y = df['label']
    
    # 모델 학습
    model = LogisticRegression()
    model.fit(X, y)
    
    return tfidf, model

# 데이터 로드 및 모델 학습
try:
    tfidf, model = load_and_train_model('spam.csv')
except FileNotFoundError:
    st.error("파일을 찾을 수 없습니다. 'spam.csv'가 같은 경로에 있는지 확인해주세요.")
    st.stop()

# UI 구성
st.title("🚫 스팸 문자 AI 분석기")
st.write("문장 속에 숨겨진 위험 요소를 인공지능이 분석해 드립니다.")

user_input = st.text_area("분석할 문자 메시지를 입력하세요:", placeholder="예: Winner! Claim your prize now by calling 09061701461.")

if st.button("분석 시작"):
    if user_input:
        # 1. 예측
        vec_input = tfidf.transform([user_input])
        prediction = model.predict(vec_input)[0]
        probability = model.predict_proba(vec_input)[0]

        # 결과 표시
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            if prediction == 1:
                st.error(f"### 결과: ⚠️ 스팸(Spam)일 확률이 높습니다!")
            else:
                st.success(f"### 결과: ✅ 정상(Ham) 문자입니다.")
            
            st.metric("스팸 확률", f"{probability[1]*100:.2f}%")

        # 2. 시각적 근거 분석 (Feature Importance)
        with col2:
            st.write("### 📊 왜 그렇게 판단했나요?")
            
            # 입력 문장에 포함된 단어들의 가중치 추출
            feature_names = np.array(tfidf.get_feature_names_out())
            coeffs = model.coef_[0]
            
            # 현재 문장에 포함된 단어 필터링
            words_in_text = tfidf.inverse_transform(vec_input)[0]
            if len(words_in_text) > 0:
                word_weights = []
                for word in words_in_text:
                    idx = np.where(feature_names == word)[0][0]
                    word_weights.append((word, coeffs[idx]))
                
                # 가중치 기준 정렬
                word_weights.sort(key=lambda x: x[1], reverse=True)
                df_weights = pd.DataFrame(word_weights, columns=['단어', '위험도 가중치'])

                # 시각화
                fig, ax = plt.subplots()
                sns.barplot(data=df_weights, x='위험도 가중치', y='단어', palette='coolwarm', ax=ax)
                plt.title("단어별 스팸 기여도 (높을수록 위험)")
                st.pyplot(fig)
                
                st.caption("위 그래프에서 양수(빨간색 방향) 값이 큰 단어들이 스팸으로 판단하게 만든 주요 키워드입니다.")
            else:
                st.info("분석할 수 있는 유의미한 단어가 부족합니다.")
    else:
        st.warning("분석할 내용을 입력해주세요.")
