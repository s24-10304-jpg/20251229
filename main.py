import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from kiwipiepy import Kiwi

# Kiwi 초기화 (한국어 분석용)
kiwi = Kiwi()

# 통합 토크나이저: 영어는 소문자화, 한국어는 형태소 분석
def smart_tokenizer(text):
    if pd.isna(text): return []
    # 한국어 형태소 추출
    tokens = kiwi.tokenize(text)
    korean_words = [t.form for t in tokens if t.tag in ['NNG', 'NNP', 'VV', 'VA']]
    # 영어 단어 추출 (알파벳 2글자 이상)
    import re
    english_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    return korean_words + english_words

@st.cache_data
def load_and_train_model(file_path):
    # 데이터 로드 및 결측치 제거
    df = pd.read_csv(file_path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['target', 'text']
    df = df.dropna()
    df['label'] = df['target'].map({'ham': 0, 'spam': 1})
    
    # 모델 학습 (smart_tokenizer 사용)
    tfidf = TfidfVectorizer(tokenizer=smart_tokenizer, max_features=3000, token_pattern=None)
    X = tfidf.fit_transform(df['text'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    return tfidf, model

# 앱 시작
st.title("🚫 AI 스팸 분석 및 근거 제시")

try:
    tfidf, model = load_and_train_model('spam.csv')
    
    user_input = st.text_area("분석할 문자를 입력하세요 (한글/영어 가능):")

    if st.button("스팸 여부 분석"):
        if user_input:
            vec = tfidf.transform([user_input])
            prob = model.predict_proba(vec)[0][1]
            
            # 결과 시각화
            col1, col2 = st.columns(2)
            with col1:
                st.metric("스팸 확률", f"{prob*100:.1f}%")
                if prob > 0.5:
                    st.error("🚨 스팸으로 의심됩니다!")
                else:
                    st.success("✅ 정상적인 메시지입니다.")

            # 왜 스팸인가? (시각적 근거)
            with col2:
                st.subheader("📊 핵심 단어 분석")
                feature_names = tfidf.get_feature_names_out()
                # 해당 문장에 포함된 단어들의 가중치 가져오기
                words_in_input = smart_tokenizer(user_input)
                weights = []
                for word in set(words_in_input):
                    if word in feature_names:
                        idx = np.where(feature_names == word)[0][0]
                        weights.append((word, model.coef_[0][idx]))
                
                if weights:
                    df_weights = pd.DataFrame(weights, columns=['단어', '위험도']).sort_values('위험도', ascending=False)
                    fig, ax = plt.subplots()
                    sns.barplot(data=df_weights, x='위험도', y='단어', palette='RdBu_r', ax=ax)
                    st.pyplot(fig)
                    
                    # 텍스트 설명
                    top_word = df_weights.iloc[0]['단어']
                    if df_weights.iloc[0]['위험도'] > 0:
                        st.info(f"💡 분석 결과, **'{top_word}'**와(과) 같은 단어가 스팸 판단에 가장 큰 영향을 주었습니다.")
                else:
                    st.write("학습된 단어가 없어 상세 분석이 어렵습니다.")
except Exception as e:
    st.error(f"오류 발생: {e}")
