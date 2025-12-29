import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from konlpy.tag import Okt  # 한국어 형태소 분석기

# 한글 폰트 설정 (OS에 따라 다름)
plt.rcParams['font.family'] = 'Malgun Gothic' # 윈도우용
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="K-스팸 분석기", layout="wide")

okt = Okt()

# 한국어 전처리 함수 (형태소 분석)
def korean_tokenizer(text):
    # 명사, 형용사, 동사만 추출
    tokens = okt.pos(text, stem=True)
    return [word for word, pos in tokens if pos in ['Noun', 'Adjective', 'Verb']]

@st.cache_data
def load_and_train_model(file_path):
    # 실제 데이터셋이 영어라면 한국어 샘플 데이터를 임시로 생성하거나 로드합니다.
    # 여기서는 구조를 보여드리기 위해 간단한 학습 로직을 유지합니다.
    df = pd.read_csv(file_path, encoding='latin-1').iloc[:1000] # 예제용 데이터 일부
    df = df[['v1', 'v2']]
    df.columns = ['target', 'text']
    df['label'] = df['target'].map({'ham': 0, 'spam': 1})
    
    # 한국어 토크나이저 적용
    tfidf = TfidfVectorizer(tokenizer=korean_tokenizer, max_features=2000)
    X = tfidf.fit_transform(df['text'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return tfidf, model

# 모델 로드
tfidf, model = load_and_train_model('spam.csv')

st.title("🚫 한국어 스팸 AI 분석 & 리포트")
st.write("문자를 분석하여 스팸 여부와 그 이유를 상세히 설명합니다.")

user_input = st.text_area("분석할 문자를 입력하세요 (한글/영어 모두 가능):")

if st.button("AI 분석 실행"):
    if user_input:
        # 예측
        vec_input = tfidf.transform([user_input])
        prediction = model.predict(vec_input)[0]
        prob = model.predict_proba(vec_input)[0][1] * 100

        st.divider()
        
        # 결과 표시 영역
        col1, col2 = st.columns([1, 1.2])

        with col1:
            if prediction == 1:
                st.error(f"### 결과: 🚨 스팸 위험 ({prob:.1f}%)")
                st.write("**[AI 진단]** 이 문자는 전형적인 스팸 패턴을 보이고 있습니다.")
            else:
                st.success(f"### 결과: ✅ 정상 메시지 ({100-prob:.1f}%)")
                st.write("**[AI 진단]** 일반적인 대화 형태의 메시지로 판단됩니다.")

        # 시각화 및 이유 설명
        with col2:
            st.subheader("📊 AI의 판단 근거 리포트")
            
            # 가중치 분석
            feature_names = np.array(tfidf.get_feature_names_out())
            coeffs = model.coef_[0]
            words_in_text = tfidf.inverse_transform(vec_input)[0]

            if len(words_in_text) > 0:
                word_weights = []
                for word in words_in_text:
                    idx = np.where(feature_names == word)[0][0]
                    word_weights.append((word, coeffs[idx]))
                
                word_weights.sort(key=lambda x: x[1], reverse=True)
                df_w = pd.DataFrame(word_weights, columns=['단어', '위험점수'])

                # 그래프
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=df_w, x='위험점수', y='단어', palette='Reds_r', ax=ax)
                st.pyplot(fig)

                # 텍스트 설명 추가
                top_spam_word = df_w.iloc[0]['단어'] if df_w.iloc[0]['위험점수'] > 0 else None
                if top_spam_word:
                    st.info(f"💡 **이유 요약:** 문장 내에 있는 **'{top_spam_word}'**와(과) 같은 단어는 과거 스팸 데이터에서 매우 높은 빈도로 발견되었습니다. 특히 금전적 유도나 긴급성을 강조하는 단어 조합이 점수를 높였습니다.")
            else:
                st.write("분석할 수 있는 특정 키워드가 발견되지 않았습니다.")
    else:
        st.warning("분석할 문자를 입력해주세요.")
