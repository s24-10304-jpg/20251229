import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from konlpy.tag import Okt
import joblib

# 형태소 분석기 초기화
okt = Okt()

def tokenizer(text):
    # 한국어에서 의미 있는 단어(명사, 동사 등)만 추출
    return [word for word, pos in okt.pos(text) if pos in ['Noun', 'Verb', 'Adjective']]

# 1. 데이터 로드 (한국어 데이터는 utf-8-sig 또는 cp949 인코딩이 많음)
try:
    df = pd.read_csv('spam_ko.csv', encoding='utf-8-sig')
except:
    # 예시 데이터가 없을 경우를 대비한 간단한 샘플 데이터 생성
    data = {
        'label': [1, 1, 0, 0],
        'text': ['[광고] 무료 거부 080 대출 상담', '고객님 당첨되셨습니다 클릭하세요', '오늘 점심 뭐 먹을래?', '엄마 나 학원 끝났어']
    }
    df = pd.DataFrame(data)

# 2. 벡터화 (한국어 토크나이저 적용)
tfidf = TfidfVectorizer(tokenizer=tokenizer)
X = tfidf.fit_transform(df['text'])
y = df['label']

# 3. 모델 학습 및 저장
model = MultinomialNB()
model.fit(X, y)

joblib.dump(model, 'spam_model_ko.pkl')
joblib.dump(tfidf, 'tfidf_ko.pkl')
print("한국어 모델 저장 완료!")
