import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 데이터 로드 및 전처리
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 벡터화 및 모델 학습
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X = tfidf.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# 파일 저장
joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
print("모델 및 벡터라이저 저장 완료!")
