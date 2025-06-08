import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

news_df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

news_df['content'] = news_df['title'] + " " + news_df['text']

X = news_df['content']
y = news_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Model accuracy on test set:", model.score(X_test_tfidf, y_test))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
