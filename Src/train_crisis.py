import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

DATA_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'crisis_dataset.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'crisis_detector.joblib')

def train():
    df = pd.read_csv(DATA_CSV)
    X = df['text_clean'].astype(str)
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Crisis detection report:\n", classification_report(y_test, y_pred))
    joblib.dump(pipeline, MODEL_OUT)
    print("Saved crisis detector to", MODEL_OUT)
    return pipeline

if __name__ == "__main__":
    train()
