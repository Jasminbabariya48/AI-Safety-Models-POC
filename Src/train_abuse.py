import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from utils import save_model
import joblib

DATA_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'abuse_dataset.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'abuse_detector.joblib')

def train():
    df = pd.read_csv(DATA_CSV)
    X = df['text_clean'].astype(str)
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification report:", classification_report(y_test, y_pred))
    # Save model and metadata
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    print("Saved abuse detector to", MODEL_OUT)
    return pipeline, report

if __name__ == "__main__":
    train()
