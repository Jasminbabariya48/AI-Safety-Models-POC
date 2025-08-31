import os
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'abuse_detector.joblib')

def eval_abuse(test_csv=None):
    model = joblib.load(MODEL_PATH)
    if test_csv is None:
        test_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'abuse_dataset.csv')
    df = pd.read_csv(test_csv)
    X = df['text_clean'].astype(str)
    y = df['label'].astype(int)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    return classification_report(y, y_pred, output_dict=True)

if __name__ == "__main__":
    eval_abuse()
