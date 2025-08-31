import re
import joblib
import os
from typing import List

def simple_clean(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s.,!?\'"]+', ' ', text)
    return text.strip()

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)