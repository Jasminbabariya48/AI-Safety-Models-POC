import os
import pandas as pd
from utils import simple_clean

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def prepare_abuse_dataset(output_csv=os.path.join(DATA_DIR, 'abuse_dataset.csv')):
    os.makedirs(DATA_DIR, exist_ok=True)
    # Enhanced synthetic abuse detection dataset
    data = [
        ("I love you", 0),
        ("You are an idiot", 1),
        ("Have a nice day", 0),
        ("I will kill you", 1),
        ("This is awesome", 0),
        ("You're worthless", 1),
        ("You are stupid", 1),
        ("That's amazing!", 0),
        ("I hate you", 1),
        ("Great job!", 0),
        ("You're a moron", 1),
        ("Wonderful news!", 0),
        ("I will hurt you", 1),
        ("Fantastic work", 0),
        ("You're disgusting", 1),
        ("Excellent performance", 0),
        ("You should die", 1),
        ("Beautiful day today", 0),
        ("Nobody likes you", 1),
        ("Keep up the good work", 0)
    ]
    df = pd.DataFrame(data, columns=['text','label'])
    df['text_clean'] = df['text'].astype(str).apply(simple_clean)
    df.to_csv(output_csv, index=False)
    print(f"Saved abuse dataset to {output_csv} with {len(df)} samples")
    return output_csv

def prepare_crisis_dataset(output_csv=os.path.join(DATA_DIR, 'crisis_dataset.csv')):
    os.makedirs(DATA_DIR, exist_ok=True)
    # Enhanced synthetic crisis detection dataset
    rows = [
        ("I want to kill myself", 1),
        ("I feel like dying", 1),
        ("I am so sad and hopeless", 1),
        ("I can't go on anymore", 1),
        ("Life is not worth living", 1),
        ("I had a bad day", 0),
        ("Everything is fine", 0),
        ("I feel alone but I'll be okay", 0),
        ("Today was tough but manageable", 0),
        ("I'm feeling great today", 0),
        ("I need help with my homework", 0),
        ("I'm thinking about ending it all", 1),
        ("Nobody would care if I was gone", 1),
        ("The pain is too much to bear", 1),
        ("I'm really struggling with life", 1),
        ("Things will get better tomorrow", 0),
        ("I have so much to live for", 0),
        ("I can't handle this pain anymore", 1),
        ("Looking forward to the weekend", 0),
        ("I see no way out of this situation", 1)
    ]
    df = pd.DataFrame(rows, columns=['text','label'])
    df['text_clean'] = df['text'].apply(simple_clean)
    df.to_csv(output_csv, index=False)
    print(f"Saved crisis dataset to {output_csv} with {len(df)} samples")
    return output_csv

if __name__ == "__main__":
    prepare_abuse_dataset()
    prepare_crisis_dataset()