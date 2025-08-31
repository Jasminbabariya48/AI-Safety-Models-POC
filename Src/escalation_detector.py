from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import deque
import numpy as np

analyzer = SentimentIntensityAnalyzer()

class EscalationDetector:
    def __init__(self, window_size=6):
        self.window_size = window_size
        self.recent = deque(maxlen=window_size)  # store tuples (text, sentiment_compound, abuse_flag)

    def add_turn(self, text, abuse_flag=False):
        s = analyzer.polarity_scores(text)
        compound = s['compound']
        self.recent.append((text, compound, abuse_flag))

    def escalation_score(self):
        if not self.recent:
            return 0.0
        compounds = np.array([c for (_, c, _) in self.recent])
        abuse_flags = np.array([1 if ab else 0 for (_, _, ab) in self.recent])
        # negative average + slope + abuse frequency
        avg_neg = -np.mean(np.minimum(compounds, 0))
        # slope of compound values (recent trend)
        x = np.arange(len(compounds))
        if len(compounds) >= 2:
            slope = np.polyfit(x, compounds, 1)[0]
        else:
            slope = 0.0
        abuse_freq = abuse_flags.mean()
        # Combine into score 0..1 (heuristic)
        score = (avg_neg * 0.6) + (max(-slope, 0) * 0.2) + (abuse_freq * 0.4)
        score = max(0.0, min(1.0, score))
        return float(score)

    def needs_escalation(self, threshold=0.5):
        return self.escalation_score() >= threshold
