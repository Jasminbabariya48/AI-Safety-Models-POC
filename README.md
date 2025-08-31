# Technical Report — AI Safety Models POC

## 1. Objective & Scope
This POC demonstrates a modular suite of AI Safety Models to improve user safety in conversational systems, covering:
- Abuse Language Detection (binary abusive vs non-abusive)
- Escalation Pattern Recognition (conversation-level escalation score)
- Crisis Intervention (detection of self-harm indicators)
- Content Filtering (age-appropriate rules and blocking)

This project is a CPU-first POC that emphasizes modularity, interpretability, and a straightforward integration path into production systems.

## 2. Data Sources & Preprocessing
For demonstration we use publicly available datasets where possible via the `datasets` library. The POC includes:
- A public hate-speech dataset if available (`hate_speech18`, `tweet_eval/hate`, or similar) for abuse detection (preprocessed to text/label).
- Synthetic / small crisis dataset (for demonstration of pipeline).
Preprocessing steps:
- Lowercasing, URL removal, special character normalization.
- Tokenization via NLTK for basic text handling.
- TF-IDF vectorization (1-2 grams, cap features) for lightweight models.

Data privacy note: only publicly-available anonymized datasets are used. No PII is stored or included.

## 3. Model Architectures & Training
### Abuse Detection
- Model: TF-IDF vectorizer + Logistic Regression (CPU-friendly).
- Rationale: quick to train, interpretable weights, low-latency inference. Suitable for a POC and for on-device/edge deployment.
- Optional: fine-tune `distilbert-base-uncased` for production accuracy (trade-off: requires GPU / longer training).

### Crisis Detection
- Model: TF-IDF + Logistic Regression trained on a small suicide/self-harm labeled dataset for demo.
- Also includes keyword-based rules for highest-sensitivity triggers (e.g., "kill myself", "want to die").

### Escalation Recognition
- Approach: rule-based rolling window using sentiment intensity (VADER) plus abuse frequency and trend detection.
- Score computed as a normalized combination of negative sentiment average, negative slope (increasing negativity), and abuse frequency.

### Content Filtering
- Rule-based age gating: specific keywords trigger block/flag/allow decisions based on `user_age`.

## 4. Integration & Real-time Design
- Integration demonstrated using a Flask chat simulator `src/app.py`.
- Each incoming message is processed near-real-time:
  1. Clean text
  2. Abuse classifier (score + flag)
  3. Crisis classifier (score + flag)
  4. Age-based content filter (action)
  5. Escalation detector updates conversation context and computes escalation score
  6. Combined decisioning: if crisis_flag OR escalation_score > threshold → recommend human intervention.

Latency and scalability:
- TF-IDF + LR inference is very fast on CPU (<10ms typical for short text).
- For production scale, wrap models into inference microservices (REST/gRPC), use batching, cache features for repeated users, and autoscale based on load.

## 5. Evaluation
- Scripts provided (`src/evaluate.py`) compute precision/recall/F1 on held-out sets.
- For POC, metrics are illustrative. Production requires larger carefully-labeled evaluation sets and stratified sampling for demographic fairness.

## 6. Ethical Considerations
- Bias mitigation: training with balanced datasets, test on representative demographic slices, and add model explainability (feature importance for TF-IDF).
- Human-in-the-loop: automatic blocking is avoided for high-impact events; instead, models flag content for rapid human review and escalation.
- Privacy: logs should be redacted and PII removed; only required metadata stored; retention policies enforced.
- Transparency: produce clear audit trails for moderation actions, and provide avenues for appeals.

## 7. Leadership & Team Strategy
- Iterative roadmap:
  1. Start with TF-IDF pipelines and rule-based systems for fast iteration.
  2. Collect high-quality labeled data from real conversations (with user consent), focusing on edge cases.
  3. Add transformer-based models and deploy via staged rollout (A/B tests).
  4. Build robust evaluation and monitoring dashboards (data drift, false positives/negatives).
- Team composition: NLP engineers, ML infra, data annotators, product manager, privacy & legal counsel, human moderators.
- Risk mitigation: regular bias audits, privacy reviews, and human oversight.

## 8. Limitations & Improvements
- Limitations:
  - Small demo datasets limit generalization.
  - Heuristics may misclassify sarcasm, multilingual text, and slang.
- Improvements:
  - Fine-tune transformers with domain-specific data.
  - Add multi-lingual pipelines, slang normalization, and user-behavior signals.
  - Add a feedback loop to capture human moderation decisions as labeled data.

## 9. Conclusion
This POC demonstrates a practical, modular approach to conversational safety that is interpretable and easily extended. The design prioritizes fast experimentation and safe human oversight while providing a clear path to production-grade models.
