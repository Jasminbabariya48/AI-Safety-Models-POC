from flask import Flask, request, jsonify, render_template_string
import os
import joblib
from escalation_detector import EscalationDetector
from content_filter import age_appropriate
from utils import simple_clean
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ABUSE_MODEL_PATH = os.path.join(MODEL_DIR, 'abuse_detector.joblib')
CRISIS_MODEL_PATH = os.path.join(MODEL_DIR, 'crisis_detector.joblib')

app = Flask(__name__)
escalation = EscalationDetector(window_size=6)
sent_analyzer = SentimentIntensityAnalyzer()

# Load models if present
abuse_model = None
crisis_model = None
if os.path.exists(ABUSE_MODEL_PATH):
    abuse_model = joblib.load(ABUSE_MODEL_PATH)
if os.path.exists(CRISIS_MODEL_PATH):
    crisis_model = joblib.load(CRISIS_MODEL_PATH)

@app.route('/')
def index():
    # minimal page to input conversation turns
    html = """
    <!doctype html><html><head><title>AI Safety POC Chat</title></head><body>
    <h2>AI Safety POC Chat Simulator</h2>
    <form action="/send" method="post">
      User age: <input name="age" value="20"/> <br/><br/>
      Message: <input name="message" size="80"/> <br/><br/>
      <input type="submit" value="Send"/>
    </form>
    <div id="out">
    {% if result %}
      <h3>Result</h3>
      <pre>{{result}}</pre>
    {% endif %}
    </div>
    </body></html>
    """
    return render_template_string(html)

@app.route('/send', methods=['POST'])
def send():
    msg = request.form.get('message','')
    age = int(request.form.get('age', 20))
    clean = simple_clean(msg)
    # abuse inference
    abuse_flag = False
    abuse_score = None
    if abuse_model:
        proba = abuse_model.predict_proba([clean])[0]
        # assume binary where class 1 is abusive
        abuse_score = float(proba[1])
        abuse_flag = abuse_score > 0.5
    # crisis inference
    crisis_flag = False
    crisis_score = None
    if crisis_model:
        crisis_score = float(crisis_model.predict_proba([clean])[0][1])
        crisis_flag = crisis_score > 0.5
    # content filter based on age
    cf = age_appropriate(clean, age)
    # escalation detector gets the new turn (we treat any abuse flag as contributing)
    escalation.add_turn(msg, abuse_flag=abuse_flag)
    esc_score = escalation.escalation_score()
    needs_escalation = escalation.needs_escalation()
    result = {
        'message': msg,
        'clean': clean,
        'abuse_score': abuse_score,
        'abuse_flag': abuse_flag,
        'crisis_score': crisis_score,
        'crisis_flag': crisis_flag,
        'content_filter': cf,
        'escalation_score': esc_score,
        'needs_escalation': needs_escalation
    }
    return render_template_string("""
    <!doctype html><html><body>
      <a href="/">Back</a>
      <h3>Result</h3>
      <pre>{{result}}</pre>
    </body></html>
    """, result=result)

@app.route('/api/message', methods=['POST'])
def api_message():
    payload = request.json
    msg = payload.get('message','')
    age = int(payload.get('age', 20))
    clean = simple_clean(msg)
    abuse_flag = False
    abuse_score = None
    if abuse_model:
        proba = abuse_model.predict_proba([clean])[0]
        abuse_score = float(proba[1])
        abuse_flag = abuse_score > 0.5
    crisis_flag = False
    crisis_score = None
    if crisis_model:
        crisis_score = float(crisis_model.predict_proba([clean])[0][1])
        crisis_flag = crisis_score > 0.5
    cf = age_appropriate(clean, age)
    escalation.add_turn(msg, abuse_flag=abuse_flag)
    esc_score = escalation.escalation_score()
    return jsonify({
        'message': msg,
        'clean': clean,
        'abuse_score': abuse_score,
        'abuse_flag': abuse_flag,
        'crisis_score': crisis_score,
        'crisis_flag': crisis_flag,
        'content_filter': cf,
        'escalation_score': esc_score,
        'needs_escalation': escalation.needs_escalation()
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
