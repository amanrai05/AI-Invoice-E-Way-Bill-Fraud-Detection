"""
api.py — REST API for real-time fraud scoring
Author : Aman Kumar | Roll No: 2310991770
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
import pandas as pd
from fraud_detector import InvoiceFraudDetector

app = Flask(__name__)
detector = None


def load_or_train_detector():
    global detector
    model_path = 'models/detector.pkl'
    if os.path.exists(model_path):
        print(f"[API] Loading model from {model_path}")
        detector = InvoiceFraudDetector.load(model_path)
    else:
        print("[API] Training fresh model with sample data...")
        from data_generator import generate_sample_data
        os.makedirs('data/sample', exist_ok=True)
        df_train, _, labels, _ = generate_sample_data(n_train=600, n_test=50)
        detector = InvoiceFraudDetector()
        detector.fit(df_train, labels)
        detector.save(model_path)
    print("[API] Detector ready.")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': detector is not None})


@app.route('/score', methods=['POST'])
def score_single():
    """
    Score a single invoice.
    Body: JSON object representing one invoice row.
    """
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    df = pd.DataFrame([data])
    results = detector.predict(df)
    row = results.iloc[0].to_dict()
    explanation = detector.explain(row)

    return jsonify({
        'invoice_number':   row['invoice_number'],
        'final_risk_score': row['final_risk_score'],
        'risk_level':       row['risk_level'],
        'flag_for_review':  bool(row['flag_for_review']),
        'triggered_rules':  row['triggered_rules'],
        'rule_score':       row['rule_score'],
        'anomaly_score':    row['anomaly_score'],
        'ml_score':         row['ml_score'],
        'explanation':      explanation
    })


@app.route('/score_batch', methods=['POST'])
def score_batch():
    """
    Score a batch of invoices.
    Body: JSON array of invoice objects.
    """
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({'error': 'Expected JSON array'}), 400

    df = pd.DataFrame(data)
    results = detector.predict(df)
    flagged_count = int(results['flag_for_review'].sum())

    return jsonify({
        'total': len(results),
        'flagged': flagged_count,
        'results': results.to_dict(orient='records')
    })


@app.route('/rules', methods=['GET'])
def list_rules():
    from fraud_detector import RULES
    return jsonify(RULES)


@app.route('/explain/<invoice_number>', methods=['GET'])
def explain_invoice(invoice_number):
    """Returns explanation for a previously scored invoice stored in output CSV."""
    out_path = 'output/detection_results.csv'
    if not os.path.exists(out_path):
        return jsonify({'error': 'No results found. Run /score first.'}), 404

    df = pd.read_csv(out_path)
    match = df[df['invoice_number'] == invoice_number]
    if match.empty:
        return jsonify({'error': f'Invoice {invoice_number} not found'}), 404

    row = match.iloc[0].to_dict()
    return jsonify({'explanation': detector.explain(row), 'data': row})


if __name__ == '__main__':
    load_or_train_detector()
    app.run(host='0.0.0.0', port=5000, debug=False)
