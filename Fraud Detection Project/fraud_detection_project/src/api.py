"""
api.py — Professional REST API and Web Dashboard Server
Author : Aman Kumar | Roll No: 2310991770
"""
import os, sys, json
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from werkzeug.utils import secure_filename

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from fraud_detector import InvoiceFraudDetector

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'data/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('output', exist_ok=True)

detector = None

def load_detector():
    global detector
    model_path = 'models/detector.pkl'
    if os.path.exists(model_path):
        print("[API] Loading trained model...")
        detector = InvoiceFraudDetector.load(model_path)
    else:
        print("[API] No model found, training with sample data...")
        from data_generator import generate_sample_data
        df_train, _, labels, _ = generate_sample_data(n_train=400, n_test=50)
        detector = InvoiceFraudDetector()
        detector.fit(df_train, labels)
        detector.save(model_path)
    print("[API] System Ready.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    out_path = 'output/detection_results.csv'
    if not os.path.exists(out_path):
        return jsonify({'total': 0, 'critical': 0, 'high': 0, 'flagged_rate': 0, 'total_value': '0'})
    
    df = pd.read_csv(out_path)
    total = len(df)
    if total == 0:
        return jsonify({'total': 0, 'critical': 0, 'high': 0, 'flagged_rate': 0, 'total_value': '0'})
        
    critical = int((df['risk_level'] == 'CRITICAL').sum())
    high = int((df['risk_level'] == 'HIGH').sum())
    flagged = int(df['flag_for_review'].sum())
    
    return jsonify({
        'total': total,
        'critical': critical,
        'high': high,
        'flagged_rate': round(100 * flagged / total, 1),
        'total_value': f"₹{df['invoice_amount'].sum():,.0f}"
    })

@app.route('/api/invoices', methods=['GET'])
def get_invoices():
    out_path = 'output/detection_results.csv'
    if not os.path.exists(out_path):
        return jsonify([])
    
    df = pd.read_csv(out_path)
    # Return all flagged invoices
    flagged = df[df['flag_for_review'] == True].sort_values('final_risk_score', ascending=False)
    return jsonify(flagged.to_dict(orient='records'))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process with AI Detector
        try:
            df = pd.read_csv(filepath)
            results = detector.predict(df)
            results.to_csv('output/detection_results.csv', index=False)
            return jsonify({
                'status': 'success', 
                'count': len(results),
                'flagged': int(results['flag_for_review'].sum())
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV.'}), 400

@app.route('/api/explain/<invoice_number>')
def explain(invoice_number):
    out_path = 'output/detection_results.csv'
    if not os.path.exists(out_path):
        return jsonify({'error': 'No data found'}), 404
    
    df = pd.read_csv(out_path)
    match = df[df['invoice_number'] == invoice_number]
    if match.empty:
        return jsonify({'error': 'Invoice not found'}), 404
    
    row = match.iloc[0].to_dict()
    explanation = detector.explain(row)
    return jsonify({'explanation': explanation, 'data': row})

if __name__ == '__main__':
    load_detector()
    app.run(host='0.0.0.0', port=5000, debug=False)
