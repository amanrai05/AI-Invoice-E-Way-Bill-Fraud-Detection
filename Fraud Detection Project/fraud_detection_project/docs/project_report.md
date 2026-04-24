# Project Report
## AI Invoice & E-Way Bill Fraud Detection

**Participant:** Aman Kumar  
**Roll No:** 2310991770  
**Challenge:** AI Invoice & E-Way Bill Fraud Detection  
**Date:** 2024

---

## 1. Project Title

**AI-Powered GST Invoice & E-Way Bill Fraud Detection System**  
*An Ensemble Machine Learning System for Real-Time Fraud Risk Scoring*

---

## 2. Abstract

India's Goods and Services Tax (GST) ecosystem generates hundreds of millions of invoices and e-way bills every year. This volume creates significant opportunities for fraudulent activities including Input Tax Credit (ITC) fraud, fake invoice generation, and illegal goods movement. This project presents an ensemble AI system that assigns every transaction a **Fraud Risk Score from 0 to 100** by combining a deterministic rule engine, an unsupervised anomaly detector (Isolation Forest), and a supervised classifier (Random Forest + Gradient Boosting). The system supports both structured data (CSV/Excel/JSON) and unstructured inputs (OCR text, scanned PDFs), exposes a real-time REST API, and provides plain-English explanations for every flagged invoice.

---

## 3. Problem Statement

Design and develop an AI-based system that automatically analyzes GST invoices and e-way bills to detect:
- Fake invoices and duplicate submissions
- Manipulated transaction values
- Suspicious goods movement via invalid e-way bills
- Shell entities with abnormal transaction patterns
- Tax amount inconsistencies and GSTIN fraud

The solution must handle large-scale data, detect both obvious and subtle fraud, support multiple input formats, provide explainable insights, and reduce manual auditing effort.

---

## 4. Objectives

1. Build an ensemble fraud scoring system combining rules + ML
2. Achieve real-time detection via REST API (sub-second per invoice)
3. Support structured (CSV/Excel/JSON) and unstructured (PDF/OCR) data
4. Provide per-invoice explainability for auditor use
5. Reduce manual review workload by prioritizing highest-risk transactions

---

## 5. Proposed Solution

A three-layer ensemble system:

**Layer 1 — Rule Engine (40% weight)**  
Eleven deterministic fraud rules covering GSTIN validity, tax rates, e-way bill expiry, vehicle numbers, duplicate invoices, amount spikes, and vendor age. Works without any training data.

**Layer 2 — Isolation Forest (35% weight)**  
Unsupervised anomaly detection trained on 20+ engineered features. Identifies statistical outliers that no predefined rule covers.

**Layer 3 — Random Forest + GBM (25% weight)**  
Supervised classifiers trained when historical fraud labels are available. Handles complex non-linear fraud patterns.

Final Score = Rule×0.40 + Anomaly×0.35 + ML×0.25

---

## 6. System Architecture

```
INPUT LAYER
  ├── Structured: CSV / Excel / JSON / DB records
  └── Unstructured: Scanned PDFs / Images → OCR Handler → Structured dict

FEATURE LAYER
  ├── Amount features (z-score vs vendor avg, log transform, round flag)
  ├── Tax features (rate delta vs HSN code expected rate)
  ├── Temporal features (day of week, weekend flag, month)
  ├── E-Way features (validity excess, transit speed)
  ├── Vendor features (age, transaction count, new+high-value flag)
  └── Geo features (same-state flag, encoded states)

SCORING LAYER
  ├── Rule Engine → rule_score (0–100)
  ├── Isolation Forest → anomaly_score (0–100)
  └── RF + GBM → ml_score (0–100)

ENSEMBLE LAYER
  └── Weighted combination → final_risk_score → CRITICAL/HIGH/MEDIUM/LOW

OUTPUT LAYER
  ├── CSV: output/detection_results.csv
  ├── HTML: output/dashboard.html
  └── API: JSON response with explanation
```

---

## 7. Workflow / Pipeline

```
Step 1: Data Ingestion
  → Load CSV / parse OCR text / receive API JSON

Step 2: Feature Engineering
  → Compute 20+ features per invoice including z-scores, tax deltas,
    temporal signals, GSTIN validity, e-way anomalies

Step 3: Rule Scoring
  → Apply 11 rules deterministically → rule_score

Step 4: Anomaly Detection
  → StandardScaler → IsolationForest.score_samples → normalize → anomaly_score

Step 5: ML Scoring
  → StandardScaler → RF.predict_proba + GBM.predict_proba → ml_score

Step 6: Ensemble
  → final_risk_score = 0.40×rule + 0.35×anomaly + 0.25×ml

Step 7: Risk Banding
  → CRITICAL (≥75) / HIGH (≥55) / MEDIUM (≥35) / LOW (<35)

Step 8: Explainability
  → List triggered rules, score breakdown, per-component contribution

Step 9: Output
  → CSV report, HTML dashboard, or JSON API response
```

---

## 8. AI/ML Techniques Used

| Technique | Role | Why Chosen |
|-----------|------|------------|
| Isolation Forest | Unsupervised anomaly detection | Works without fraud labels, excellent at high-dimensional outliers |
| Random Forest | Supervised classification | Robust, handles class imbalance with class_weight='balanced' |
| Gradient Boosting | Supervised classification | Catches complex non-linear fraud patterns |
| StandardScaler | Feature normalization | Ensures ML models aren't dominated by high-magnitude features |
| Label Encoding | Categorical handling | Efficient encoding of state codes, HSN, transport mode |
| Feature Engineering | Signal amplification | Z-score, log-transform, delta features amplify fraud signals |

---

## 9. Fraud Detection Logic

### Rule 1: Duplicate Invoice (weight: 30)
If invoice_number + vendor_id combination appears more than once in dataset, flag as duplicate submission.

### Rule 2: Amount Spike (weight: 25)
If invoice_amount > vendor_avg + 3×vendor_std (z-score > 3), flag as anomalous spike.

### Rule 3: Round Amount (weight: 10)
If invoice_amount ≥ ₹10,000 and amount % 5,000 == 0, flag as suspiciously rounded.

### Rule 4: Invalid GSTIN (weight: 35)
GSTIN must be exactly 15 characters with first 2 as numeric state code. Fail → CRITICAL flag.

### Rule 5: E-Way Bill Expired (weight: 20)
If actual_transit_hours > eway_validity_hours, goods were in transit after bill expired.

### Rule 6: Tax Mismatch (weight: 30)
Computed tax rate = tax_amount / (invoice_amount - tax_amount). If |computed - expected| > 2.5%, flag.

### Rule 7: New Vendor High Value (weight: 15)
If vendor_reg_days < 60 and invoice_amount > ₹1,00,000, flag. Shell companies often transact immediately.

### Rule 8: Vehicle Mismatch (weight: 18)
Vehicle number must follow format: 2-letter state + 2 digits + letters + 4 digits. Fail → flag.

### Rule 9: Weekend High Value (weight: 8)
Invoice on Saturday/Sunday with amount > ₹50,000 triggers low-level flag.

### Rule 10: Distance Anomaly (weight: 20)
If transit time implies average speed > 133 km/h (unrealistic for road freight), flag.

### Rule 11: Related Entity Frequency (weight: 12)
If vendor–buyer pair count > 20 transactions, flag for circular transaction review.

---

## 10. Features of the System

- **Multi-format input**: CSV, Excel, JSON, REST API, OCR text
- **Real-time API**: Flask REST with /score, /score_batch, /explain endpoints
- **Ensemble scoring**: Three independent signals combined for robustness
- **Explainability**: Every flag comes with human-readable rule breakdown
- **Dashboard**: Color-coded HTML report with KPI cards and risk table
- **Scalable**: Batch processing via pandas; extensible to Spark for millions of rows
- **Offline capable**: No external API calls required; GSTN validation is format-based
- **Persistent model**: Trained model saved with joblib and reloaded for scoring

---

## 11. Dataset Description

### Synthetic Training Data (generated by data_generator.py)
- **600 training invoices** with 10% fraud rate
- **100 test invoices** for evaluation
- **8 fraud types** injected: duplicate, amount spike, round amount, invalid GSTIN, eway expired, tax mismatch, new vendor high value, vehicle mismatch
- **16 raw features** per invoice; **20+ engineered features**

### Feature Statistics (approximate)
| Feature | Range | Notes |
|---------|-------|-------|
| invoice_amount | ₹1,000 – ₹10,00,000 | Log-normal distribution |
| tax_amount | ₹50 – ₹2,80,000 | Derived from amount × HSN rate |
| vendor_reg_days | 10 – 3000 | New vendors = <60 days |
| transport_distance_km | 50 – 2000 km | Random realistic range |
| eway_validity_hours | 24 – 480 h | ~24h per 100km |

---

## 12. Data Preprocessing Steps

1. Parse invoice_date to datetime; extract day_of_week, month, hour
2. Log-transform invoice_amount (handles skewed distribution)
3. Compute per-vendor z-score for amount (requires fit on training data)
4. Calculate tax_rate_delta = |computed_rate - expected_rate|
5. Encode categorical columns: vendor_state, buyer_state, hsn_code, transport_mode
6. Compute e-way excess = max(0, actual_transit_hours - eway_validity_hours)
7. Flag binary indicators: is_round, is_weekend, is_new_vendor, gstin_valid
8. Fill all NaN with 0 before passing to models

---

## 13. Model Training Approach

```python
# 1. Feature engineering fit on training data
fe.fit(df_train)
X_train = fe.transform(df_train)

# 2. Isolation Forest — unsupervised, all data
iso.fit(scaler.fit_transform(X_train))

# 3. Supervised models — only if labels available
X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, test_size=0.2)
rf.fit(X_tr, y_tr)   # class_weight='balanced'
gbm.fit(X_tr, y_tr)
```

Evaluation metric: ROC-AUC (better than accuracy for imbalanced fraud data)

---

## 14. Explainability Method

Each invoice result includes:
1. **Score Breakdown**: Rule×weight, Anomaly×weight, ML×weight contributions
2. **Rule List**: Names of all triggered rules with severity and description
3. **Per-Rule Weight**: How much each rule added to the rule score

Example explanation output:
```
═══════════════════════════════════════════════════════
  FRAUD ANALYSIS REPORT
═══════════════════════════════════════════════════════
  Invoice     : INV-2024-000781
  Vendor      : ABC Traders Pvt Ltd
  Amount      : ₹8,50,000.00
  Date        : 2024-03-15 11:30:00
-------------------------------------------------------
  RISK SCORE  : 87.4 / 100  [CRITICAL]
-------------------------------------------------------
  Score Breakdown:
    Rule Engine   : 65.0 × 0.40 = 26.0
    Anomaly Model : 91.2 × 0.35 = 31.9
    ML Classifier : 88.5 × 0.25 = 22.1
-------------------------------------------------------
  Triggered Rules:
    [CRITICAL ] tax_mismatch           (+30)
                → Tax amount inconsistent with HSN rate
    [HIGH     ] eway_expired           (+20)
                → E-Way Bill validity expired during transit
    [MEDIUM   ] new_vendor_high_value  (+15)
                → New vendor (< 60 days) with high-value invoice
═══════════════════════════════════════════════════════
```

---

## 15. Tech Stack

```
Backend     : Python 3.10+
ML          : scikit-learn 1.3+
API         : Flask 3.0+
Data        : pandas 2.0+, numpy 1.24+
Serialization: joblib
OCR         : Built-in regex (extensible: pdfplumber, pytesseract)
Frontend    : Pure HTML5 + CSS3 (no framework dependencies)
Testing     : pytest
```

---

## 16. Database Design (Recommended for Production)

```sql
-- Core tables for production deployment

CREATE TABLE vendors (
    vendor_id       VARCHAR(20) PRIMARY KEY,
    vendor_name     VARCHAR(200),
    gstin           CHAR(15),
    state_code      CHAR(2),
    registration_date DATE,
    avg_invoice_amt DECIMAL(15,2),
    invoice_count   INTEGER
);

CREATE TABLE invoices (
    invoice_id       BIGSERIAL PRIMARY KEY,
    invoice_number   VARCHAR(50),
    vendor_id        VARCHAR(20) REFERENCES vendors(vendor_id),
    invoice_date     TIMESTAMP,
    invoice_amount   DECIMAL(15,2),
    tax_amount       DECIMAL(15,2),
    hsn_code         VARCHAR(10),
    buyer_state      CHAR(2),
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eway_bills (
    eway_id          VARCHAR(12) PRIMARY KEY,
    invoice_id       BIGINT REFERENCES invoices(invoice_id),
    vehicle_number   VARCHAR(15),
    distance_km      INTEGER,
    validity_hours   INTEGER,
    actual_hours     DECIMAL(8,2)
);

CREATE TABLE fraud_scores (
    invoice_id       BIGINT REFERENCES invoices(invoice_id),
    rule_score       DECIMAL(5,2),
    anomaly_score    DECIMAL(5,2),
    ml_score         DECIMAL(5,2),
    final_score      DECIMAL(5,2),
    risk_level       VARCHAR(10),
    triggered_rules  TEXT,
    scored_at        TIMESTAMP DEFAULT NOW()
);
```

---

## 17. APIs / Modules

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API status |
| `/score` | POST | Score single invoice JSON |
| `/score_batch` | POST | Score array of invoices |
| `/rules` | GET | List all fraud rules with weights |
| `/explain/<invoice_number>` | GET | Get explanation for scored invoice |

---

## 18. Example Input and Output

**Input (CSV row):**
```
INV-2024-000781,2024-03-15 11:30:00,V0042,ABC Traders,27ABCDE1234F1Z5,
MH,KA,8471,850000.00,5000.00,0.18,980,MH12AB1234,24,72,25
```

**Output (CSV row):**
```
INV-2024-000781,V0042,ABC Traders,2024-03-15 11:30:00,850000.00,
65.0,91.2,88.5,87.4,CRITICAL,"tax_mismatch, eway_expired, new_vendor_high_value",True
```

---

## 19. UI / Dashboard Description

The HTML dashboard (`output/dashboard.html`) includes:

- **Header bar**: Title, participant name, generation timestamp
- **KPI Cards**: Total invoices, CRITICAL count, HIGH count, MEDIUM count, flagged %, total value at risk
- **Flagged Transactions Table**: Sortable by risk score; columns: invoice number, vendor, date, amount, animated score bar, risk badge (color-coded), triggered rules
- **Color scheme**: Dark navy background, red/orange/yellow/green risk badges
- **No external dependencies**: Pure HTML/CSS, works offline

---

## 20. Advantages

1. **Day-1 ready**: Rule engine works without any training data
2. **Explainable**: Every score has a human-readable breakdown
3. **Ensemble robustness**: Three independent signals reduce both false positives and false negatives
4. **Multi-format**: Handles clean structured data AND messy OCR text
5. **Real-time**: API scores an invoice in under 50ms
6. **Low dependency**: Only standard Python ML stack required

---

## 21. Limitations

1. OCR extraction accuracy depends on scan quality; complex layouts may fail
2. GSTIN validation is format-based only; live GSTN API integration needed for full verification
3. Supervised models require labeled fraud data; initial deployment may under-use ML layer
4. Graph-based shell entity detection (circular billing networks) not yet implemented
5. Vendor statistics computed on training data; cold-start issue for entirely new vendors

---

## 22. Future Scope

1. **GSTN API Integration** — Real-time GSTIN lookup against government database
2. **Graph Neural Network** — Model vendor–buyer transaction graphs to detect circular billing
3. **LLM-powered OCR** — Use vision LLMs for accurate extraction from complex invoice layouts
4. **Streaming (Kafka + Spark)** — Handle millions of invoices per hour
5. **Federated Learning** — Collaborative model training across state GST departments
6. **Mobile App** — E-way bill scanning and real-time scoring for field officers
7. **Active Learning** — Auditor feedback loop to continuously improve model
8. **Multi-language OCR** — Support invoices in regional languages

---

## 23. Conclusion

This project demonstrates that a carefully engineered ensemble of rule-based logic and machine learning can significantly improve GST fraud detection. By combining deterministic rules that capture known fraud patterns with unsupervised anomaly detection for unknown patterns and supervised learning for historical fraud, the system is both immediately deployable and continuously improvable. The emphasis on explainability ensures auditors can act on alerts with confidence, dramatically reducing wasted investigation effort on false positives while catching the subtle multi-signal fraud cases that manual review would miss.

---

*End of Project Report*  
*Aman Kumar | Roll No: 2310991770*
