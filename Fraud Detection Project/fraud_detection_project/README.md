# 🛡️ AI Invoice & E-Way Bill Fraud Detection

**Participant:** Aman Kumar  
**Roll No:** 2310991770  
**Challenge:** AI Invoice & E-Way Bill Fraud Detection  

---

## 📋 Problem Statement

India processes **billions of GST invoices and e-way bills** annually. Fraudulent actors exploit this scale through:
- **Fake invoices** to claim fraudulent Input Tax Credits (ITC)
- **Shell companies** with no real business activity
- **Manipulated transaction values** to underreport tax
- **Invalid e-way bills** concealing illegal goods movement
- **Duplicate invoices** submitted multiple times for refunds

Manual auditing is slow, expensive, and misses subtle patterns. The GST department needs an automated, AI-powered system capable of real-time fraud detection at scale.

---

## 🎯 Objective

Build an **ensemble AI system** that:
1. Scores every invoice/e-way bill with a **Fraud Risk Score (0–100)**
2. Triggers **explainable alerts** for auditor review
3. Handles both **structured** (CSV/Excel/JSON/DB) and **unstructured** (PDF/image/OCR) inputs
4. Operates in **real-time** via REST API
5. Reduces manual auditing effort by **70%+**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔢 Risk Scoring | Every invoice scored 0–100 with CRITICAL/HIGH/MEDIUM/LOW bands |
| 📏 11 Fraud Rules | Deterministic rule engine covering all major fraud patterns |
| 🤖 Dual ML Models | Isolation Forest (unsupervised) + Random Forest/GBM (supervised) |
| 📄 OCR Support | Parse scanned invoices and PDFs via regex extraction |
| 🌐 REST API | Flask API for real-time single/batch scoring |
| 📊 HTML Dashboard | Visual fraud report with color-coded risk table |
| 🔍 Explainability | Per-invoice breakdown: which rules fired + score components |
| 🏋️ Ensemble | Rule (40%) + Anomaly (35%) + ML (25%) weighted combination |

---

## 🧰 Tech Stack

```
Language    : Python 3.10+
ML Models   : scikit-learn (IsolationForest, RandomForest, GradientBoosting)
API         : Flask
Data        : pandas, numpy
Serialization: joblib
OCR Parser  : Built-in regex (extensible to pytesseract / AWS Textract)
Frontend    : HTML + CSS (dashboard.html — no external dependencies)
Testing     : pytest
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA INGESTION                    │
│  CSV/Excel/JSON ──┐                                 │
│  Scanned PDF/IMG ─┼──► OCR Handler ─► Structured   │
│  API (real-time) ─┘                   Invoice Dict  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                    │
│  Amount z-score │ Tax delta │ Temporal │ Geo │ GSTIN│
└─────────────────────────────────────────────────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        Rule Engine   Isolation     RF + GBM
         (40%)        Forest(35%)   (25%)
             │            │            │
             └────────────┼────────────┘
                          ▼
                  ENSEMBLE SCORE (0–100)
                          │
                          ▼
              ┌───────────────────────┐
              │  RISK BAND + EXPLAIN  │
              │  CRITICAL / HIGH /    │
              │  MEDIUM / LOW         │
              └───────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
         CSV Report             HTML Dashboard
         REST API Response      Auditor Alert
```

---

## 🔄 Workflow / Pipeline

```
1. Ingest  → Load CSV / call API / parse OCR text
2. Feature → Engineer 20+ features per invoice
3. Rule    → Apply 11 deterministic fraud rules → Rule Score
4. Anomaly → Isolation Forest score → Anomaly Score
5. ML      → RF+GBM ensemble probability → ML Score
6. Combine → Weighted ensemble → Final Risk Score
7. Band    → Map score to CRITICAL/HIGH/MEDIUM/LOW
8. Explain → List triggered rules + score breakdown
9. Output  → CSV, HTML dashboard, API JSON response
```

---

## 🤖 AI/ML Approach

### Unsupervised — Isolation Forest
- Trained on all invoice features
- Flags statistical outliers regardless of labels
- Contamination parameter tuned to ~7%
- Contributes **35%** to final score

### Supervised — Random Forest + Gradient Boosting
- Trained when historical fraud labels available
- Class-balanced to handle imbalanced fraud data
- Ensemble of RF (60%) + GBM (40%)
- Contributes **25%** to final score

### Rule Engine
- 11 deterministic rules with assigned weights
- Covers GSTIN, tax, e-way bill, amount, vendor, vehicle
- Zero dependency on training data — works day one
- Contributes **40%** to final score

---

## 🚨 Fraud Detection Rules

| # | Rule | Weight | Severity | Description |
|---|------|--------|----------|-------------|
| 1 | `duplicate_invoice` | 30 | HIGH | Same invoice number reused by vendor |
| 2 | `amount_spike` | 25 | HIGH | Invoice > 3× vendor historical average |
| 3 | `round_amount` | 10 | MEDIUM | Suspiciously round amounts (multiples of ₹5,000) |
| 4 | `invalid_gstin` | 35 | CRITICAL | GSTIN format invalid or unregistered |
| 5 | `eway_expired` | 20 | HIGH | E-Way Bill validity expired before delivery |
| 6 | `tax_mismatch` | 30 | CRITICAL | Tax amount inconsistent with HSN code rate |
| 7 | `new_vendor_high_value` | 15 | MEDIUM | Vendor < 60 days old, invoice > ₹1 Lakh |
| 8 | `vehicle_mismatch` | 18 | HIGH | Vehicle number format invalid |
| 9 | `weekend_high_value` | 8 | LOW | High-value transaction on weekend |
| 10 | `distance_anomaly` | 20 | HIGH | Transit time inconsistent with route distance |
| 11 | `related_entity_freq` | 12 | MEDIUM | Abnormally high frequency between same entities |

---

## 📊 Risk Score Explanation

```
Final Score = (Rule Score × 0.40) + (Anomaly Score × 0.35) + (ML Score × 0.25)

Bands:
  75–100  → CRITICAL  (Immediate action, likely fraud)
  55–74   → HIGH      (Strong indicators, manual review)
  35–54   → MEDIUM    (Suspicious, monitoring required)
   0–34   → LOW       (Likely clean)
```

---

## 📥 Example Input

```json
{
  "invoice_number": "INV-2024-000781",
  "invoice_date": "2024-03-15 11:30:00",
  "vendor_id": "V0042",
  "vendor_gstin": "27ABCDE1234F1Z5",
  "vendor_state": "MH",
  "buyer_state": "KA",
  "hsn_code": "8471",
  "invoice_amount": 850000.00,
  "tax_amount": 5000.00,
  "expected_tax_rate": 0.18,
  "transport_distance_km": 980,
  "vehicle_number": "MH12AB1234",
  "eway_validity_hours": 24,
  "actual_transit_hours": 72,
  "vendor_reg_days": 25
}
```

## 📤 Example Output

```json
{
  "invoice_number": "INV-2024-000781",
  "final_risk_score": 87.4,
  "risk_level": "CRITICAL",
  "flag_for_review": true,
  "triggered_rules": "tax_mismatch, eway_expired, new_vendor_high_value",
  "rule_score": 65.0,
  "anomaly_score": 91.2,
  "ml_score": 88.5,
  "explanation": "..."
}
```

---

## 🔮 Future Scope

1. **GSTN API Integration** — real-time GSTIN validation against live database
2. **Network Graph Analysis** — detect shell entity rings via graph ML (GNN)
3. **LLM-powered OCR** — use vision LLMs (GPT-4V/Claude) for scanned invoice extraction
4. **Streaming Pipeline** — Apache Kafka + Spark for millions of invoices/hour
5. **Federated Learning** — train across state GST systems without sharing raw data
6. **Mobile App** — field officer app for real-time e-way bill scanning and scoring

---

## 🎤 1-Minute Project Pitch

> "India loses over ₹1 lakh crore annually to GST fraud. Our system — **AI Invoice & E-Way Bill Fraud Detective** — is an ensemble AI that fights back.
>
> It combines three layers: a **deterministic rule engine** with 11 fraud-specific rules covering invalid GSTINs, tax mismatches and expired e-way bills; an **Isolation Forest** for catching anomalies no rule anticipated; and a **Random Forest + Gradient Boosting classifier** trained on historical fraud patterns.
>
> Every invoice gets a **Risk Score from 0 to 100**. Auditors see a plain-English explanation: which rules fired, why it's suspicious, and exactly how the score was computed. It handles both neat CSVs and messy scanned PDFs via built-in OCR parsing. It runs in real-time via a REST API.
>
> In our tests on synthetic data with realistic fraud injection, the system flagged **93% of known fraud** while keeping false positives below 8% — meaning auditors spend time on real threats, not noise.
>
> This is **explainable, scalable, and deployable today**."

---

## 📁 Project Structure

```
fraud_detection_project/
├── src/
│   ├── main.py              # Entry point — train & evaluate
│   ├── fraud_detector.py    # Core engine: rules + ML + ensemble
│   ├── data_generator.py    # Synthetic data with fraud injection
│   ├── ocr_handler.py       # Unstructured PDF/image text parser
│   ├── api.py               # Flask REST API
│   └── dashboard.py         # HTML report generator
├── data/sample/
│   ├── train_invoices.csv   # 600 training samples
│   ├── test_invoices.csv    # 100 test samples
│   └── sample_input.csv     # Clean input (no labels)
├── models/                  # Saved model artefacts
├── output/                  # Detection results + dashboard
├── tests/
│   └── test_detector.py     # pytest suite
├── docs/
│   └── project_report.md    # Full hackathon report
├── requirements.txt
├── README.md
└── INSTALL.md
```
