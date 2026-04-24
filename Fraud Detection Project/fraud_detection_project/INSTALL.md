# 📦 INSTALL.md — Setup & Run Guide

**AI Invoice & E-Way Bill Fraud Detection**  
Participant: Aman Kumar | Roll No: 2310991770

---

## ✅ Prerequisites

- **Python 3.10 or higher** — `python --version`
- **pip** package manager — `pip --version`
- ~200 MB disk space for model artefacts
- Internet connection for initial package install

---

## 🔧 Installation Steps

### 1. Extract the ZIP

```bash
unzip fraud_detection_project.zip
cd fraud_detection_project
```

### 2. (Recommended) Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical ops |
| scikit-learn | ≥1.3.0 | ML models |
| joblib | ≥1.3.0 | Model serialization |
| flask | ≥3.0.0 | REST API |
| openpyxl | ≥3.1.0 | Excel support |
| pytest | ≥7.4.0 | Unit tests |

---

## ▶️ Commands to Run

### Option A — Full Pipeline (train + detect + report)

```bash
cd src
python main.py
```

This will:
1. Generate 700 synthetic invoices with injected fraud patterns
2. Train the ensemble model
3. Run detection on test set
4. Save `output/detection_results.csv`

### Option B — Score Your Own CSV

```bash
cd src
python main.py --mode score --input ../data/sample/sample_input.csv
```

### Option C — Train on Custom Data

```bash
cd src
python main.py --mode train --train path/to/train.csv --test path/to/test.csv
```

### Option D — Start REST API

```bash
cd src
python api.py
```

API runs at `http://localhost:5000`

### Option E — Generate HTML Dashboard

```bash
cd src
python dashboard.py
# Opens output/dashboard.html in browser
```

---

## 🧪 How to Test with Sample Data

### 1. Generate fresh sample data

```bash
cd src
python data_generator.py
```

Saves to `data/sample/`:
- `train_invoices.csv` — 600 rows with labels
- `test_invoices.csv` — 100 rows with labels
- `sample_input.csv` — 100 rows without labels

### 2. Run unit tests

```bash
cd fraud_detection_project
pytest tests/ -v
```

Expected: All tests pass ✅

### 3. Test API with curl

Start API first:
```bash
cd src && python api.py
```

Single invoice score:
```bash
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "invoice_number": "INV-TEST-001",
    "invoice_date": "2024-06-15 10:00:00",
    "vendor_id": "V0001",
    "vendor_name": "Test Vendor",
    "vendor_gstin": "BADGSTIN123",
    "vendor_state": "MH",
    "buyer_state": "KA",
    "hsn_code": "8471",
    "invoice_amount": 500000,
    "tax_amount": 500,
    "expected_tax_rate": 0.18,
    "transport_distance_km": 900,
    "vehicle_number": "INVALID",
    "eway_validity_hours": 10,
    "actual_transit_hours": 72,
    "vendor_reg_days": 15
  }'
```

Health check:
```bash
curl http://localhost:5000/health
```

List all rules:
```bash
curl http://localhost:5000/rules
```

---

## 🗂️ Input CSV Column Reference

| Column | Type | Example | Required |
|--------|------|---------|----------|
| invoice_number | str | INV-2024-000001 | Yes |
| invoice_date | datetime | 2024-03-15 10:30:00 | Yes |
| vendor_id | str | V0001 | Yes |
| vendor_name | str | ABC Pvt Ltd | No |
| vendor_gstin | str | 27ABCDE1234F1Z5 | Yes |
| vendor_state | str | MH | Yes |
| buyer_state | str | KA | Yes |
| hsn_code | str | 8471 | Yes |
| invoice_amount | float | 85000.00 | Yes |
| tax_amount | float | 15300.00 | Yes |
| expected_tax_rate | float | 0.18 | Yes |
| transport_distance_km | int | 980 | Yes |
| vehicle_number | str | MH12AB1234 | Yes |
| eway_validity_hours | int | 48 | Yes |
| actual_transit_hours | float | 36.5 | Yes |
| vendor_reg_days | int | 365 | Yes |
| is_fraud (optional) | int | 0 or 1 | No (for training) |

---

## 🔍 OCR / Unstructured Data

```bash
cd src
python ocr_handler.py
```

This demonstrates extraction from a sample scanned invoice text. For real PDFs, install `pdfplumber`:

```bash
pip install pdfplumber pytesseract Pillow
```

Then adapt `ocr_handler.py` to call `pdfplumber.open(path)` and pipe the text to `extract_invoice_from_text()`.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: pandas` | Run `pip install -r requirements.txt` |
| `No module named sklearn` | Run `pip install scikit-learn` |
| `FileNotFoundError: models/detector.pkl` | Run `python main.py` once to train |
| API returns `Model not loaded` | Start `api.py` after running `main.py` first |
| Port 5000 already in use | Change port in `api.py`: `app.run(port=5001)` |
| Tests fail with path error | Run pytest from `fraud_detection_project/` root |
| Windows venv activation error | Use `venv\Scripts\activate.bat` |
| Pandas `FutureWarning` spam | Warnings suppressed in code; safe to ignore |

---

## 📂 Output Files

After running `main.py`:

```
output/
├── detection_results.csv   # All invoices with risk scores
└── dashboard.html          # Open in browser for visual report
```

---

## 📞 Support

**Aman Kumar** | Roll No: 2310991770  
Challenge: AI Invoice & E-Way Bill Fraud Detection
