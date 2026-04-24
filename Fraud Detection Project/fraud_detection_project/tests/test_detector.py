"""
test_detector.py — Unit & integration tests
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest
from fraud_detector import RuleEngine, FeatureEngineer, InvoiceFraudDetector, RULES
from data_generator import generate_sample_data
from ocr_handler import extract_invoice_from_text, SAMPLE_OCR_TEXT


# ─── FIXTURES ────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def sample_data():
    os.makedirs('data/sample', exist_ok=True)
    return generate_sample_data(n_train=200, n_test=30, save_csv=False)


@pytest.fixture(scope='module')
def trained_detector(sample_data):
    df_train, _, labels, _ = sample_data
    det = InvoiceFraudDetector()
    det.fit(df_train, labels)
    return det


# ─── RULE ENGINE TESTS ───────────────────────────────────────────────────────

class TestRuleEngine:
    def test_invalid_gstin_flagged(self):
        engine = RuleEngine()
        engine.fit(pd.DataFrame([{
            'vendor_id': 'V001', 'invoice_number': 'INV-001',
            'invoice_amount': 5000, 'tax_amount': 900, 'expected_tax_rate': 0.18
        }]))
        score, rules = engine.score_row({
            'vendor_id': 'V001', 'invoice_number': 'INV-002',
            'vendor_gstin': 'BADGSTIN',
            'invoice_amount': 5000, 'tax_amount': 900,
            'expected_tax_rate': 0.18, 'vendor_reg_days': 500
        })
        assert 'invalid_gstin' in rules
        assert score > 0

    def test_duplicate_invoice_flagged(self):
        engine = RuleEngine()
        df = pd.DataFrame([
            {'vendor_id': 'V001', 'invoice_number': 'INV-001', 'invoice_amount': 5000, 'tax_amount': 900, 'expected_tax_rate': 0.18},
            {'vendor_id': 'V001', 'invoice_number': 'INV-001', 'invoice_amount': 5000, 'tax_amount': 900, 'expected_tax_rate': 0.18},
        ])
        engine.fit(df)
        score, rules = engine.score_row({
            'vendor_id': 'V001', 'invoice_number': 'INV-001',
            'vendor_gstin': '27ABCDE1234F1Z5',
            'invoice_amount': 5000, 'tax_amount': 900,
            'expected_tax_rate': 0.18, 'vendor_reg_days': 500,
            'eway_validity_hours': 48, 'actual_transit_hours': 10
        })
        assert 'duplicate_invoice' in rules

    def test_tax_mismatch_flagged(self):
        engine = RuleEngine()
        engine.fit(pd.DataFrame([{
            'vendor_id': 'V001', 'invoice_number': 'INV-X',
            'invoice_amount': 10000, 'tax_amount': 100, 'expected_tax_rate': 0.18
        }]))
        score, rules = engine.score_row({
            'vendor_id': 'V001', 'invoice_number': 'INV-Y',
            'vendor_gstin': '27ABCDE1234F1Z5',
            'invoice_amount': 10000, 'tax_amount': 100,
            'expected_tax_rate': 0.18, 'vendor_reg_days': 365
        })
        assert 'tax_mismatch' in rules

    def test_new_vendor_high_value(self):
        engine = RuleEngine()
        engine.fit(pd.DataFrame([{
            'vendor_id': 'V099', 'invoice_number': 'INV-A',
            'invoice_amount': 5000, 'tax_amount': 900, 'expected_tax_rate': 0.18
        }]))
        score, rules = engine.score_row({
            'vendor_id': 'V099', 'invoice_number': 'INV-B',
            'vendor_gstin': '27ABCDE1234F1Z5',
            'invoice_amount': 500000, 'tax_amount': 90000,
            'expected_tax_rate': 0.18, 'vendor_reg_days': 20
        })
        assert 'new_vendor_high_value' in rules

    def test_score_capped_at_100(self):
        engine = RuleEngine()
        engine.fit(pd.DataFrame([{
            'vendor_id': 'V001', 'invoice_number': 'INV-001',
            'invoice_amount': 5000, 'tax_amount': 900, 'expected_tax_rate': 0.18
        }] * 5))
        # Every bad signal at once
        score, _ = engine.score_row({
            'vendor_id': 'V001', 'invoice_number': 'INV-001',
            'vendor_gstin': 'INVALID',
            'invoice_amount': 1000000, 'tax_amount': 10,
            'expected_tax_rate': 0.18, 'vendor_reg_days': 5,
            'eway_validity_hours': 5, 'actual_transit_hours': 100,
            'vehicle_number': 'INVALID_9999',
            'invoice_date': '2024-01-07 10:00:00'
        })
        assert score <= 100


# ─── FEATURE ENGINEER ────────────────────────────────────────────────────────

class TestFeatureEngineer:
    def test_features_no_nan(self, sample_data):
        df_train, df_test, _, _ = sample_data
        fe = FeatureEngineer().fit(df_train)
        X = fe.transform(df_test)
        assert not X.isnull().values.any(), "Features contain NaN"

    def test_feature_count_positive(self, sample_data):
        df_train, df_test, _, _ = sample_data
        fe = FeatureEngineer().fit(df_train)
        X = fe.transform(df_test)
        assert X.shape[1] > 5


# ─── END-TO-END DETECTOR ─────────────────────────────────────────────────────

class TestDetector:
    def test_predict_returns_dataframe(self, trained_detector, sample_data):
        _, df_test, _, _ = sample_data
        results = trained_detector.predict(df_test)
        assert isinstance(results, pd.DataFrame)
        assert 'final_risk_score' in results.columns

    def test_scores_in_range(self, trained_detector, sample_data):
        _, df_test, _, _ = sample_data
        results = trained_detector.predict(df_test)
        assert results['final_risk_score'].between(0, 100).all()

    def test_risk_levels_valid(self, trained_detector, sample_data):
        _, df_test, _, _ = sample_data
        results = trained_detector.predict(df_test)
        valid = {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'}
        assert set(results['risk_level'].unique()).issubset(valid)

    def test_explain_returns_string(self, trained_detector, sample_data):
        _, df_test, _, _ = sample_data
        results = trained_detector.predict(df_test.head(1))
        row = results.iloc[0].to_dict()
        explanation = trained_detector.explain(row)
        assert isinstance(explanation, str)
        assert 'RISK SCORE' in explanation


# ─── OCR HANDLER ─────────────────────────────────────────────────────────────

class TestOCR:
    def test_extract_gstin(self):
        invoice = extract_invoice_from_text(SAMPLE_OCR_TEXT)
        assert invoice['vendor_gstin'] == '27ABCDE1234F1Z5'

    def test_extract_amount(self):
        invoice = extract_invoice_from_text(SAMPLE_OCR_TEXT)
        assert invoice['invoice_amount'] > 0

    def test_extract_invoice_number(self):
        invoice = extract_invoice_from_text(SAMPLE_OCR_TEXT)
        assert invoice['invoice_number'] != ''

    def test_output_compatible_with_detector(self, trained_detector):
        invoice = extract_invoice_from_text(SAMPLE_OCR_TEXT, 'V0042')
        df = pd.DataFrame([invoice])
        results = trained_detector.predict(df)
        assert len(results) == 1
        assert 0 <= results.iloc[0]['final_risk_score'] <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
