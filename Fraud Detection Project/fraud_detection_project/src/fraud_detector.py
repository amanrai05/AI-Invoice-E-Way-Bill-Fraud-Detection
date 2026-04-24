"""
fraud_detector.py
Ensemble fraud detection: Rule Engine + Isolation Forest + Random Forest
Author : Aman Kumar | Roll No: 2310991770
Challenge: AI Invoice & E-Way Bill Fraud Detection
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# ─── FRAUD RULES REGISTRY ────────────────────────────────────────────────────

RULES = {
    'duplicate_invoice':       {'weight': 30, 'severity': 'HIGH',     'desc': 'Same invoice number reused by vendor'},
    'amount_spike':            {'weight': 25, 'severity': 'HIGH',     'desc': 'Invoice amount > 3× vendor historical average'},
    'round_amount':            {'weight': 10, 'severity': 'MEDIUM',   'desc': 'Suspiciously round invoice amount'},
    'invalid_gstin':           {'weight': 35, 'severity': 'CRITICAL', 'desc': 'GSTIN format invalid or unregistered'},
    'eway_expired':            {'weight': 20, 'severity': 'HIGH',     'desc': 'E-Way Bill validity expired during transit'},
    'tax_mismatch':            {'weight': 30, 'severity': 'CRITICAL', 'desc': 'Tax amount inconsistent with HSN rate'},
    'new_vendor_high_value':   {'weight': 15, 'severity': 'MEDIUM',   'desc': 'New vendor (< 60 days) with high-value invoice'},
    'vehicle_mismatch':        {'weight': 18, 'severity': 'HIGH',     'desc': 'Vehicle number format invalid or suspicious'},
    'weekend_high_value':      {'weight':  8, 'severity': 'LOW',      'desc': 'High-value transaction on weekend/holiday'},
    'distance_anomaly':        {'weight': 20, 'severity': 'HIGH',     'desc': 'Transit hours inconsistent with route distance'},
    'related_entity_freq':     {'weight': 12, 'severity': 'MEDIUM',   'desc': 'Abnormally high transaction frequency between entities'},
}

RISK_BANDS = [
    (75, 'CRITICAL'),
    (55, 'HIGH'),
    (35, 'MEDIUM'),
    ( 0, 'LOW'),
]


# ─── RULE ENGINE ─────────────────────────────────────────────────────────────

class RuleEngine:
    def __init__(self):
        self.vendor_stats = {}
        self.invoice_seen = {}
        self.pair_counts = {}

    def fit(self, df: pd.DataFrame):
        self.vendor_stats = df.groupby('vendor_id').agg(
            avg_amt=('invoice_amount', 'mean'),
            std_amt=('invoice_amount', 'std'),
            count=('invoice_amount', 'count')
        ).to_dict('index')

        self.invoice_seen = df.groupby(
            df['vendor_id'] + '||' + df['invoice_number']
        ).size().to_dict()

        if 'buyer_state' in df.columns:
            self.pair_counts = df.groupby(
                df['vendor_id'] + '||' + df['buyer_state']
            ).size().to_dict()
        return self

    def score_row(self, row: dict) -> tuple:
        score = 0
        triggered = []

        vid = row.get('vendor_id', '')
        amt = float(row.get('invoice_amount', 0))
        stats = self.vendor_stats.get(vid, {})

        # 1. Duplicate invoice
        key = f"{vid}||{row.get('invoice_number','')}"
        if self.invoice_seen.get(key, 0) > 1:
            score += RULES['duplicate_invoice']['weight']
            triggered.append('duplicate_invoice')

        # 2. Amount spike vs vendor average
        avg = stats.get('avg_amt', amt)
        std = max(stats.get('std_amt', 1), 1)
        if avg > 0 and amt > avg + 3 * std:
            score += RULES['amount_spike']['weight']
            triggered.append('amount_spike')

        # 3. Suspiciously round amount
        if amt >= 10000 and amt % 5000 == 0:
            score += RULES['round_amount']['weight']
            triggered.append('round_amount')

        # 4. Invalid GSTIN (15-char alphanumeric check)
        gstin = str(row.get('vendor_gstin', ''))
        if len(gstin) != 15 or not gstin[:2].isdigit():
            score += RULES['invalid_gstin']['weight']
            triggered.append('invalid_gstin')

        # 5. E-Way Bill expired
        validity = float(row.get('eway_validity_hours', 9999))
        actual   = float(row.get('actual_transit_hours', 0))
        if actual > validity:
            score += RULES['eway_expired']['weight']
            triggered.append('eway_expired')

        # 6. Tax mismatch
        inv_amt = amt
        tax_amt = float(row.get('tax_amount', 0))
        exp_rate = float(row.get('expected_tax_rate', 0.18))
        if inv_amt > 0:
            calc_rate = tax_amt / max(inv_amt - tax_amt, 1)
            if abs(calc_rate - exp_rate) > 0.025:
                score += RULES['tax_mismatch']['weight']
                triggered.append('tax_mismatch')

        # 7. New vendor high-value
        reg_days = int(row.get('vendor_reg_days', 9999))
        if reg_days < 60 and amt > 100000:
            score += RULES['new_vendor_high_value']['weight']
            triggered.append('new_vendor_high_value')

        # 8. Vehicle number format
        veh = str(row.get('vehicle_number', ''))
        if 'INVALID' in veh.upper() or len(veh) < 8:
            score += RULES['vehicle_mismatch']['weight']
            triggered.append('vehicle_mismatch')

        # 9. Weekend high-value
        try:
            d = pd.to_datetime(row.get('invoice_date'))
            if d.weekday() >= 5 and amt > 50000:
                score += RULES['weekend_high_value']['weight']
                triggered.append('weekend_high_value')
        except Exception:
            pass

        # 10. Distance anomaly
        dist = float(row.get('transport_distance_km', 0))
        if dist > 0 and actual > 0:
            expected_h = dist / 40  # ~40 km/h avg
            if actual < expected_h * 0.3:  # too fast — suspicious
                score += RULES['distance_anomaly']['weight']
                triggered.append('distance_anomaly')

        # 11. Related entity frequency
        pair_key = f"{vid}||{row.get('buyer_state','')}"
        if self.pair_counts.get(pair_key, 0) > 20:
            score += RULES['related_entity_freq']['weight']
            triggered.append('related_entity_freq')

        return min(score, 100), triggered


# ─── FEATURE ENGINEER ────────────────────────────────────────────────────────

class FeatureEngineer:
    def __init__(self):
        self.vendor_stats = {}
        self.encoders = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        self.vendor_stats = df.groupby('vendor_id').agg(
            avg_amt=('invoice_amount', 'mean'),
            std_amt=('invoice_amount', 'std'),
            cnt=('invoice_amount', 'count')
        ).to_dict('index')

        for col in ['vendor_state', 'buyer_state', 'hsn_code', 'transport_mode']:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str).fillna('UNK'))
                self.encoders[col] = le
        self.fitted = True
        return self

    def _safe_encode(self, col, val):
        le = self.encoders.get(col)
        if le is None:
            return 0
        val = str(val)
        if val not in le.classes_:
            return -1
        return int(le.transform([val])[0])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        feats['log_amount']      = np.log1p(df['invoice_amount'].fillna(0))
        feats['is_round']        = ((df['invoice_amount'] % 5000 == 0) & (df['invoice_amount'] >= 10000)).astype(int)

        def vendor_zscore(row):
            s = self.vendor_stats.get(row['vendor_id'], {})
            avg = s.get('avg_amt', row['invoice_amount'])
            std = max(s.get('std_amt', 1), 1)
            return (row['invoice_amount'] - avg) / std
        feats['amount_zscore'] = df.apply(vendor_zscore, axis=1)

        feats['vendor_cnt'] = df['vendor_id'].map(lambda v: self.vendor_stats.get(v, {}).get('cnt', 1))

        feats['tax_rate_calc']  = df['tax_amount'] / (df['invoice_amount'] - df['tax_amount'] + 1e-6)
        feats['tax_rate_delta'] = abs(feats['tax_rate_calc'] - df['expected_tax_rate'].fillna(0.18))

        df2 = df.copy()
        df2['invoice_date'] = pd.to_datetime(df2['invoice_date'], errors='coerce')
        feats['day_of_week'] = df2['invoice_date'].dt.dayofweek.fillna(0)
        feats['is_weekend']  = (feats['day_of_week'] >= 5).astype(int)
        feats['month']       = df2['invoice_date'].dt.month.fillna(1)

        if 'eway_validity_hours' in df.columns:
            feats['eway_excess'] = (df['actual_transit_hours'] - df['eway_validity_hours']).clip(0)
            feats['eway_ratio']  = df['actual_transit_hours'] / (df['eway_validity_hours'] + 1)

        if 'transport_distance_km' in df.columns and 'actual_transit_hours' in df.columns:
            feats['speed_kmh'] = df['transport_distance_km'] / (df['actual_transit_hours'] + 1)

        if 'vendor_reg_days' in df.columns:
            feats['is_new_vendor']        = (df['vendor_reg_days'] < 60).astype(int)
            feats['new_vendor_high_value'] = ((df['vendor_reg_days'] < 60) & (df['invoice_amount'] > 100000)).astype(int)

        for col in ['vendor_state', 'buyer_state', 'hsn_code', 'transport_mode']:
            if col in df.columns:
                feats[col + '_enc'] = df[col].apply(lambda v: self._safe_encode(col, v))

        gstin_valid = df['vendor_gstin'].apply(
            lambda g: 1 if isinstance(g, str) and len(g) == 15 and g[:2].isdigit() else 0
        ) if 'vendor_gstin' in df.columns else pd.Series(1, index=df.index)
        feats['gstin_valid'] = gstin_valid

        return feats.fillna(0)


# ─── ML MODELS ───────────────────────────────────────────────────────────────

class MLEngine:
    def __init__(self):
        self.iso   = IsolationForest(n_estimators=200, contamination=0.07, random_state=42)
        self.rf    = RandomForestClassifier(n_estimators=200, max_depth=12,
                                            class_weight='balanced', random_state=42, n_jobs=-1)
        self.gbm   = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.has_labels = False
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names = list(X.columns)
        Xs = self.scaler.fit_transform(X)
        self.iso.fit(Xs)

        if y is not None and len(np.unique(y)) > 1:
            X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2,
                                                        stratify=y, random_state=42)
            self.rf.fit(X_tr, y_tr)
            self.gbm.fit(X_tr, y_tr)
            print("── RF  ──")
            print(classification_report(y_te, self.rf.predict(X_te)))
            print(f"ROC-AUC: {roc_auc_score(y_te, self.rf.predict_proba(X_te)[:,1]):.4f}")
            self.has_labels = True
        return self

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X)
        raw = self.iso.score_samples(Xs)
        mn, mx = raw.min(), raw.max()
        return ((1 - (raw - mn) / (mx - mn + 1e-9)) * 100).clip(0, 100)

    def fraud_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.has_labels:
            return np.zeros(len(X))
        Xs = self.scaler.transform(X)
        p_rf  = self.rf.predict_proba(Xs)[:, 1]
        p_gbm = self.gbm.predict_proba(Xs)[:, 1]
        return (p_rf * 0.6 + p_gbm * 0.4) * 100

    def top_features(self, X: pd.DataFrame, top_n=5) -> list:
        if not self.has_labels or not self.feature_names:
            return []
        imp = self.rf.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        return [(self.feature_names[i], round(float(imp[i]), 4)) for i in idx]

    def save(self, path='models/ml_engine.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path='models/ml_engine.pkl'):
        return joblib.load(path)


# ─── MAIN DETECTOR ───────────────────────────────────────────────────────────

class InvoiceFraudDetector:
    """
    Ensemble detector combining:
      Rule Engine   (40%)  – deterministic fraud rules
      Anomaly Score (35%)  – Isolation Forest unsupervised
      ML Score      (25%)  – RF + GBM supervised (when labels available)
    """
    W = {'rule': 0.40, 'anomaly': 0.35, 'ml': 0.25}

    def __init__(self):
        self.fe      = FeatureEngineer()
        self.rules   = RuleEngine()
        self.ml      = MLEngine()

    def fit(self, df: pd.DataFrame, labels=None):
        print("[1/3] Fitting feature engineer...")
        self.fe.fit(df)
        print("[2/3] Fitting rule engine...")
        self.rules.fit(df)
        print("[3/3] Training ML engine...")
        X = self.fe.transform(df)
        self.ml.fit(X, labels)
        print("✓ Detector ready.\n")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.fe.transform(df)
        a_scores = self.ml.anomaly_scores(X)
        m_scores = self.ml.fraud_proba(X)

        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            r_score, triggered = self.rules.score_row(row.to_dict())
            final = (r_score * self.W['rule'] +
                     a_scores[i] * self.W['anomaly'] +
                     m_scores[i] * self.W['ml'])
            final = round(min(final, 100), 2)

            risk = 'LOW'
            for threshold, band in RISK_BANDS:
                if final >= threshold:
                    risk = band
                    break

            rows.append({
                'invoice_number':   row.get('invoice_number', ''),
                'vendor_id':        row.get('vendor_id', ''),
                'vendor_name':      row.get('vendor_name', ''),
                'invoice_date':     row.get('invoice_date', ''),
                'invoice_amount':   row.get('invoice_amount', 0),
                'rule_score':       round(r_score, 2),
                'anomaly_score':    round(a_scores[i], 2),
                'ml_score':         round(m_scores[i], 2),
                'final_risk_score': final,
                'risk_level':       risk,
                'triggered_rules':  ', '.join(triggered) if triggered else 'None',
                'flag_for_review':  final >= 35
            })

        return pd.DataFrame(rows)

    def explain(self, result_row: dict) -> str:
        rules = [r.strip() for r in result_row.get('triggered_rules', '').split(',') if r.strip() != 'None']
        lines = [
            "=" * 55,
            f"  FRAUD ANALYSIS REPORT",
            "=" * 55,
            f"  Invoice     : {result_row['invoice_number']}",
            f"  Vendor      : {result_row.get('vendor_name', result_row['vendor_id'])}",
            f"  Amount      : ₹{float(result_row['invoice_amount']):,.2f}",
            f"  Date        : {result_row['invoice_date']}",
            "-" * 55,
            f"  RISK SCORE  : {result_row['final_risk_score']:.1f} / 100  [{result_row['risk_level']}]",
            "-" * 55,
            "  Score Breakdown:",
            f"    Rule Engine   : {result_row['rule_score']:.1f} × 0.40 = {result_row['rule_score']*0.40:.1f}",
            f"    Anomaly Model : {result_row['anomaly_score']:.1f} × 0.35 = {result_row['anomaly_score']*0.35:.1f}",
            f"    ML Classifier : {result_row['ml_score']:.1f} × 0.25 = {result_row['ml_score']*0.25:.1f}",
            "-" * 55,
            "  Triggered Rules:",
        ]
        if rules:
            for r in rules:
                info = RULES.get(r, {})
                sev  = info.get('severity', '')
                desc = info.get('desc', r)
                wt   = info.get('weight', 0)
                lines.append(f"    [{sev:8s}] {r}  (+{wt})")
                lines.append(f"              → {desc}")
        else:
            lines.append("    None — flagged by anomaly/ML models")
        lines.append("=" * 55)
        return '\n'.join(lines)

    def save(self, path='models/detector.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Detector saved → {path}")

    @staticmethod
    def load(path='models/detector.pkl'):
        return joblib.load(path)
