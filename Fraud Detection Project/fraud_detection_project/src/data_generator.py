"""
data_generator.py
Generates synthetic GST Invoice & E-Way Bill data with injected fraud patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

random.seed(42)
np.random.seed(42)

STATES = ['MH', 'DL', 'KA', 'TN', 'GJ', 'UP', 'RJ', 'WB', 'MP', 'AP']
TRANSPORT_MODES = ['road', 'rail', 'air', 'ship']
HSN_CODES = ['8471', '8517', '9403', '7308', '3004', '8443', '6203', '8501', '2710', '8704']
HSN_TAX_RATES = {
    '8471': 0.18, '8517': 0.18, '9403': 0.18, '7308': 0.18,
    '3004': 0.12, '8443': 0.18, '6203': 0.05, '8501': 0.18,
    '2710': 0.18, '8704': 0.28
}


def random_gstin(state_idx=None):
    code = str((state_idx or random.randint(1, 36))).zfill(2)
    pan = ''.join(random.choices(string.ascii_uppercase, k=5))
    pan += ''.join(random.choices(string.digits, k=4))
    pan += random.choice(string.ascii_uppercase)
    return f"{code}{pan}1Z{random.choice(string.digits + string.ascii_uppercase)}"


def generate_vendor_pool(n=50):
    vendors = []
    for i in range(n):
        s_idx = random.randint(1, len(STATES))
        vendors.append({
            'vendor_id': f'V{str(i+1).zfill(4)}',
            'vendor_name': f'Vendor_{i+1} Pvt Ltd',
            'vendor_gstin': random_gstin(s_idx + 26),
            'vendor_state': STATES[s_idx - 1],
            'avg_invoice_amount': float(np.random.lognormal(10, 1.2)),
            'reg_days_ago': random.randint(30, 3000)
        })
    return vendors


def make_invoice(vendor, invoice_date, counter, fraud_type=None):
    hsn = random.choice(HSN_CODES)
    tax_rate = HSN_TAX_RATES[hsn]
    buyer_state = random.choice(STATES)
    base_amt = max(vendor['avg_invoice_amount'] * float(np.random.lognormal(0, 0.4)), 1000.0)
    dist_km = random.randint(50, 2000)
    transit_h = dist_km / random.uniform(35, 60) + random.uniform(1, 4)
    validity_h = max(24, (dist_km // 100) * 24)
    vehicle_no = f"MH{''.join(random.choices(string.digits, k=2))}{''.join(random.choices(string.ascii_uppercase, k=2))}{''.join(random.choices(string.digits, k=4))}"

    rec = {
        'invoice_number': f'INV-2024-{str(counter).zfill(6)}',
        'invoice_date': invoice_date.strftime('%Y-%m-%d %H:%M:%S'),
        'vendor_id': vendor['vendor_id'],
        'vendor_name': vendor['vendor_name'],
        'vendor_gstin': vendor['vendor_gstin'],
        'vendor_state': vendor['vendor_state'],
        'buyer_state': buyer_state,
        'hsn_code': hsn,
        'invoice_amount': round(base_amt, 2),
        'tax_amount': round(base_amt * tax_rate / (1 + tax_rate), 2),
        'expected_tax_rate': tax_rate,
        'transport_mode': random.choice(TRANSPORT_MODES),
        'transport_distance_km': dist_km,
        'vehicle_number': vehicle_no,
        'eway_validity_hours': validity_h,
        'actual_transit_hours': round(transit_h, 2),
        'vendor_reg_days': vendor['reg_days_ago'],
        'is_fraud': 0,
        'fraud_type': 'none'
    }

    if fraud_type == 'duplicate':
        rec['invoice_number'] = f'INV-2024-{str(max(1, counter - random.randint(1,5))).zfill(6)}'
        rec['is_fraud'] = 1; rec['fraud_type'] = 'duplicate_invoice'
    elif fraud_type == 'amount_spike':
        rec['invoice_amount'] = round(vendor['avg_invoice_amount'] * random.uniform(6, 12), 2)
        rec['tax_amount'] = round(rec['invoice_amount'] * tax_rate / (1 + tax_rate), 2)
        rec['is_fraud'] = 1; rec['fraud_type'] = 'amount_spike'
    elif fraud_type == 'round_amount':
        rec['invoice_amount'] = float(random.choice([50000, 100000, 200000, 500000, 1000000]))
        rec['tax_amount'] = round(rec['invoice_amount'] * tax_rate / (1 + tax_rate), 2)
        rec['is_fraud'] = 1; rec['fraud_type'] = 'round_amount'
    elif fraud_type == 'gstin_fraud':
        rec['vendor_gstin'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
        rec['is_fraud'] = 1; rec['fraud_type'] = 'invalid_gstin'
    elif fraud_type == 'eway_expired':
        rec['eway_validity_hours'] = random.randint(8, 18)
        rec['actual_transit_hours'] = random.randint(30, 80)
        rec['is_fraud'] = 1; rec['fraud_type'] = 'eway_expired'
    elif fraud_type == 'tax_mismatch':
        rec['tax_amount'] = round(rec['invoice_amount'] * 0.01, 2)
        rec['is_fraud'] = 1; rec['fraud_type'] = 'tax_mismatch'
    elif fraud_type == 'new_vendor':
        rec['invoice_amount'] = round(random.uniform(250000, 800000), 2)
        rec['vendor_reg_days'] = random.randint(10, 45)
        rec['is_fraud'] = 1; rec['fraud_type'] = 'new_vendor_high_value'
    elif fraud_type == 'vehicle_mismatch':
        rec['vehicle_number'] = 'INVALID_' + ''.join(random.choices(string.digits, k=4))
        rec['is_fraud'] = 1; rec['fraud_type'] = 'vehicle_mismatch'

    return rec


def generate_sample_data(n_train=600, n_test=100, fraud_rate=0.10, save_csv=True):
    vendors = generate_vendor_pool(50)
    fraud_types = [
        'duplicate', 'amount_spike', 'round_amount', 'gstin_fraud',
        'eway_expired', 'tax_mismatch', 'new_vendor', 'vehicle_mismatch'
    ]
    records = []
    base_date = datetime(2024, 1, 1)
    for i in range(n_train + n_test):
        vendor = random.choice(vendors)
        inv_date = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(8, 18))
        ft = random.choice(fraud_types) if random.random() < fraud_rate else None
        records.append(make_invoice(vendor, inv_date, i + 1, ft))

    df = pd.DataFrame(records)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test = df.iloc[n_train:].reset_index(drop=True)

    if save_csv:
        df_train.to_csv('data/sample/train_invoices.csv', index=False)
        df_test.to_csv('data/sample/test_invoices.csv', index=False)
        df_test.drop(columns=['is_fraud', 'fraud_type']).to_csv('data/sample/sample_input.csv', index=False)
        print(f"[DataGen] Train: {len(df_train)} | Test: {len(df_test)} rows saved.")

    labels_train = df_train['is_fraud'].values
    labels_test = df_test['is_fraud'].values
    return df_train, df_test, labels_train, labels_test


if __name__ == '__main__':
    generate_sample_data()
