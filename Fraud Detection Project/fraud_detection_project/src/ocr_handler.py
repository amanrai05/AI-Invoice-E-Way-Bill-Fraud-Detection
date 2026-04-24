"""
ocr_handler.py — Unstructured data ingestion (PDFs, images, scanned invoices)
Author : Aman Kumar | Roll No: 2310991770
"""

import re
import json
from datetime import datetime


# ─── REGEX PATTERNS ──────────────────────────────────────────────────────────

PATTERNS = {
    'invoice_number': [
        r'(?:Invoice\s*(?:No|Number|#)\s*[:\-]?\s*)([A-Z0-9\-/]+)',
        r'(?:INV[-/]?)(\d{4,})',
    ],
    'invoice_date': [
        r'(\d{2}[/-]\d{2}[/-]\d{4})',
        r'(\d{4}[/-]\d{2}[/-]\d{2})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})',
    ],
    'gstin': [
        r'\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1})\b',
    ],
    'amount': [
        r'(?:Total\s*(?:Amount|Value|Invoice)\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*)([\d,]+(?:\.\d{1,2})?)',
        r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)',
    ],
    'tax_amount': [
        r'(?:GST|IGST|CGST|SGST|Tax)\s*(?:Amount)?\s*[:\-]?\s*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
    ],
    'vehicle_number': [
        r'\b([A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{4})\b',
    ],
    'eway_bill_number': [
        r'(?:E-?Way\s*Bill\s*(?:No|Number)?\s*[:\-]?\s*)(\d{12})',
    ],
    'distance_km': [
        r'(?:Distance\s*[:\-]?\s*)(\d+)\s*(?:km|kms|kilometers?)',
    ],
    'hsn_code': [
        r'(?:HSN\s*(?:Code|No)?\s*[:\-]?\s*)(\d{4,8})',
    ],
}


def extract_field(text: str, field: str) -> str | None:
    patterns = PATTERNS.get(field, [])
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().replace(',', '')
    return None


def parse_amount(raw: str | None) -> float:
    if raw is None:
        return 0.0
    try:
        return float(str(raw).replace(',', '').replace('₹', '').strip())
    except ValueError:
        return 0.0


def parse_date(raw: str | None) -> str:
    if raw is None:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d', '%d %B %Y', '%d %b %Y']:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    return raw


def extract_invoice_from_text(text: str, vendor_id: str = 'UNKNOWN') -> dict:
    """
    Parse raw OCR text / extracted PDF text into structured invoice dict.
    Returns a dict compatible with InvoiceFraudDetector.predict().
    """
    inv_number = extract_field(text, 'invoice_number') or 'OCR-UNKNOWN'
    inv_date   = parse_date(extract_field(text, 'invoice_date'))
    gstin      = extract_field(text, 'gstin') or 'INVALID'
    raw_amt    = extract_field(text, 'amount')
    raw_tax    = extract_field(text, 'tax_amount')
    vehicle    = extract_field(text, 'vehicle_number') or ''
    eway_num   = extract_field(text, 'eway_bill_number') or ''
    distance   = extract_field(text, 'distance_km')
    hsn        = extract_field(text, 'hsn_code') or '9999'

    invoice_amount = parse_amount(raw_amt)
    tax_amount     = parse_amount(raw_tax)

    # Infer tax rate from amount and tax
    expected_tax_rate = 0.18
    if invoice_amount > 0 and tax_amount > 0:
        calc = tax_amount / max(invoice_amount - tax_amount, 1)
        # Snap to known GST slabs
        for slab in [0.05, 0.12, 0.18, 0.28]:
            if abs(calc - slab) < 0.03:
                expected_tax_rate = slab
                break

    return {
        'invoice_number':       inv_number,
        'invoice_date':         inv_date,
        'vendor_id':            vendor_id,
        'vendor_name':          vendor_id,
        'vendor_gstin':         gstin,
        'vendor_state':         gstin[:2] if len(gstin) >= 2 else '27',
        'buyer_state':          'MH',
        'hsn_code':             hsn,
        'invoice_amount':       invoice_amount,
        'tax_amount':           tax_amount,
        'expected_tax_rate':    expected_tax_rate,
        'transport_mode':       'road',
        'transport_distance_km': float(distance or 0),
        'vehicle_number':       vehicle,
        'eway_validity_hours':  24,
        'actual_transit_hours': 10,
        'vendor_reg_days':      365,
        '_source':              'ocr',
        '_eway_bill_number':    eway_num,
    }


def process_ocr_texts(texts: list[str], vendor_ids: list[str] = None) -> list[dict]:
    """Batch process list of OCR texts → list of invoice dicts"""
    results = []
    for i, text in enumerate(texts):
        vid = (vendor_ids or [])[i] if vendor_ids and i < len(vendor_ids) else f'OCR_V{i+1}'
        results.append(extract_invoice_from_text(text, vid))
    return results


# ─── DEMO ────────────────────────────────────────────────────────────────────

SAMPLE_OCR_TEXT = """
TAX INVOICE

Invoice No   : INV-2024-000781
Invoice Date : 15/03/2024

Supplier:
  ABC Traders Pvt Ltd
  GSTIN: 27ABCDE1234F1Z5
  Mumbai, Maharashtra

Buyer:
  XYZ Corp Ltd
  GSTIN: 29XYZPQ5678G1Z8
  Bengaluru, Karnataka

HSN Code : 8471
Description: Computer Hardware
Quantity   : 10 units

  Taxable Value : Rs. 85,000.00
  CGST (9%)     : Rs.  7,650.00
  SGST (9%)     : Rs.  7,650.00
  Total Amount  : Rs. 1,00,300.00

E-Way Bill No : 231042507891
Vehicle No    : MH12AB1234
Distance      : 980 km
"""


if __name__ == '__main__':
    invoice = extract_invoice_from_text(SAMPLE_OCR_TEXT, vendor_id='V0042')
    print("Extracted Invoice:")
    print(json.dumps(invoice, indent=2))
