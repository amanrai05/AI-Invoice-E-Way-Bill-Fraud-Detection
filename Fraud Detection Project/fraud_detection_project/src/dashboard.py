"""
dashboard.py — Generates HTML fraud detection report/dashboard
Author : Aman Kumar | Roll No: 2310991770
"""

import os
import pandas as pd
from datetime import datetime


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Fraud Detection Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }}
  header {{ background: linear-gradient(135deg, #1e3a8a, #0f172a);
            padding: 24px 32px; border-bottom: 2px solid #334155; }}
  header h1 {{ font-size: 1.8rem; color: #60a5fa; }}
  header p  {{ font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }}
  .kpi-bar {{ display: flex; gap: 16px; padding: 24px 32px; flex-wrap: wrap; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px;
          padding: 18px 24px; flex: 1; min-width: 160px; }}
  .kpi .val {{ font-size: 2rem; font-weight: 700; }}
  .kpi .lbl {{ font-size: 0.78rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; }}
  .kpi.critical .val {{ color: #f87171; }}
  .kpi.high .val     {{ color: #fb923c; }}
  .kpi.medium .val   {{ color: #facc15; }}
  .kpi.ok .val       {{ color: #4ade80; }}
  .section {{ padding: 0 32px 32px; }}
  h2 {{ font-size: 1.1rem; color: #93c5fd; margin-bottom: 14px; padding-top: 8px;
        border-top: 1px solid #1e293b; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #1e3a8a; color: #bfdbfe; padding: 10px 12px; text-align: left; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #1e293b; }}
  .badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; }}
  .CRITICAL {{ background: #7f1d1d; color: #fca5a5; }}
  .HIGH     {{ background: #7c2d12; color: #fdba74; }}
  .MEDIUM   {{ background: #713f12; color: #fde68a; }}
  .LOW      {{ background: #14532d; color: #86efac; }}
  .score-bar {{ display: flex; align-items: center; gap: 8px; }}
  .bar-outer {{ background: #334155; border-radius: 4px; height: 8px; width: 100px; }}
  .bar-inner {{ height: 8px; border-radius: 4px; }}
  footer {{ padding: 16px 32px; color: #475569; font-size: 0.78rem;
            border-top: 1px solid #1e293b; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>🛡️ AI Invoice &amp; E-Way Bill Fraud Detection</h1>
  <p>Participant: Aman Kumar &nbsp;|&nbsp; Roll No: 2310991770 &nbsp;|&nbsp;
     Generated: {generated_at}</p>
</header>

<div class="kpi-bar">
  <div class="kpi ok">
    <div class="val">{total}</div>
    <div class="lbl">Total Invoices</div>
  </div>
  <div class="kpi critical">
    <div class="val">{critical}</div>
    <div class="lbl">Critical Risk</div>
  </div>
  <div class="kpi high">
    <div class="val">{high}</div>
    <div class="lbl">High Risk</div>
  </div>
  <div class="kpi medium">
    <div class="val">{medium}</div>
    <div class="lbl">Medium Risk</div>
  </div>
  <div class="kpi ok">
    <div class="val">{flagged_pct}%</div>
    <div class="lbl">Flagged Rate</div>
  </div>
  <div class="kpi ok">
    <div class="val">₹{total_amt}</div>
    <div class="lbl">Total Value at Risk</div>
  </div>
</div>

<div class="section">
  <h2>🚨 Flagged Transactions</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Invoice No</th>
        <th>Vendor</th>
        <th>Date</th>
        <th>Amount (₹)</th>
        <th>Risk Score</th>
        <th>Risk Level</th>
        <th>Triggered Rules</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
</div>

<footer>
  AI Invoice &amp; E-Way Bill Fraud Detection &nbsp;|&nbsp;
  Aman Kumar &nbsp;|&nbsp; Roll No: 2310991770
</footer>
</body>
</html>"""


def score_color(score):
    if score >= 75: return '#ef4444'
    if score >= 55: return '#f97316'
    if score >= 35: return '#eab308'
    return '#22c55e'


def generate_dashboard(results_csv='output/detection_results.csv',
                       out_html='output/dashboard.html'):
    if not os.path.exists(results_csv):
        print(f"[Dashboard] Results not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    flagged = df[df['flag_for_review'] == True].sort_values('final_risk_score', ascending=False)

    total    = len(df)
    critical = int((df['risk_level'] == 'CRITICAL').sum())
    high     = int((df['risk_level'] == 'HIGH').sum())
    medium   = int((df['risk_level'] == 'MEDIUM').sum())
    pct      = round(100 * len(flagged) / max(total, 1), 1)
    amt_risk = f"{flagged['invoice_amount'].sum():,.0f}"

    table_rows = []
    for i, row in enumerate(flagged.itertuples(), 1):
        score = row.final_risk_score
        color = score_color(score)
        bar_w = min(int(score), 100)
        table_rows.append(f"""      <tr>
        <td>{i}</td>
        <td><b>{row.invoice_number}</b></td>
        <td>{row.vendor_name if hasattr(row, 'vendor_name') else row.vendor_id}</td>
        <td>{str(row.invoice_date)[:10]}</td>
        <td>₹{float(row.invoice_amount):,.2f}</td>
        <td>
          <div class="score-bar">
            <div class="bar-outer"><div class="bar-inner" style="width:{bar_w}%;background:{color}"></div></div>
            <span style="color:{color}">{score:.1f}</span>
          </div>
        </td>
        <td><span class="badge {row.risk_level}">{row.risk_level}</span></td>
        <td><small>{row.triggered_rules}</small></td>
      </tr>""")

    html = HTML_TEMPLATE.format(
        generated_at=datetime.now().strftime('%d %b %Y %H:%M'),
        total=total, critical=critical, high=high, medium=medium,
        flagged_pct=pct, total_amt=amt_risk,
        rows='\n'.join(table_rows)
    )

    os.makedirs('output', exist_ok=True)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[Dashboard] Saved → {out_html}")


if __name__ == '__main__':
    generate_dashboard()
