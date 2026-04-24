"""
main.py — Entry point
AI Invoice & E-Way Bill Fraud Detection
Author : Aman Kumar | Roll No: 2310991770
"""

import os, sys, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_sample_data
from fraud_detector import InvoiceFraudDetector


def train_and_evaluate(train_csv=None, test_csv=None):
    print("=" * 60)
    print("  AI Invoice & E-Way Bill Fraud Detection")
    print("  Participant : Aman Kumar | Roll: 2310991770")
    print("=" * 60 + "\n")

    if train_csv and os.path.exists(train_csv):
        df_train = pd.read_csv(train_csv)
        df_test  = pd.read_csv(test_csv) if test_csv and os.path.exists(test_csv) else df_train.sample(50)
        labels_train = df_train.get('is_fraud', None)
        if labels_train is not None:
            labels_train = labels_train.values
    else:
        print("[INFO] No CSV provided — generating synthetic data...\n")
        os.makedirs('data/sample', exist_ok=True)
        df_train, df_test, labels_train, _ = generate_sample_data(n_train=600, n_test=100)

    detector = InvoiceFraudDetector()
    detector.fit(df_train, labels_train)
    detector.save('models/detector.pkl')

    print("Running predictions on test set...\n")
    results = detector.predict(df_test)

    os.makedirs('output', exist_ok=True)
    results.to_csv('output/detection_results.csv', index=False)

    flagged  = results[results['flag_for_review']].sort_values('final_risk_score', ascending=False)
    critical = results[results['risk_level'] == 'CRITICAL']
    high     = results[results['risk_level'] == 'HIGH']

    print("─" * 60)
    print("  DETECTION SUMMARY")
    print("─" * 60)
    print(f"  Total invoices analysed : {len(results)}")
    print(f"  Flagged for review      : {len(flagged)}  ({100*len(flagged)/max(len(results),1):.1f}%)")
    print(f"  CRITICAL risk           : {len(critical)}")
    print(f"  HIGH risk               : {len(high)}")
    print(f"  Results saved           : output/detection_results.csv")
    print("─" * 60)

    if len(flagged) > 0:
        print("\n  TOP 5 FLAGGED INVOICES:")
        print(flagged[['invoice_number', 'vendor_id', 'invoice_amount',
                        'final_risk_score', 'risk_level', 'triggered_rules']].head(5).to_string(index=False))
        print()

        # Explain top flagged
        print("\n  DETAILED EXPLANATION — TOP FRAUD CANDIDATE:")
        top = flagged.iloc[0].to_dict()
        print(detector.explain(top))

    return results, detector


def run_on_file(input_csv: str):
    """Score a new CSV file using saved model (or retrain on-the-fly)."""
    if not os.path.exists(input_csv):
        print(f"[ERROR] File not found: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {len(df)} records from {input_csv}")

    model_path = 'models/detector.pkl'
    if os.path.exists(model_path):
        print(f"[INFO] Loading saved model from {model_path}")
        detector = InvoiceFraudDetector.load(model_path)
    else:
        print("[INFO] No saved model found — training on input data...")
        labels = df.get('is_fraud', None)
        if labels is not None:
            labels = labels.values
        detector = InvoiceFraudDetector()
        detector.fit(df, labels)
        detector.save(model_path)

    results = detector.predict(df)
    os.makedirs('output', exist_ok=True)
    out_path = 'output/scored_' + os.path.basename(input_csv)
    results.to_csv(out_path, index=False)
    print(f"[INFO] Scored results → {out_path}")

    flagged = results[results['flag_for_review']]
    print(f"\nFlagged: {len(flagged)}/{len(results)} invoices")
    if len(flagged):
        print(flagged[['invoice_number', 'final_risk_score', 'risk_level', 'triggered_rules']].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='AI Invoice Fraud Detector')
    parser.add_argument('--mode',  choices=['train', 'score'], default='train',
                        help='train: full pipeline | score: score a CSV file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input CSV (for score mode)')
    parser.add_argument('--train', type=str, default=None,
                        help='Path to training CSV (optional for train mode)')
    parser.add_argument('--test',  type=str, default=None,
                        help='Path to test CSV (optional for train mode)')
    args = parser.parse_args()

    if args.mode == 'train':
        train_and_evaluate(args.train, args.test)
    elif args.mode == 'score':
        if not args.input:
            print("[ERROR] --input required for score mode")
            sys.exit(1)
        run_on_file(args.input)


if __name__ == '__main__':
    main()
