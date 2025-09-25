
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import time
from sms_classifier.predictor import classify_csv

def parse_args():
    p = argparse.ArgumentParser(description="Classify SMS messages with a pre-trained model.")
    p.add_argument("--input", "-i", required=True, help="Path to input CSV (must contain a text column).")
    p.add_argument("--output", "-o", required=True, help="Path to write predictions CSV.")
    p.add_argument("--model", "-m", default="model.pkl", help="Path to model.pkl")
    p.add_argument("--text-col", default="message", help="Name of the text column in the CSV.")
    p.add_argument("--chunksize", type=int, default=5000, help="Rows per streaming chunk when reading the CSV.")
    p.add_argument("--batch-size", type=int, default=512, help="Texts per inference batch.")
    p.add_argument("--workers", type=int, default=0, help="Number of worker processes. 0/1 = single-process.")
    p.add_argument("--proba", action="store_true", help="Also output spam probability if model supports it.")
    p.add_argument("--benchmark", action="store_true", help="Print throughput (rows/sec) at the end.")
    return p.parse_args()

def main():
    args = parse_args()
    start = time.perf_counter()
    rows_in, rows_out = classify_csv(
        input_csv=args.input,
        output_csv=args.output,
        model_path=args.model,
        text_column=args.text_col,
        chunksize=args.chunksize,
        batch_size=args.batch_size,
        workers=args.workers,
        include_proba=args.proba,
    )
    elapsed = time.perf_counter() - start
    if args.benchmark:
        rps = (rows_out / elapsed) if elapsed > 0 else 0
        print(f"Processed {rows_out} rows in {elapsed:.2f}s -> {rps:.1f} rows/sec")
    else:
        print(f"Wrote {rows_out} predictions to '{args.output}'.")


if __name__ == "__main__":
    main()
