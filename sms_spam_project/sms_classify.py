#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sms_classifier.predictor import classify_csv

# --- Windows console fix for tqdm/logging redraws (safe no-op elsewhere) ---
try:
    import colorama
    colorama.just_fix_windows_console()
except Exception:
    pass

# --- Optional: silence sklearn pickle version mismatch warnings globally ---
import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


# ============================================================
# CLI + Validation Utilities
# ============================================================

class FriendlyArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser with clearer 'missing required' messages.
    """
    def error(self, message: str) -> None:
        if "the following arguments are required:" in message:
            missing = message.split(":")[-1].strip()
            sys.stderr.write(f"\n✖ Missing required option(s): {missing}\n")
            sys.stderr.write("   Use -h/--help to see usage and examples.\n\n")
            self.print_help(sys.stderr)
            sys.exit(2)
        super().error(message)


@dataclass
class Settings:
    """
    Holds validated runtime configuration for the CLI.
    All paths are resolved and validated before use.
    """
    input: Path
    output: Path
    model: Path
    text_col: str
    chunksize: int
    batch_size: int
    workers: int
    # deprecated / no-op
    proba: bool
    # new predictor options
    threshold: float
    label_column: str
    proba_column: str
    show_batch_progress: bool
    # misc
    benchmark: bool
    force: bool
    log_level: str


def parse_args(argv: list[str] | None = None) -> Settings:
    """
    Parse command-line arguments and return concrete Settings.
    """
    p = FriendlyArgumentParser(
        description="Classify SMS messages with a pre-trained model."
    )
    p.add_argument("--input", "-i", required=True,
                   help="Path to input CSV (must contain a text column).")
    p.add_argument("--output", "-o", required=True,
                   help="Path to write predictions CSV.")
    p.add_argument("--model", "-m", default="model.pkl",
                   help="Path to model.pkl file.")
    p.add_argument("--text-col", default="message",
                   help="Name of the text column in the CSV.")
    p.add_argument("--chunksize", type=int, default=5000,
                   help="Rows per streaming chunk when reading the CSV.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Texts per inference batch.")
    p.add_argument("--workers", type=int, default=0,
                   help="Number of worker processes. 0/1 = single-process.")

    # Deprecated: predictor always writes probabilities now
    p.add_argument("--proba", action="store_true",
                   help="(Deprecated) No-op: probabilities are always written.")

    # New knobs to align with predictor.py
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for positive label (default: 0.5).")
    p.add_argument("--label-column", default="prediction",
                   help="Output column name for labels (default: prediction).")
    p.add_argument("--proba-column", default="spam_probability",
                   help="Output column name for probabilities (default: spam_probability).")
    p.add_argument("--show-batch-progress", action="store_true",
                   help="Show a nested progress bar for batches inside each chunk.")

    p.add_argument("--benchmark", action="store_true",
                   help="Print throughput (rows/sec) at the end.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output file if present.")
    p.add_argument("--log-level", default=os.getenv("SMS_LOG_LEVEL", "INFO"),
                   choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                   help="Logging level (default: %(default)s).")

    a = p.parse_args(argv)

    settings = Settings(
        input=Path(a.input),
        output=Path(a.output),
        model=Path(a.model),
        text_col=a.text_col,
        chunksize=a.chunksize,
        batch_size=a.batch_size,
        workers=a.workers,
        proba=a.proba,  # ignored
        threshold=a.threshold,
        label_column=a.label_column,
        proba_column=a.proba_column,
        show_batch_progress=a.show_batch_progress,
        benchmark=a.benchmark,
        force=a.force,
        log_level=a.log_level.upper(),
    )
    validate_settings(settings, parser=p)
    return settings


def validate_settings(s: Settings, parser: argparse.ArgumentParser) -> None:
    """
    Validate parsed CLI settings for existence, permissions, and logical constraints.
    """
    problems: list[str] = []

    # Files/dirs
    if not s.input.exists():
        problems.append(f"Input CSV not found: {s.input!s}")
    elif not s.input.is_file():
        problems.append(f"Input path is not a file: {s.input!s}")

    if not s.model.exists():
        problems.append(f"Model not found: {s.model!s}")
    elif not s.model.is_file():
        problems.append(f"Model path is not a file: {s.model!s}")

    out_dir = s.output.parent if s.output.parent != Path("") else Path(".")
    if not out_dir.exists():
        problems.append(f"Output directory does not exist: {out_dir!s}")
    elif not os.access(out_dir, os.W_OK):
        problems.append(f"Output directory is not writable: {out_dir!s}")
    elif s.output.exists() and not s.force:
        problems.append(
            f"Output file already exists: {s.output!s} "
            f"(use --force to overwrite)"
        )

    # Numeric
    if not s.text_col or not s.text_col.strip():
        problems.append("Text column name (--text-col) must be a non-empty string.")
    if s.chunksize <= 0:
        problems.append("--chunksize must be a positive integer.")
    if s.batch_size <= 0:
        problems.append("--batch-size must be a positive integer.")
    if s.workers < 0:
        problems.append("--workers must be >= 0.")
    if not (0.0 <= s.threshold <= 1.0):
        problems.append("--threshold must be between 0.0 and 1.0.")

    if problems:
        sys.stderr.write("\n✖ Invalid arguments:\n")
        for msg in problems:
            sys.stderr.write(f"  - {msg}\n")
        sys.stderr.write("\n")
        parser.print_help(sys.stderr)
        sys.exit(2)


def setup_logging(level: str) -> None:
    """
    Configure the root logger for console output.
    """
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.debug("Logging initialized at %s", level)


# ============================================================
# Main entrypoint
# ============================================================

def main(argv: list[str] | None = None) -> int:
    """
    Parse args, configure logging, and run classification.
    """
    # Graceful Ctrl-C
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))  # 130 = SIGINT

    args = parse_args(argv)
    setup_logging(args.log_level)

    logging.info("Starting classification")
    logging.debug(
        "Config: input=%s output=%s model=%s text_col=%s chunksize=%d batch_size=%d "
        "workers=%d force=%s log_level=%s threshold=%.3f label_col=%s proba_col=%s "
        "show_batch_progress=%s",
        args.input, args.output, args.model, args.text_col,
        args.chunksize, args.batch_size, args.workers,
        args.force, args.log_level, args.threshold,
        args.label_column, args.proba_column, args.show_batch_progress,
    )

    start = time.perf_counter()

    try:
        rows_in, rows_out = classify_csv(
            input_csv=str(args.input),
            output_csv=str(args.output),
            model_path=str(args.model),
            text_column=args.text_col,
            chunksize=args.chunksize,
            batch_size=args.batch_size,
            workers=args.workers,
            threshold=args.threshold,
            proba_column=args.proba_column,
            label_column=args.label_column,
            show_batch_progress=args.show_batch_progress,
        )
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as e:
        logging.exception("Classification failed: %s", e)
        return 1

    elapsed = time.perf_counter() - start
    if args.benchmark:
        rps = (rows_out / elapsed) if elapsed > 0 else 0.0
        logging.info("Processed %d rows in %.2fs -> %.1f rows/sec", rows_out, elapsed, rps)
    else:
        logging.info("Wrote %d predictions to '%s'.", rows_out, args.output)

    if rows_in != rows_out:
        logging.warning("Row count mismatch: input=%d, output=%d", rows_in, rows_out)

    logging.info("Done.")
    return 0


# ============================================================
# Script entrypoint
# ============================================================

if __name__ == "__main__":
    sys.exit(main())
