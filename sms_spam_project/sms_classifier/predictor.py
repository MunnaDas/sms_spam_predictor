from __future__ import annotations

import os
import sys
import math
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List
import difflib
import logging

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Make Windows consoles handle carriage returns/ANSI properly
try:
    import colorama
    colorama.just_fix_windows_console()
except Exception:
    pass

# --- Optional: silence sklearn pickle version warnings (you added this) ---
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from .batching import batched
from .io_utils import read_messages_csv
from .model_loader import ModelLoader


# -----------------------------
# Worker-side model management
# -----------------------------

_WORKER_LOADER: ModelLoader | None = None


def _init_worker(model_path: str) -> None:
    """Each worker loads its own model instance (avoids per-batch reloads)."""
    global _WORKER_LOADER
    _WORKER_LOADER = ModelLoader(model_path)


def _predict_proba_worker(texts: list[str]) -> list[float]:
    """Predict spam probabilities using the process-global model instance."""
    if _WORKER_LOADER is None:
        raise RuntimeError("Worker model not initialized")
    if not hasattr(_WORKER_LOADER, "predict_proba"):
        raise RuntimeError("ModelLoader does not implement predict_proba()")
    return _WORKER_LOADER.predict_proba(texts)  # type: ignore[attr-defined]


# -----------------------------
# Schema pre-validation (fail fast)
# -----------------------------

def _resolve_text_column(input_csv: str, requested: str) -> str:
    """
    Validate that `requested` exists in the CSV header.
    - Case-insensitive match is accepted (auto-corrects to real casing).
    - If missing, suggests closest column names.

    Returns:
        The exact column name as it appears in the CSV.

    Raises:
        ValueError if column is not found.
    """
    try:
        # Read only header; let pandas infer compression by suffix (.gz, .zip, etc.)
        header_df = pd.read_csv(input_csv, nrows=0)
    except Exception as e:
        raise ValueError(f"Unable to read CSV header from '{input_csv}': {e}") from e

    cols = list(header_df.columns)
    if not cols:
        raise ValueError(f"Input CSV '{input_csv}' has no columns.")

    # Exact match first
    if requested in cols:
        return requested

    # Case-insensitive match
    lowered = {c.lower(): c for c in cols}
    key = requested.lower()
    if key in lowered:
        # auto-correct to exact casing from file
        return lowered[key]

    # Closest suggestions
    suggestions = difflib.get_close_matches(requested, cols, n=5, cutoff=0.6)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""

    raise ValueError(
        f"Text column '{requested}' not found in CSV header. "
        f"Available columns: {', '.join(cols)}.{hint}"
    )


# -----------------------------
# Progress helpers
# -----------------------------

def _count_rows_fast(csv_path: str) -> int | None:
    """
    Fast line counter for plain-text CSVs (uncompressed).
    Returns number of data rows excluding the header, or None if unknown.
    """
    p = Path(csv_path)
    # Heuristic: if compressed, skip fast count
    if p.suffix.lower() in {".gz", ".bz2", ".xz", ".zip"}:
        return None
    try:
        count = 0
        # 1MB buffered reads; count b'\n'
        with open(p, "rb", buffering=1024 * 1024) as f:
            while True:
                buf = f.read(1024 * 1024)
                if not buf:
                    break
                count += buf.count(b"\n")
        # subtract header if any lines
        return max(0, count - 1)
    except Exception:
        return None


# -----------------------------
# Main classification pipeline
# -----------------------------

def classify_csv(
    input_csv: str,
    output_csv: str,
    model_path: str,
    text_column: str = "message",
    chunksize: Optional[int] = 5000,
    batch_size: int = 512,
    workers: int = 0,
    threshold: float = 0.5,
    proba_column: str = "spam_probability",
    label_column: str = "prediction",
    show_batch_progress: bool = False,
) -> Tuple[int, int]:
    """
    Classify messages from input_csv and write results to output_csv.

    - Validates schema up front (fails fast if `text_column` is missing).
    - Always outputs both probability and class label columns.
    - Displays a robust progress bar (Windows-friendly; true bar when row count known).
    - Atomic first write, streaming append thereafter.
    - Parallelized inference with per-worker model loading.

    Returns:
        (rows_read, rows_written)
    """
    # ---- Fail fast: verify input schema and normalize the column name ----
    resolved_text_col = _resolve_text_column(input_csv, text_column)

    rows_read = 0
    rows_written = 0
    first_write = True
    start_time = time.perf_counter()

    out_path = Path(output_csv)
    tmp_dir = out_path.parent if out_path.parent != Path("") else Path(".")
    tmp_prefix = f".{out_path.name}."

    def _to_strings(series: pd.Series) -> list[str]:
        # Robust conversion: nullable string dtype + no 'nan' tokens
        return series.astype("string").fillna("").tolist()

    def _write_chunk(df: pd.DataFrame) -> None:
        nonlocal first_write
        if first_write:
            fd, tmp_path = tempfile.mkstemp(prefix=tmp_prefix, dir=tmp_dir)
            os.close(fd)
            tmp_file = Path(tmp_path)
            df.to_csv(tmp_file, index=False)
            os.replace(tmp_file, out_path)  # atomic
            first_write = False
        else:
            df.to_csv(out_path, index=False, mode="a", header=False)

    # Build the base chunk iterator from your CSV reader
    base_iter = read_messages_csv(input_csv, text_column=resolved_text_col, chunksize=chunksize)

    # Estimate total chunks so tqdm can draw a *visual* bar
    total_chunks: int | None = None
    if chunksize and chunksize > 0:
        total_rows = _count_rows_fast(input_csv)
        if total_rows is not None:
            total_chunks = max(1, math.ceil(total_rows / chunksize))

    # Wrap logging so log lines don't break the bar; use stdout + ASCII for Windows
    with logging_redirect_tqdm():
        chunk_iter = tqdm(
            base_iter,
            desc="Classifying chunks",
            unit="chunk",
            total=total_chunks,    # enables visual bar + percent
            disable=False,         # force rendering even if not a TTY
            file=sys.stdout,       # avoid stderr clashes
            dynamic_ncols=True,
            mininterval=0.2,
            ascii=True,            # safer in Windows consoles
            leave=True,
            smoothing=0,
        )

        if workers and workers > 1:
            # -------- Parallel path --------
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(model_path,),
            ) as ex:
                for chunk in chunk_iter:
                    texts = _to_strings(chunk[resolved_text_col])
                    rows_read += len(texts)

                    batches: list[list[str]] = list(batched(texts, batch_size))
                    futures = {ex.submit(_predict_proba_worker, b): i for i, b in enumerate(batches)}

                    ordered_blocks: list[List[float] | None] = [None] * len(batches)
                    if show_batch_progress:
                        batch_pbar = tqdm(
                            total=len(batches),
                            desc="Batches",
                            unit="batch",
                            disable=False,
                            file=sys.stdout,
                            dynamic_ncols=True,
                            mininterval=0.1,
                            ascii=True,
                            leave=False,
                            smoothing=0,
                        )
                    else:
                        batch_pbar = None

                    try:
                        for fut in as_completed(futures):
                            idx = futures[fut]
                            ordered_blocks[idx] = fut.result()
                            if batch_pbar is not None:
                                batch_pbar.update(1)
                    except Exception:
                        for f in futures:
                            f.cancel()
                        if batch_pbar is not None:
                            batch_pbar.close()
                        raise
                    finally:
                        if batch_pbar is not None:
                            batch_pbar.close()

                    probs: list[float] = [p for block in ordered_blocks for p in block]  # type: ignore[arg-type]
                    labels: list[int] = [1 if p >= threshold else 0 for p in probs]

                    out = chunk.copy()
                    out[label_column] = labels[: len(out)]
                    out[proba_column] = probs[: len(out)]

                    _write_chunk(out)
                    rows_written += len(out)

                    # Update throughput postfix
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    rps = rows_written / elapsed
                    chunk_iter.set_postfix(rows=rows_written, rps=f"{rps:.1f}")
        else:
            # -------- Single-process path --------
            loader = ModelLoader(model_path)
            if not hasattr(loader, "predict_proba"):
                raise RuntimeError("ModelLoader does not implement predict_proba().")

            for chunk in chunk_iter:
                texts = _to_strings(chunk[resolved_text_col])
                rows_read += len(texts)

                probs = loader.predict_proba(texts)  # type: ignore[attr-defined]
                labels = [1 if p >= threshold else 0 for p in probs]

                out = chunk.copy()
                out[label_column] = labels
                out[proba_column] = probs

                _write_chunk(out)
                rows_written += len(out)

                elapsed = max(time.perf_counter() - start_time, 1e-6)
                rps = rows_written / elapsed
                chunk_iter.set_postfix(rows=rows_written, rps=f"{rps:.1f}")

        chunk_iter.close()

    logging.getLogger(__name__).info(
        "Finished classify_csv: rows_in=%d rows_out=%d threshold=%.3f",
        rows_read, rows_written, threshold
    )
    return rows_read, rows_written
