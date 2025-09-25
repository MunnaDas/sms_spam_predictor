from __future__ import annotations
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

from .model_loader import ModelLoader
from .io_utils import read_messages_csv
from .batching import batched


def _predict_batch(model_path: str, texts: list[str]) -> list[int]:
    # Each process loads its own model instance for CPU parallelism
    loader = ModelLoader(model_path)
    return loader.predict(texts)


def classify_csv(
    input_csv: str,
    output_csv: str,
    model_path: str,
    text_column: str = "message",
    chunksize: Optional[int] = 5000,
    batch_size: int = 512,
    workers: int = 0,
    include_proba: bool = False,
) -> Tuple[int, int]:
    """Classify messages from input_csv and write results to output_csv.

    Returns: (rows_read, rows_written)
    """
    rows_read = 0
    rows_written = 0
    first_write = True

    if workers and workers > 1:
        # Parallel, ordered inference per chunk using indexed futures
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in read_messages_csv(input_csv, text_column=text_column, chunksize=chunksize):
                texts = chunk[text_column].astype(str).tolist()
                rows_read += len(texts)

                # Create indexed batches so we can reconstruct order
                indexed_batches = [(i, b) for i, b in enumerate(batched(texts, batch_size))]
                future_to_index = {ex.submit(_predict_batch, model_path, b): i for i, b in indexed_batches}

                # Collect results as they complete
                results: dict[int, list[int]] = {}
                for fut in as_completed(future_to_index):
                    idx = future_to_index[fut]
                    results[idx] = fut.result()

                # Flatten in original order
                ordered_preds: list[int] = []
                for i in range(len(indexed_batches)):
                    ordered_preds.extend(results[i])

                # Optional probability (single-process for simplicity & determinism)
                proba = None
                if include_proba:
                    loader = ModelLoader(model_path)
                    proba = loader.predict_proba(texts)

                out = chunk.copy()
                out["prediction"] = ordered_preds[: len(out)]
                if proba is not None:
                    out["spam_probability"] = proba[: len(out)]

                mode = "w" if first_write else "a"
                header = first_write
                out.to_csv(output_csv, index=False, mode=mode, header=header)
                if first_write:
                    first_write = False
                rows_written += len(out)
    else:
        # Single-process, streaming
        loader = ModelLoader(model_path)
        for chunk in read_messages_csv(input_csv, text_column=text_column, chunksize=chunksize):
            texts = chunk[text_column].astype(str).tolist()
            rows_read += len(texts)

            preds = loader.predict(texts)
            proba = loader.predict_proba(texts) if include_proba else None

            out = chunk.copy()
            out["prediction"] = preds
            if proba is not None:
                out["spam_probability"] = proba

            mode = "w" if first_write else "a"
            header = first_write
            out.to_csv(output_csv, index=False, mode=mode, header=header)
            if first_write:
                first_write = False
            rows_written += len(out)

    return rows_read, rows_written
