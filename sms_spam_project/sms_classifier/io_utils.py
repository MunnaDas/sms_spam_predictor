
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Iterator, Optional

def read_messages_csv(path: str, text_column: str = "message", chunksize: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Stream messages from a CSV file.
    If chunksize is None, yields a single DataFrame. Otherwise yields chunks.
    """
    if chunksize:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            if text_column not in chunk.columns:
                raise KeyError(f"Column '{text_column}' not found in CSV. Available: {list(chunk.columns)}")
            yield chunk
    else:
        df = pd.read_csv(path)
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
        yield df

def write_predictions_csv(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
