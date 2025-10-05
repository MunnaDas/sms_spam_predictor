
from __future__ import annotations
from typing import Iterable, List, TypeVar

T = TypeVar("T")

def batched(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
