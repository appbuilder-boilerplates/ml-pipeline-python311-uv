"""One data file into what the model needs: features, labels as the file holds them, one id per row."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


class Dataset(TypedDict):
    labels: list[Any]
    ids: list[str]
    features: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    """A CSV, TSV or JSONL file as a DataFrame (the formats the eval platform also reads)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported data file {str(path)!r}: use .csv, .tsv or .jsonl")


def load_data(path: str | Path, *, label_column: str = "label", id_column: str = "id") -> Dataset:
    """Every row of `path`: its id (as text, unique), its label exactly as the file holds it, and the remaining
    columns as features. Labels are encoded for the model inside `train`/`predict`, never here."""
    frame = read_table(path)
    missing = [column for column in (label_column, id_column) if column not in frame.columns]
    if missing:
        columns = ", ".join(map(str, frame.columns))
        raise ValueError(f"{path}: missing column(s) {', '.join(missing)}; columns are {columns}")
    ids = frame[id_column].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: the id column {id_column!r} has duplicate values; every row needs a unique id")
    return {
        "labels": frame[label_column].tolist(),
        "ids": ids,
        "features": frame.drop(columns=[label_column, id_column]),
    }
