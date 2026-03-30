from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)
    return output_path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_separator(path: str | Path) -> str:
    suffixes = Path(path).suffixes
    if ".tsv" in suffixes or ".bed" in suffixes:
        return "\t"
    return ","


def read_table(path: str | Path, sep: str | None = None, **kwargs: Any) -> pd.DataFrame:
    table_path = Path(path)
    return pd.read_csv(
        table_path,
        sep=sep or infer_separator(table_path),
        compression="infer",
        **kwargs,
    )


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=index)
    return output_path


def parse_json_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parsed = json.loads(text)
    return [float(item) for item in parsed]

