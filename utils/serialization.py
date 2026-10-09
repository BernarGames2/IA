"""JSON-safe conversion of numpy/pandas/dataclass/pydantic objects.

Kept free of plotting imports so the web API can serialise results without matplotlib.
Non-finite floats become ``None`` (JSON has no NaN/inf).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


def to_jsonable(o: object) -> object:  # noqa: C901
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="split", date_format="iso", default_handler=str))
    if isinstance(o, pd.Series):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [to_jsonable(x) for x in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o) and not isinstance(o, type):
        return to_jsonable(asdict(o))
    if hasattr(o, "model_dump"):
        return to_jsonable(o.model_dump())  # type: ignore[attr-defined]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)
