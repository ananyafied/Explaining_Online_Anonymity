"""Saving and loading step-2 results: outputs/step2_explain/<family>/*.csv"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import Data


def out_dir(D: Data, family: str) -> Path:
    p = D.root / "outputs" / "step2_explain" / family
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table(D: Data, family: str, name: str, df: pd.DataFrame, index=False):
    path = out_dir(D, family) / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def save_scorecard(D: Data, family: str, scorecard, reload_checks=None, meta=None):
    """Writes scorecard.csv (long format), reload_checks.csv and meta.json for one notebook."""
    p = save_table(D, family, "scorecard", scorecard.df)
    if reload_checks is not None:
        save_table(D, family, "reload_checks", pd.DataFrame(reload_checks))
    if meta is not None:
        (out_dir(D, family) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("saved", p)
    return p
