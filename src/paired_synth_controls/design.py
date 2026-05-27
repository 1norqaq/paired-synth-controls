"""Design-matrix helpers for anonymous categorical audit data."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DesignSpec:
    outcome_col: str
    q_col: str
    cluster_col: str | None
    categorical_cols: list[str]
    references: dict[str, str]


def build_category_orders(df: pd.DataFrame, spec: DesignSpec) -> dict[str, list[str]]:
    orders: dict[str, list[str]] = {}
    for col in spec.categorical_cols:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        ref = str(spec.references[col])
        if ref not in vals:
            raise ValueError(f"Reference {ref!r} not found in column {col!r}; found {vals[:10]}...")
        orders[col] = [ref] + [v for v in vals if v != ref]
    return orders


def codes_from_orders(df: pd.DataFrame, orders: dict[str, list[str]]) -> dict[str, np.ndarray]:
    codes: dict[str, np.ndarray] = {}
    for col, cats in orders.items():
        c = pd.Categorical(df[col].astype(str), categories=cats).codes.astype(np.int16)
        if (c < 0).any():
            raise ValueError(f"Column {col!r} contains values not present in category order.")
        codes[col] = c
    return codes


def design_from_codes(
    codes: dict[str, np.ndarray],
    q: np.ndarray,
    orders: dict[str, list[str]],
    categorical_cols: list[str],
    *,
    include_q: bool = True,
) -> tuple[np.ndarray, list[str]]:
    parts = [np.ones(len(q), dtype=float)]
    names = ["const"]
    if include_q:
        parts.append(np.asarray(q, dtype=float))
        names.append("Q")
    for col in categorical_cols:
        cats = orders[col]
        c = codes[col]
        D = (c[:, None] == np.arange(1, len(cats), dtype=np.int16)[None, :]).astype(float)
        parts.append(D)
        names.extend([f"{col}_{cat}" for cat in cats[1:]])
    return np.column_stack(parts), names


def make_designs(df: pd.DataFrame, spec: DesignSpec):
    orders = build_category_orders(df, spec)
    codes = codes_from_orders(df, orders)
    q = df[spec.q_col].to_numpy(dtype=float)
    X_good, names_good = design_from_codes(codes, q, orders, spec.categorical_cols, include_q=True)
    X_noq, names_noq = design_from_codes(codes, q, orders, spec.categorical_cols, include_q=False)
    y = df[spec.outcome_col].to_numpy(dtype=float)
    clusters = None if spec.cluster_col is None else df[spec.cluster_col].to_numpy()
    return y, q, clusters, orders, codes, X_good, names_good, X_noq, names_noq
