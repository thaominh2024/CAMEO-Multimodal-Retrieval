"""Candidate filtering"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def normalize_id(value: object) -> str:
    return str(value).split('.')[0].strip()


def capacity_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r'(\d+(?:\.\d+)?)\s*(ml|g|gram)', str(value).lower())
    return float(match.group(1)) if match else None


def price_bounds(parsed: dict, reference_price: float) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    raw = parsed.get('price_range')
    nums = [] if raw is None else [int(x) for x in re.findall(r'\d+', str(raw))]
    action = str(parsed.get('price_action') or '').upper()
    question = str(parsed.get('question') or '').lower()
    if len(nums) == 1:
        val = nums[0]
        if action == 'CHEAPER' or 'under' in question or 'less' in question:
            return (0, val), (0, int(val*1.2))
        return (int(val*.9), int(val*1.1)), (int(val*.8), int(val*1.2))
    if len(nums) >= 2:
        lo, hi = min(nums), max(nums)
        return (lo, hi), (int(lo*.8), int(hi*1.2))
    if action == 'CHEAPER':
        return (0, max(0, int(reference_price)-1)), (0, max(0, int(reference_price*1.2)))
    if action == 'EXPENSIVE':
        return (int(reference_price)+1, float('inf')), (int(reference_price*.8), float('inf'))
    return None, None


def filter_candidates(database: pd.DataFrame, ref_product_id: object, parsed: dict) -> pd.DataFrame:
    """Return candidates with a `_price_tier` of 0 (strict) or 1 (relaxed).

    Preserves the notebook's two-band approach. A separate strict-only
    policy can be implemented once benchmark behavior is verified.
    """
    required = {'product_id','brand','origin','price','capacity'}
    missing = required.difference(database.columns)
    if missing:
        raise ValueError(f'Missing catalog columns: {sorted(missing)}')
    ref_id = normalize_id(ref_product_id)
    ids = database['product_id'].map(normalize_id)
    matches = database.loc[ids == ref_id]
    if matches.empty:
        raise ValueError(f'Reference product not found: {ref_id}')
    ref = matches.iloc[0]
    candidates = database.loc[ids != ref_id].copy()
    brands = parsed.get('brand')
    if isinstance(brands, str):
        brands = brands.split(',')
    if isinstance(brands, (list,tuple,np.ndarray)) and len(brands):
        def clean(v): return re.sub(r"[’’'\"\[\]]",'',str(v).lower()).strip()
        allowed = {clean(v) for v in brands if str(v).strip()}
        candidates = candidates.loc[candidates['brand'].map(clean).isin(allowed)]
    origin = parsed.get('origin')
    if isinstance(origin,str) and origin.strip() and origin.lower() not in {'nan','null'}:
        candidates = candidates.loc[candidates['origin'].astype(str).str.lower() == origin.lower()]
    if str(parsed.get('capacity_action') or '').upper() == 'LARGER':
        reference_capacity = capacity_value(ref.get('capacity'))
        if reference_capacity is not None:
            candidates = candidates.loc[candidates['capacity'].map(capacity_value).map(lambda v: v is not None and v > reference_capacity)]
    strict, relaxed = price_bounds(parsed, float(ref['price']))
    candidates['_price_tier'] = 0
    if strict is None:
        return candidates
    prices = pd.to_numeric(candidates['price'], errors='coerce')
    strict_mask = prices.between(*strict)
    relaxed_mask = prices.between(*relaxed) & ~strict_mask
    result = candidates.loc[strict_mask | relaxed_mask].copy()
    result.loc[relaxed_mask.loc[result.index], '_price_tier'] = 1
    return result.drop_duplicates(subset='product_id')
