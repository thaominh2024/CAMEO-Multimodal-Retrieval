"""Hybrid MagicLens + Filtering scoring"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _clean(value: object) -> str:
    if value is None or isinstance(value,float) and np.isnan(value):
        return ''
    return str(value).replace('[','').replace(']','').replace("'",'')


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Match notebook normalization (constant scores stay unchanged)."""
    if len(scores) == 0:
        return scores
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min())
    return scores


def rank_candidates(candidates: pd.DataFrame, parsed: dict, text_model,
                    magic_searcher=None, ref_image_path: str | None = None,
                    alpha: float = .6, top_k: int = 15) -> pd.DataFrame:
    """Rank candidates; price tier priority precedes fusion score.

    ``alpha`` weights visual similarity when a MagicLens index is supplied.
    Without it, text similarity alone determines ranking.
    """
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0 and 1')
    if top_k < 1:
        raise ValueError('top_k must be positive')
    result = candidates.copy()
    if result.empty:
        return result
    required = {'product_name','type','object','ingredient','purpose'}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f'Missing text columns: {sorted(missing)}')
    query_text = ' . '.join(filter(None, [
        _clean(parsed.get('question')),
        *[_clean(parsed.get(key)) for key in ('object','purpose','target_ingredient','type')]
    ]))
    texts = (result['product_name'].fillna('').astype(str) + ' . ' +
             result['type'].fillna('').astype(str) + ' . ' +
             result['object'].fillna('').astype(str) + ' . ' +
             result['ingredient'].fillna('').astype(str) + ' . ' +
             result['purpose'].fillna('').astype(str)).tolist()
    from sklearn.metrics.pairwise import cosine_similarity
    raw_text = cosine_similarity(text_model.encode([query_text]),text_model.encode(texts))[0]
    result['text_score'] = _minmax(np.asarray(raw_text,dtype=float))
    if magic_searcher is not None and ref_image_path and Path(ref_image_path).is_file():
        if 'image_path' not in result:
            raise ValueError('image_path is required for MagicLens scoring')
        names = result['image_path'].map(lambda p: Path(str(p)).name).tolist()
        raw_map = magic_searcher.score_candidates(ref_image_path,str(parsed.get('question') or ''),names)
        # Notebook assigned 0 for candidates missing from the MagicLens index.
        raw_magic = np.asarray([raw_map.get(n) if raw_map.get(n) is not None else 0.0 for n in names],dtype=float)
        result['magic_score'] = _minmax(raw_magic)
        result['final_score'] = alpha*result['magic_score'] + (1-alpha)*result['text_score']
    else:
        result['final_score'] = result['text_score']
    sort_cols = ['final_score']
    ascending = [False]
    if '_price_tier' in result.columns:
        sort_cols.insert(0,'_price_tier')
        ascending.insert(0,True)
    return result.sort_values(sort_cols,ascending=ascending,kind='stable').head(top_k)
