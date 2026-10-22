from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from src.extraction.query_parser import llm_query_parser
from src.retrieval.filtering import filter_candidates, normalize_id
from src.retrieval.ranking import rank_candidates

class RetrievalPipeline:
    def __init__(self, products: pd.DataFrame, text_encoder: Any, *,
                 magic_searcher: Any = None, query_client: Any = None,
                 query_model: str = "qwen2.5:7b", alpha: float = 0.6,
                 top_k: int = 10, image_root: str | Path | None = None):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0, 1]')
        self.products = products.copy()
        self.text_encoder = text_encoder
        self.magic_searcher = magic_searcher
        self.query_client = query_client
        self.query_model = query_model
        self.alpha = alpha
        self.top_k = top_k
        self.image_root = Path(image_root) if image_root else None

    def retrieve(self, reference_id: str, question: str, *,
                 parsed_query: dict | None = None, image_path: str | None = None) -> pd.DataFrame:
        matches = self.products.loc[self.products['product_id'].map(normalize_id) == normalize_id(reference_id)]
        if matches.empty:
            raise ValueError(f'Unknown reference product: {reference_id}')
        reference = matches.iloc[0]
        if parsed_query is None:
            parsed_query = llm_query_parser(self.products, question, reference.to_dict(),
                                            model=self.query_model, client=self.query_client)
        parsed = dict(parsed_query)
        parsed['question'] = question
        filtered = filter_candidates(self.products, reference_id, parsed)
        if filtered.empty:
            return filtered
        if image_path is None and self.image_root is not None and pd.notna(reference.get('image_path')):
            image_path = str(self.image_root / str(reference['image_path']))
        return rank_candidates(filtered, parsed, self.text_encoder,
                               magic_searcher=self.magic_searcher,
                               ref_image_path=image_path, alpha=self.alpha, top_k=self.top_k)
