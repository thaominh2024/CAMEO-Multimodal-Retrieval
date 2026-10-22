import json
import re
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


class CosmeticQuerySchema(BaseModel):
    brand: Optional[List[str]] = Field(default=None, description="Brands explicitly requested")
    origin: Optional[str] = Field(default=None, description="Country of origin, if mentioned")
    price_action: Literal["CHEAPER", "EXPENSIVE", "NONE"] = "NONE"
    price_range: Optional[List[int]] = None
    target_ingredient: Optional[str] = None
    purpose: Optional[str] = None
    object: Optional[str] = None
    capacity_action: Optional[Literal["Larger", "Smaller"]] = None


def _normalize_text(value: object) -> str:
    return re.sub(r"[^\w\s]", "", str(value).lower())


def _mentioned_values(database: pd.DataFrame, column: str, query: str) -> list[str]:
    """Find catalog values occurring as complete words within query."""
    if column not in database.columns:
        raise ValueError(f"Catalog is missing required column: {column}")
    normalized_query = _normalize_text(query)
    matches = []
    for item in database[column].dropna().unique():
        normalized = _normalize_text(item)
        if len(normalized) > 1 and re.search(rf"\b{re.escape(normalized)}\b", normalized_query):
            matches.append(str(item))
    return list(dict.fromkeys(matches))


def llm_query_parser(
    df_database: pd.DataFrame,
    query_en: str,
    ref_product_info: dict,
    *,
    model: str = "qwen2.5:7b",
    client=None,
) -> dict:
    """Extract query constraints using Ollama, matching notebook behavior.

    The Ollama import is delayed to allow mock-based tests without the server.
    ``object`` is augmented with 'all skin types' as in the original notebook.
    """
    if client is None:
        import ollama
        client = ollama

    filtered_brands = _mentioned_values(df_database, "brand", query_en)
    filtered_origins = _mentioned_values(df_database, "origin", query_en)

    system_prompt = f"""You are a strict NLP parser for a cosmetic store.
Analyze the user query based on the reference product data.
You MUST extract the information into the exact JSON schema provided.

CRITICAL RULES:
1. "brand": Match with Allowed Brands. Return list of strings, or null.
2. "origin": Extract country of origin if mentioned. Else null.
3. "price_action": "CHEAPER" / "EXPENSIVE" / "NONE".
4. "price_range": Convert 'around 30k' to [30000], '200k-300k' to [200000, 300000]. Else null.
5. "target_ingredient": Extract requested ingredient/material. Else null.
6. "purpose": Extract keyword for benefit or requirement. Else null.
7. "object": Extract skin type or target user. Else null.
8. "capacity_action": "Larger" / "Smaller" if requested, else null.

ALLOWED BRANDS: {filtered_brands}
ALLOWED ORIGIN: {filtered_origins}"""
    user_content = (
        "[ACTUAL DATA TO PARSE]:\n"
        f"- Ref Product: {json.dumps({k: ref_product_info.get(k) for k in ('price','brand','ingredient')}, ensure_ascii=False)}\n"
        f"- Query: {json.dumps(query_en, ensure_ascii=False)}"
    )
    response = client.chat(
        model=model,
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}],
        format=CosmeticQuerySchema.model_json_schema(),
        options={'temperature': 0.0, 'num_predict': 150},
    )
    # Ollama may return a dict or a typed response object.
    message = response['message'] if isinstance(response, dict) else response.message
    content = message['content'] if isinstance(message, dict) else message.content
    try:
        raw = json.loads(content.strip())
        if not isinstance(raw, dict):
            raise ValueError('Model output is not a JSON object')
        parsed = CosmeticQuerySchema.model_validate(raw).model_dump()
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid Ollama query-parser response: {exc}") from exc

    # Preserve notebook post-processing for brands and skin types.
    if not parsed['brand']:
        parsed['brand'] = filtered_brands or None
    if parsed['object'] not in (None, ""):
        parsed['object'] = [parsed['object'], 'all skin types']
    return parsed


def parse_question_rows(
    questions: pd.DataFrame,
    products: pd.DataFrame,
    *,
    model: str = "qwen2.5:7b",
    client=None,
) -> pd.DataFrame:
    """Parse question/product_id rows and return original fields + constraints."""
    for column in ('question', 'product_id'):
        if column not in questions:
            raise ValueError(f"Questions missing required column: {column}")
    if 'product_id' not in products:
        raise ValueError('Products missing required column: product_id')
    rows = []
    for _, row in questions.iterrows():
        reference = products.loc[products['product_id'] == row['product_id']]
        ref_info = reference.iloc[0].to_dict() if not reference.empty else {}
        rows.append(llm_query_parser(products, str(row['question']), ref_info, model=model, client=client))
    return pd.concat([questions.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
