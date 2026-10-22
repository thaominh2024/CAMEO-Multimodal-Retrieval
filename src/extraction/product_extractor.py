import argparse
import json
import re
from pathlib import Path
import pandas as pd

FIELDS = ("ingredient", "capacity", "type")

PROMPT = """You are an expert e-commerce data extraction AI. Read the PRODUCT NAME and output a JSON object.

RULES:
1. NO HALLUCINATIONS. Extract exact words present in the text.
2. If information is missing, output an empty string "".

KEY DEFINITIONS:
- "ingredient": The main active ingredients, natural extracts, plants, herbs, or special materials (e.g., Aloe Vera, Centella, Mint, Red Ginseng, Rice milk, Snail, Tea Tree). DO NOT extract adjectives (Gentle, Purifying), product formats (Gel, Foam), or skin issues (Acne, Oily, Dirt).
- "capacity": Volume or weight containing numbers and units (e.g., 150ml, 591 ml, 50ml, 130ml, 80g).
- "type": The specific format of the product (e.g., Facial Cleanser, Foam Cleanser, Gel Cleanser, Face Wash).

--- EXAMPLES ---
Text: "Civic Rice milk Rice Facial Cleanser"
JSON: {{"ingredient": "Rice milk", "capacity": "", "type": "Facial Cleanser"}}

Text: "Centella and Mint Extract Facial Cleanser"
JSON: {{"ingredient": "Centella, Mint", "capacity": "", "type": "Facial Cleanser"}}

Text: "Cetaphil Facial Cleanser 591 ml (Guaranteed 100%)"
JSON: {{"ingredient": "", "capacity": "591 ml", "type": "Facial Cleanser"}}
--- END OF EXAMPLES ---

PRODUCT NAME: {text}
JSON:
"""


def load_model(repo_id: str, filename: str, n_ctx: int):
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        verbose=False,
        n_ctx=n_ctx,
        repeat_penalty=1.2,
    )


def parse_model_output(content: str) -> dict[str, str]:
    match = re.search(r"\{.*?\}", content or "", flags=re.DOTALL)
    if not match:
        return {key: "N/A" for key in FIELDS}
    try:
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Model output must be a JSON object")
        return {key: str(payload.get(key) or "") for key in FIELDS}
    except (ValueError, TypeError):
        return {key: "N/A" for key in FIELDS}

def extract_product_info(product_name: str, model) -> dict[str, str]:
    response = model.create_chat_completion(
        messages=[{"role": "user", "content": PROMPT.format(text=json.dumps(str(product_name), ensure_ascii=False))}],
        temperature=0.01,
    )
    raw = response["choices"][0]["message"]["content"]
    return parse_model_output(raw)


def process_csv(input_path: str, output_path: str, start: int, limit: int | None,
                chunk_size: int, model) -> int:
    df = pd.read_csv(input_path).drop_duplicates().reset_index(drop=True)
    if "product_name" not in df.columns:
        raise ValueError("Input CSV must have a 'product_name' column")
    if start < 0 or chunk_size < 1 or (limit is not None and limit < 1):
        raise ValueError("start >= 0, chunk_size >= 1, and limit >= 1 required")
    stop = min(len(df), start + limit) if limit is not None else len(df)
    if start >= stop:
        raise ValueError(f"No rows selected: start={start}, rows={len(df)}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite on each execution, preventing duplicate output rows on reruns.
    for chunk_start in range(start, stop, chunk_size):
        chunk = df.iloc[chunk_start:min(chunk_start + chunk_size, stop)].copy()
        results = [extract_product_info(name, model) for name in chunk["product_name"].fillna("")]
        extracted = pd.DataFrame(results, index=chunk.index)
        for key in FIELDS:
            chunk[key] = extracted[key]
        chunk.to_csv(
            destination,
            mode="w" if chunk_start == start else "a",
            header=chunk_start == start,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Extracted {min(chunk_start + chunk_size, stop)-start}/{stop-start} products")
    return stop - start


def main():
    parser = argparse.ArgumentParser(description="Qwen product-name metadata extractor")
    parser.add_argument("--input", required=True, help="Input CSV (requires product_name column)")
    parser.add_argument("--output", default="data/processed/extracted_products.csv")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-3B-Instruct-GGUF")
    parser.add_argument("--model-file", default="*q4_k_m.gguf")
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()
    model = load_model(args.repo_id, args.model_file, args.n_ctx)
    count = process_csv(args.input, args.output, args.start, args.limit, args.chunk_size, model)
    print(f"Done. {count} rows written to {args.output}")


if __name__ == "__main__":
    main()
