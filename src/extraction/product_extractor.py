

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
