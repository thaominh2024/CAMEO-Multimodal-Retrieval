"""CLI for CAMEO Approach 2: query + optional cached MagicLens image index."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.pipeline import RetrievalPipeline

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CAMEO Approach 2 retrieval CLI')
    parser.add_argument('--catalog', required=True, help='Product CSV with expected schema')
    parser.add_argument('--reference-id', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--query-model', default='qwen2.5:7b')
    parser.add_argument('--text-model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--image-dir', help='Directory containing catalog images')
    parser.add_argument('--reference-image', help='Path to query image; defaults to catalog image')
    parser.add_argument('--magiclens-repo', help='Path to trusted upstream MagicLens implementation')
    parser.add_argument('--checkpoint', help='Path to MagicLens checkpoint')
    parser.add_argument('--image-index', help='Precomputed MagicLens image index saved by this project')
    parser.add_argument('--output', help='Optional output CSV')
    return parser

def main() -> None:
    args = make_parser().parse_args()
    if bool(args.magiclens_repo) != bool(args.checkpoint):
        raise SystemExit('--magiclens-repo and --checkpoint must be provided together')
    if (args.magiclens_repo or args.image_index) and not args.image_dir:
        raise SystemExit('--image-dir is needed for image retrieval')
    products = pd.read_csv(args.catalog, dtype={'product_id':str})
    from sentence_transformers import SentenceTransformer
    text_encoder = SentenceTransformer(args.text_model)
    magic_searcher = None
    if args.magiclens_repo:
        from src.models.magiclens import MagicLensSearcher, load_magiclens
        import open_clip
        model_obj, variables = load_magiclens(args.magiclens_repo, args.checkpoint)
        magic_searcher = MagicLensSearcher(model_obj, variables,
                                          open_clip.get_tokenizer('ViT-B-32'), args.image_dir)
        if args.image_index and Path(args.image_index).exists():
            magic_searcher.load_index(args.image_index)
        else:
            magic_searcher.build_index()
            if args.image_index:
                magic_searcher.save_index(args.image_index)
    pipeline = RetrievalPipeline(products, text_encoder, magic_searcher=magic_searcher,
                                 query_model=args.query_model, alpha=args.alpha,
                                 top_k=args.top_k, image_root=args.image_dir)
    ranked = pipeline.retrieve(args.reference_id, args.question,
                               image_path=args.reference_image)
    columns = [c for c in ['product_id','product_name','price','_price_tier','text_score','magic_score','final_score'] if c in ranked]
    print(ranked[columns].to_string(index=False) if not ranked.empty else 'No matching candidates')
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True,exist_ok=True)
        ranked.to_csv(destination,index=False)
        print(f'Saved {len(ranked)} results to {destination}')

if __name__ == '__main__':
    main()
