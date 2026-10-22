# CAMEO-Multimodal-Retrieval
Multimodal cosmetic product retrieval using MagicLens, LLM-based query understanding, and dynamic attribute filtering.

## Overview
In interactive e-commerce search, users frequently formulate queries relative to an existing reference item (e.g., "keep the brand and bottle style, but change to the products which are suitable for sensitive skin and made from natural ingredients like avocado").

Query Input
Reference Product: Image alongside catalog metadata (brand, origin, type, ingredient, purpose, object, price, capacity).
Modification Text: Natural language instructions specifying desired attribute modifications, replacements, or additions.
Output
A ranked list of candidate products from a gallery catalog of 4,000+ items.

## Repository Structure

```text
CAMEO-Multimodal-Retrieval/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── src/
│   ├── extraction/
│   │   ├── product_extractor.py
│   │   └── query_parser.py
│   ├── models/
│   │   ├── magiclens.py
│   │   └── text_encoder.py
│   ├── retrieval/
│   │   ├── filtering.py
│   │   └── ranking.py
│   └── pipeline.py
├── notebooks/
│   ├── extract.ipynb
│   ├── cameo-easybutokay.ipynb
│   └── magiclens.ipynb
├── data/
│   └── README.md
└── results/
    └── metrics.csv
```
