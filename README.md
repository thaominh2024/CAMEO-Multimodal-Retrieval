# CAMEO-Multimodal-Retrieval
Multimodal cosmetic product retrieval using MagicLens, LLM-based query understanding, and dynamic attribute filtering.

## Overview
In interactive e-commerce search, users frequently formulate queries relative to an existing reference item (e.g., "keep the brand and bottle style, but change to the products which are suitable for sensitive skin and made from natural ingredients like avocado").

Query Input
Reference Product: Image alongside catalog metadata (brand, origin, type, ingredient, purpose, object, price, capacity).
Modification Text: Natural language instructions specifying desired attribute modifications, replacements, or additions.
Output
A ranked list of candidate products from a gallery catalog of 4,000+ items.

## Dataset

The benchmark data is provided by the **CAMEO: Composed Access to Multimodal E-commerce Objects** task from **NTCIR-19**. Each catalog item consists of an image, product name and product descriptions.

| Field                   | Content                                                                                                                                                                                                                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Product Name**        | *Simple Gentle & Refreshing Facial Cleanser - for sensitive skin 150ml*                                                                                                                                                                                                                                                                    |
| **Product Description** | *Simple Facial Cleanser is gentle, cleanses, and refreshes for sensitive skin. It contains x2 Vitamin B5, Vitamin E, and Pro Amino Acids, which effectively cleanse the skin, sweeping away oil, dirt, and other impurities without causing skin irritation. It leaves skin soft and smooth, while providing a fresh and clean feeling...* |

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
│   ├── retrieval/
│   │   ├── filtering.py
│   │   └── ranking.py
│   └── pipeline.py
└── 
```
