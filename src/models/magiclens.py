"""MagicLens inference and image-index helpers, adapted from magiclens.ipynb.

The pretrained MagicLens implementation belongs to its original authors.
This wrapper does not include their model code or weights.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_magiclens(magiclens_repo: str | Path, checkpoint: str | Path, model_size: str = "base") -> tuple[Any, Any]:
    """Load pretrained MagicLens through its original inference.py API.

    The original Kaggle notebook imports MagicLens' model and inference modules.
    Run only with a trusted checkout of the upstream repository.
    """
    repo = Path(magiclens_repo).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (repo / "inference.py").is_file():
        raise FileNotFoundError(f"MagicLens inference.py not found: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from inference import load_model  # type: ignore[import-not-found]

    model_obj, params = load_model(model_size, str(checkpoint))
    variables = params if "params" in params else {"params": params}
    return model_obj, variables


class MagicLensSearcher:
    """Encode images and rank candidates with MagicLens embeddings.

    Expected model behavior matches the original notebook:
    model.apply(variables, {"ids": [1,1,77], "image": [1,224,224,3]})
    -> {"multimodal_embed_norm": array of shape [1,d]}.
    """

    def __init__(self, model_obj: Any, variables: Any, tokenizer: Any, image_folder: str | Path):
        self.model = model_obj
        self.variables = variables
        self.tokenizer = tokenizer
        self.image_folder = Path(image_folder).expanduser()
        self.all_vectors: np.ndarray | None = None
        self.image_paths: list[str] = []
        self._path_indices: dict[str, int] = {}

    def encode(self, image: Image.Image, text: str = "") -> np.ndarray:
        import jax.numpy as jnp

        image_array = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        tokens = np.asarray(self.tokenizer(text))
        if tokens.size != 77:
            raise ValueError(f"Tokenizer expected 77 tokens, got {tokens.size}")
        batch = {
            "ids": jnp.asarray(tokens.reshape(1, 1, 77)),
            "image": jnp.asarray(image_array.reshape(1, 224, 224, 3)),
        }
        output = self.model.apply(self.variables, batch)["multimodal_embed_norm"]
        return np.asarray(output).reshape(-1).astype(np.float32)

    def _refresh_lookup(self) -> None:
        self._path_indices = {Path(p).name: i for i, p in enumerate(self.image_paths)}

    def build_index(self, limit: int | None = None, append: bool = False) -> int:
        if not self.image_folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
        image_files = sorted(p for p in self.image_folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if append:
            processed = set(self.image_paths)
            image_files = [p for p in image_files if str(p) not in processed]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be nonnegative")
            image_files = image_files[:limit]

        vectors, paths = [], []
        for path in image_files:
            try:
                with Image.open(path) as image:
                    vec = self.encode(image)
            except (OSError, ValueError) as exc:
                print(f"Skipping {path.name}: {exc}")
                continue
            vectors.append(vec)
            paths.append(str(path))
