#!/usr/bin/env python3
"""Cache frozen 768-D SigLIP text features for all OVON categories."""

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from open_clip import create_model_from_pretrained, get_tokenizer


MODEL_NAME = "hf-hub:timm/ViT-B-16-SigLIP-256"


def collect_categories(dataset_root: Path):
    categories = set()
    for path in sorted(dataset_root.rglob("*.json.gz")):
        if path.name.startswith("._"):
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        for episode in payload.get("episodes", []):
            category = episode.get("object_category")
            if category:
                categories.add(category)
        for goals in payload.get("goals_by_category", {}).values():
            for goal in goals:
                category = goal.get("object_category")
                if category:
                    categories.add(category)
    if not categories:
        raise RuntimeError(f"No object categories found below {dataset_root}")
    return sorted(categories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    categories = collect_categories(args.dataset_root)
    model, _ = create_model_from_pretrained(MODEL_NAME)
    model = model.eval().to(args.device)
    tokenizer = get_tokenizer(MODEL_NAME)
    cache = {}
    with torch.inference_mode():
        for start in range(0, len(categories), args.batch_size):
            names = categories[start : start + args.batch_size]
            tokens = tokenizer(names, context_length=64).to(args.device)
            features = model.encode_text(tokens).float().cpu().numpy()
            if features.shape[1] != 768:
                raise RuntimeError(f"Expected 768-D SigLIP text features, got {features.shape}")
            cache.update(
                (name, np.asarray(feature, dtype=np.float32))
                for name, feature in zip(names, features)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(cache, handle)
    print(f"Saved {len(cache)} categories to {args.output}")


if __name__ == "__main__":
    main()
