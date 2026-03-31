#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import numpy as np
except ImportError:
    np = None

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a concept dictionary tensor from concept embedding files."
    )
    parser.add_argument("--embeddings-path", type=str, required=True, help="Directory or file with concept embeddings.")
    parser.add_argument("--output-dictionary-pt", type=str, required=True)
    parser.add_argument("--output-vocab-txt", type=str, required=True)
    parser.add_argument("--concept-jsonl", type=str, default=None, help="Optional query JSONL used to produce embeddings (query_id/query_text).")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--normalize", action="store_true", help="L2-normalize vectors before saving.")
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    return parser.parse_args()


def to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if np is not None and isinstance(value, np.ndarray):
        return torch.from_numpy(value).float().cpu()
    raise TypeError(f"Unsupported embedding type: {type(value)}")


def load_object(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    if suffix == ".npy":
        if np is None:
            raise ImportError("numpy is required to load .npy files.")
        return np.load(path, allow_pickle=True)
    if suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError("safetensors is required for .safetensors files. Install with: pip install safetensors")
        return load_safetensors_file(str(path), device="cpu")
    raise ValueError(f"Unsupported file suffix: {path}")


def list_embedding_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    valid_suffixes = {".pt", ".pth", ".npy", ".safetensors"}
    files: List[Path] = []
    if recursive:
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in valid_suffixes:
                    files.append(file_path)
    else:
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in valid_suffixes:
                files.append(file_path)

    files.sort()
    if not files:
        raise ValueError(f"No embedding files found under: {path}")
    return files


def pool_to_vector(tensor: torch.Tensor, pooling: str) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor
    flat = tensor.reshape(-1, tensor.shape[-1])
    if pooling == "mean":
        return flat.mean(dim=0)
    return flat.max(dim=0).values


def parse_object_to_vector(obj: Any, pooling: str) -> torch.Tensor:
    if isinstance(obj, torch.Tensor) or (np is not None and isinstance(obj, np.ndarray)):
        return pool_to_vector(to_tensor(obj), pooling)

    if isinstance(obj, dict):
        if "embeddings" in obj:
            return pool_to_vector(to_tensor(obj["embeddings"]), pooling)
        if "embedding" in obj:
            return pool_to_vector(to_tensor(obj["embedding"]), pooling)

        tensor_items = [
            to_tensor(v)
            for v in obj.values()
            if isinstance(v, torch.Tensor) or (np is not None and isinstance(v, np.ndarray))
        ]
        if len(tensor_items) == 1:
            return pool_to_vector(tensor_items[0], pooling)
        if len(tensor_items) > 1:
            pooled = [pool_to_vector(t, pooling) for t in tensor_items]
            return torch.stack(pooled, dim=0).mean(dim=0)

    raise TypeError(f"Could not parse embedding object of type {type(obj)}")


def load_concept_order(concept_jsonl: str) -> List[Tuple[str, str]]:
    ordered: List[Tuple[str, str]] = []
    with open(concept_jsonl, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            query_text = str(row["query_text"])
            ordered.append((query_id, query_text))
    return ordered


def main() -> None:
    args = parse_args()
    emb_path = Path(args.embeddings_path)
    files = list_embedding_files(emb_path, args.recursive)
    stem_to_file = {p.stem: p for p in files}

    ordered_items: List[Tuple[str, str, Path]] = []
    missing_ids: List[str] = []

    if args.concept_jsonl is not None:
        for concept_id, concept_text in load_concept_order(args.concept_jsonl):
            file_path = stem_to_file.get(concept_id)
            if file_path is None:
                missing_ids.append(concept_id)
                continue
            ordered_items.append((concept_id, concept_text, file_path))
    else:
        for file_path in files:
            ordered_items.append((file_path.stem, file_path.stem, file_path))

    if not ordered_items:
        raise ValueError("No concept embeddings found to build dictionary.")

    vectors: List[torch.Tensor] = []
    concepts: List[str] = []

    for concept_id, concept_text, file_path in ordered_items:
        obj = load_object(file_path)
        vec = parse_object_to_vector(obj, pooling=args.pooling).float()
        if vec.ndim != 1:
            raise ValueError(f"Concept {concept_id} produced non-vector shape: {tuple(vec.shape)}")
        if args.expected_dim is not None and vec.shape[0] != args.expected_dim:
            raise ValueError(
                f"Concept {concept_id} has dim {vec.shape[0]}, expected {args.expected_dim}"
            )
        vectors.append(vec)
        concepts.append(concept_text)

    dictionary = torch.stack(vectors, dim=0)
    if args.normalize:
        dictionary = F.normalize(dictionary, dim=1)

    out_dict = Path(args.output_dictionary_pt)
    out_vocab = Path(args.output_vocab_txt)
    out_dict.parent.mkdir(parents=True, exist_ok=True)
    out_vocab.parent.mkdir(parents=True, exist_ok=True)

    torch.save(dictionary, out_dict)
    with open(out_vocab, "w") as handle:
        for concept in concepts:
            handle.write(concept + "\n")

    print(f"Dictionary shape: {tuple(dictionary.shape)}")
    print(f"Concepts written: {len(concepts)}")
    print(f"Dictionary saved: {out_dict}")
    print(f"Vocabulary saved: {out_vocab}")
    if missing_ids:
        print(f"Missing concept embedding files: {len(missing_ids)}")
        print(f"First 20 missing IDs: {missing_ids[:20]}")

    if args.summary_json is not None:
        summary = {
            "dictionary_shape": list(dictionary.shape),
            "num_concepts": len(concepts),
            "embedding_dim": int(dictionary.shape[1]),
            "missing_ids": missing_ids,
        }
        out_summary = Path(args.summary_json)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(out_summary, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary saved: {out_summary}")


if __name__ == "__main__":
    main()
