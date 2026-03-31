#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

import splice
from splice.model import SPLICE

try:
    import numpy as np
except ImportError:  # Optional dependency for .npy input.
    np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label precomputed embeddings with sparse concept decompositions."
    )
    parser.add_argument("--embeddings-path", type=str, required=True, help="Path to a .pt/.pth/.npy file or a directory containing them.")
    parser.add_argument("--dictionary-path", type=str, required=True, help="Path to concept dictionary tensor with shape [num_concepts, dim].")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to vocab text file, one concept per line.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Destination JSONL file.")
    parser.add_argument("--mean-path", type=str, default=None, help="Optional path to precomputed embedding mean vector.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k concepts to keep per embedding.")
    parser.add_argument("--l1-penalty", type=float, default=0.25, help="L1 penalty for sparse decomposition.")
    parser.add_argument("--solver", type=str, default="skl", choices=["skl", "admm"], help="Sparse solver.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for decomposition.")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "single", "batch"], help="How to interpret tensor dimensions.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for embedding files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_vocab(vocab_path: str) -> List[str]:
    vocab: List[str] = []
    with open(vocab_path, "r") as handle:
        for line in handle:
            token = line.strip()
            if token:
                vocab.append(token)
    if not vocab:
        raise ValueError(f"Vocabulary file is empty: {vocab_path}")
    return vocab


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
    raise ValueError(f"Unsupported file suffix: {path}")


def list_embedding_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Embedding path not found: {path}")

    valid_suffixes = {".pt", ".pth", ".npy"}
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
        raise ValueError(f"No supported embedding files found under: {path}")
    return files


def _pool_to_vector(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor
    flat = tensor.reshape(-1, tensor.shape[-1])
    return flat.mean(dim=0)


def _yield_from_tensor(
    tensor: torch.Tensor,
    base_id: str,
    layout: str,
    ids: Optional[List[str]] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    if tensor.ndim == 0:
        raise ValueError(f"Scalar tensor is not a valid embedding: {base_id}")

    if layout == "single":
        yield base_id, _pool_to_vector(tensor)
        return

    if tensor.ndim == 1:
        yield base_id, tensor
        return

    batch_size = tensor.shape[0]
    for idx in range(batch_size):
        sample_id = ids[idx] if ids is not None and idx < len(ids) else f"{base_id}:{idx}"
        yield sample_id, _pool_to_vector(tensor[idx])


def _extract_ids(obj: Dict[str, Any]) -> Optional[List[str]]:
    for key in ["ids", "doc_ids", "query_ids", "item_ids"]:
        if key in obj and isinstance(obj[key], (list, tuple)):
            return [str(x) for x in obj[key]]
    return None


def parse_embedding_object(
    obj: Any,
    base_id: str,
    layout: str,
) -> Iterator[Tuple[str, torch.Tensor]]:
    if isinstance(obj, (torch.Tensor,)) or (np is not None and isinstance(obj, np.ndarray)):
        yield from _yield_from_tensor(to_tensor(obj), base_id, layout)
        return

    if isinstance(obj, dict):
        if "embedding" in obj:
            item_id = str(obj.get("id", obj.get("doc_id", obj.get("query_id", base_id))))
            yield item_id, _pool_to_vector(to_tensor(obj["embedding"]))
            return

        if "embeddings" in obj:
            ids = _extract_ids(obj)
            yield from _yield_from_tensor(to_tensor(obj["embeddings"]), base_id, layout, ids)
            return

        emitted = False
        for key, value in obj.items():
            if isinstance(value, torch.Tensor) or (np is not None and isinstance(value, np.ndarray)):
                emitted = True
                yield str(key), _pool_to_vector(to_tensor(value))
        if emitted:
            return

        raise TypeError(f"Could not parse dict embeddings for: {base_id}. Keys={list(obj.keys())[:10]}")

    if isinstance(obj, (list, tuple)):
        if not obj:
            return
        for idx, value in enumerate(obj):
            if isinstance(value, torch.Tensor) or (np is not None and isinstance(value, np.ndarray)):
                yield f"{base_id}:{idx}", _pool_to_vector(to_tensor(value))
            else:
                raise TypeError(f"Unsupported list element type in {base_id}: {type(value)}")
        return

    raise TypeError(f"Unsupported embedding object type at {base_id}: {type(obj)}")


def infer_layout(layout: str, embeddings_path: Path) -> str:
    if layout != "auto":
        return layout
    if embeddings_path.is_dir():
        return "single"
    return "batch"


def iter_embeddings(
    embeddings_path: Path,
    recursive: bool,
    layout: str,
) -> Iterator[Tuple[str, torch.Tensor]]:
    files = list_embedding_files(embeddings_path, recursive)
    for file_path in files:
        obj = load_object(file_path)
        base_id = file_path.stem
        yield from parse_embedding_object(obj, base_id=base_id, layout=layout)


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {tuple(vec.shape)}")
    norm = torch.linalg.norm(vec)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        raise ValueError("Embedding vector has invalid norm.")
    return vec / norm


def compute_mean(
    embeddings_path: Path,
    recursive: bool,
    layout: str,
) -> torch.Tensor:
    running_sum: Optional[torch.Tensor] = None
    count = 0
    for _, vec in iter_embeddings(embeddings_path, recursive, layout):
        vec = normalize_vector(vec)
        if running_sum is None:
            running_sum = vec.clone()
        else:
            if running_sum.shape[0] != vec.shape[0]:
                raise ValueError(f"Mixed embedding dimensions found: {running_sum.shape[0]} and {vec.shape[0]}")
            running_sum += vec
        count += 1

    if running_sum is None or count == 0:
        raise ValueError("No embeddings found to compute mean.")
    mean = running_sum / count
    return F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)


def load_dictionary(dictionary_path: str, expected_vocab_size: int) -> torch.Tensor:
    dictionary = to_tensor(load_object(Path(dictionary_path)))
    if dictionary.ndim != 2:
        raise ValueError(f"Dictionary must have shape [num_concepts, dim], got {tuple(dictionary.shape)}")
    if dictionary.shape[0] != expected_vocab_size:
        raise ValueError(
            f"Dictionary rows ({dictionary.shape[0]}) must match vocab size ({expected_vocab_size})."
        )
    dictionary = F.normalize(dictionary, dim=1)
    dictionary = F.normalize(dictionary - dictionary.mean(dim=0, keepdim=True), dim=1)
    return dictionary


def _format_concepts(
    weights: torch.Tensor,
    vocab: List[str],
    topk: int,
) -> List[Dict[str, Any]]:
    topk = min(topk, weights.shape[0])
    vals, inds = torch.topk(weights, k=topk, largest=True)
    concepts: List[Dict[str, Any]] = []
    for value, idx in zip(vals.tolist(), inds.tolist()):
        if value <= 0:
            continue
        concepts.append({"concept": vocab[idx], "weight": round(float(value), 6)})
    return concepts


def flush_batch(
    model: SPLICE,
    batch_ids: List[str],
    batch_vectors: List[torch.Tensor],
    vocab: List[str],
    topk: int,
    out_handle: Any,
) -> None:
    x = torch.stack(batch_vectors).to(model.device)
    with torch.no_grad():
        weights = model.encode_image(x)
        recon = model.recompose_image(weights)

    x_norm = F.normalize(x, dim=1)
    cosine = torch.sum(recon * x_norm, dim=1)
    l0 = torch.linalg.vector_norm(weights, ord=0, dim=1)

    for row_idx, item_id in enumerate(batch_ids):
        row = weights[row_idx].detach().cpu()
        record = {
            "id": item_id,
            "top_concepts": _format_concepts(row, vocab, topk),
            "l0_norm": float(l0[row_idx].item()),
            "cosine_similarity": float(cosine[row_idx].item()),
        }
        out_handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()

    embeddings_path = Path(args.embeddings_path)
    layout = infer_layout(args.layout, embeddings_path)
    vocab = load_vocab(args.vocab_path)
    dictionary = load_dictionary(args.dictionary_path, expected_vocab_size=len(vocab))

    if args.mean_path is None:
        print("No --mean-path provided. Computing mean from embeddings...", flush=True)
        image_mean = compute_mean(embeddings_path, args.recursive, layout)
    else:
        image_mean = to_tensor(load_object(Path(args.mean_path)))
        image_mean = _pool_to_vector(image_mean)
        image_mean = normalize_vector(image_mean)

    if image_mean.shape[0] != dictionary.shape[1]:
        raise ValueError(
            f"Embedding mean dim ({image_mean.shape[0]}) does not match dictionary dim ({dictionary.shape[1]})."
        )

    model = SPLICE(
        image_mean=image_mean,
        dictionary=dictionary,
        clip=None,
        solver=args.solver,
        l1_penalty=args.l1_penalty,
        return_weights=True,
        return_cosine=False,
        device=args.device,
    )
    model.eval()

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    batch_ids: List[str] = []
    batch_vectors: List[torch.Tensor] = []
    total = 0

    with open(args.output_jsonl, "w") as out_handle:
        for item_id, vec in iter_embeddings(embeddings_path, args.recursive, layout):
            vec = normalize_vector(vec)
            if vec.shape[0] != dictionary.shape[1]:
                raise ValueError(
                    f"Item {item_id} has dim {vec.shape[0]}, expected {dictionary.shape[1]}."
                )

            batch_ids.append(item_id)
            batch_vectors.append(vec)

            if len(batch_ids) >= args.batch_size:
                flush_batch(model, batch_ids, batch_vectors, vocab, args.topk, out_handle)
                total += len(batch_ids)
                print(f"Labeled {total} embeddings...", flush=True)
                batch_ids, batch_vectors = [], []

        if batch_ids:
            flush_batch(model, batch_ids, batch_vectors, vocab, args.topk, out_handle)
            total += len(batch_ids)

    print(f"Done. Wrote labels for {total} embeddings to: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
