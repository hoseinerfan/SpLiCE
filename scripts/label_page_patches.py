#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from splice.model import SPLICE

try:
    import numpy as np
except ImportError:  # Optional dependency for .npy inputs.
    np = None

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # Optional dependency for .safetensors inputs.
    load_safetensors_file = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patch-level concept labels for a single page."
    )
    parser.add_argument("--embedding-file", type=str, required=True, help="Path to a single embedding file (.safetensors/.pt/.pth/.npy).")
    parser.add_argument("--page-index", type=int, default=0, help="Page index to inspect when file contains multiple pages.")
    parser.add_argument("--dictionary-path", type=str, required=True, help="Path to concept dictionary tensor [num_concepts, dim].")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to vocab text file, one concept per line.")
    parser.add_argument("--mean-path", type=str, required=True, help="Path to precomputed mean vector used by SPLICE.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Destination JSONL containing one row per patch.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k concepts to keep per patch.")
    parser.add_argument("--l1-penalty", type=float, default=0.005, help="L1 penalty for sparse decomposition.")
    parser.add_argument("--solver", type=str, default="skl", choices=["skl", "admm"], help="Sparse solver.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for encoding patches.")
    parser.add_argument("--patch-limit", type=int, default=0, help="Optional limit on number of patches to process (0 means all).")
    parser.add_argument("--print-first", type=int, default=10, help="Print top concepts for the first N patches.")
    parser.add_argument("--anchors", type=str, default="", help="Comma-separated concept anchors to probe in patch scores.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if np is not None and isinstance(value, np.ndarray):
        return torch.from_numpy(value).float().cpu()
    raise TypeError(f"Unsupported tensor type: {type(value)}")


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
            raise ImportError("safetensors is required to load .safetensors files. Install with: pip install safetensors")
        return load_safetensors_file(str(path), device="cpu")
    raise ValueError(f"Unsupported file suffix: {path}")


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


def load_dictionary(dictionary_path: str, vocab_size: int) -> torch.Tensor:
    dictionary = to_tensor(load_object(Path(dictionary_path)))
    if dictionary.ndim != 2:
        raise ValueError(f"Dictionary must be 2D, got shape {tuple(dictionary.shape)}")
    if dictionary.shape[0] != vocab_size:
        raise ValueError(f"Dictionary rows ({dictionary.shape[0]}) must match vocab size ({vocab_size}).")
    dictionary = F.normalize(dictionary, dim=1)
    dictionary = F.normalize(dictionary - dictionary.mean(dim=0, keepdim=True), dim=1)
    return dictionary


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim != 1:
        raise ValueError(f"Expected 1D mean vector, got shape {tuple(vec.shape)}")
    norm = torch.linalg.norm(vec)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        raise ValueError("Mean vector has invalid norm.")
    return vec / norm


def load_mean_vector(mean_path: str) -> torch.Tensor:
    obj = load_object(Path(mean_path))
    if isinstance(obj, dict):
        if "mean" in obj:
            vec = to_tensor(obj["mean"])
        else:
            tensor_values = [to_tensor(v) for v in obj.values() if isinstance(v, torch.Tensor) or (np is not None and isinstance(v, np.ndarray))]
            if not tensor_values:
                raise ValueError(f"Could not find tensor in mean file: {mean_path}")
            vec = tensor_values[0]
    else:
        vec = to_tensor(obj)
    if vec.ndim > 1:
        vec = vec.reshape(-1, vec.shape[-1]).mean(dim=0)
    return normalize_vector(vec)


def extract_page_tensor(embedding_file: Path, page_index: int) -> Tuple[str, torch.Tensor]:
    obj = load_object(embedding_file)
    base_id = embedding_file.stem

    tensor: Optional[torch.Tensor] = None
    if isinstance(obj, dict):
        if "embeddings" in obj:
            tensor = to_tensor(obj["embeddings"])
        else:
            tensor_values = [to_tensor(v) for v in obj.values() if isinstance(v, torch.Tensor) or (np is not None and isinstance(v, np.ndarray))]
            if len(tensor_values) == 1:
                tensor = tensor_values[0]
    elif isinstance(obj, torch.Tensor) or (np is not None and isinstance(obj, np.ndarray)):
        tensor = to_tensor(obj)

    if tensor is None:
        raise ValueError(f"Unsupported embedding content in: {embedding_file}")

    if tensor.ndim == 3:
        if page_index < 0 or page_index >= tensor.shape[0]:
            raise IndexError(f"page-index {page_index} out of range [0, {tensor.shape[0] - 1}]")
        page_id = f"{base_id}:{page_index}"
        page_tensor = tensor[page_index]
    elif tensor.ndim == 2:
        if page_index != 0:
            raise IndexError("Embedding file has a single page tensor. Use --page-index 0.")
        page_id = f"{base_id}:0"
        page_tensor = tensor
    else:
        raise ValueError(f"Expected tensor ndim 2 or 3, got {tensor.ndim} from {embedding_file}")

    if page_tensor.shape[-1] <= 1:
        raise ValueError(f"Invalid page tensor shape: {tuple(page_tensor.shape)}")
    return page_id, page_tensor


def format_top_concepts(weights_row: torch.Tensor, vocab: List[str], topk: int) -> List[Dict[str, Any]]:
    k = min(topk, weights_row.shape[0])
    vals, inds = torch.topk(weights_row, k=k, largest=True)
    out: List[Dict[str, Any]] = []
    for value, idx in zip(vals.tolist(), inds.tolist()):
        if value <= 0:
            continue
        out.append({"concept": vocab[idx], "weight": round(float(value), 6)})
    return out


def main() -> None:
    args = parse_args()

    embedding_file = Path(args.embedding_file)
    if not embedding_file.is_file():
        raise FileNotFoundError(f"embedding-file not found: {embedding_file}")

    vocab = load_vocab(args.vocab_path)
    dictionary = load_dictionary(args.dictionary_path, vocab_size=len(vocab))
    image_mean = load_mean_vector(args.mean_path)
    if image_mean.shape[0] != dictionary.shape[1]:
        raise ValueError(
            f"Mean dim ({image_mean.shape[0]}) does not match dictionary dim ({dictionary.shape[1]})."
        )

    page_id, page_tensor = extract_page_tensor(embedding_file, args.page_index)
    patch_vectors = page_tensor.reshape(-1, page_tensor.shape[-1]).float().cpu()
    patch_vectors = F.normalize(patch_vectors, dim=1)

    if args.patch_limit > 0:
        patch_vectors = patch_vectors[: args.patch_limit]

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

    all_weights: List[torch.Tensor] = []
    for start in range(0, patch_vectors.shape[0], args.batch_size):
        end = min(start + args.batch_size, patch_vectors.shape[0])
        x = patch_vectors[start:end].to(model.device)
        with torch.no_grad():
            w = model.encode_image(x).detach().cpu()
        all_weights.append(w)
    weights = torch.cat(all_weights, dim=0)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        for patch_idx in range(weights.shape[0]):
            row = weights[patch_idx]
            record = {
                "id": f"{page_id}#patch{patch_idx}",
                "page_id": page_id,
                "patch_index": patch_idx,
                "top_concepts": format_top_concepts(row, vocab, args.topk),
            }
            handle.write(json.dumps(record) + "\n")

    print(f"Page ID: {page_id}")
    print(f"Patches processed: {weights.shape[0]}")
    print(f"Wrote patch labels: {out_path}")

    to_print = min(args.print_first, weights.shape[0])
    if to_print > 0:
        print("\nFirst patches:")
        for patch_idx in range(to_print):
            top_concepts = format_top_concepts(weights[patch_idx], vocab, args.topk)
            compact = [f"{c['concept']}:{c['weight']:.6f}" for c in top_concepts[:5]]
            print(f"patch {patch_idx:04d} -> {compact}")

    if args.anchors.strip():
        concept_to_idx = {concept.lower(): idx for idx, concept in enumerate(vocab)}
        anchors = [a.strip().lower() for a in args.anchors.split(",") if a.strip()]
        print("\nAnchor max hits:")
        for anchor in anchors:
            idx = concept_to_idx.get(anchor)
            if idx is None:
                print(f"{anchor} -> not_in_vocab")
                continue
            column = weights[:, idx]
            max_val, patch_idx = torch.max(column, dim=0)
            print(f"{anchor} -> patch {int(patch_idx)} weight={float(max_val):.6f}")


if __name__ == "__main__":
    main()
