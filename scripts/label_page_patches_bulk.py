#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from splice.model import SPLICE

try:
    import numpy as np
except ImportError:  # Optional dependency for .npy input.
    np = None

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # Optional dependency for .safetensors input.
    load_safetensors_file = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label all page patches for an embedding file/folder with concept vocab."
    )
    parser.add_argument("--embeddings-path", type=str, required=True, help="Path to one embedding file or a directory of files.")
    parser.add_argument("--dictionary-path", type=str, required=True, help="Concept dictionary .pt with shape [num_concepts, dim].")
    parser.add_argument("--vocab-path", type=str, required=True, help="Concept vocab .txt (one concept per line).")
    parser.add_argument("--mean-path", type=str, required=True, help="Precomputed mean vector used for centering.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory; writes one JSONL per input embedding file.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k concepts to keep for each patch.")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Minimum weight required to keep a concept in output.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Patch batch size.")
    parser.add_argument(
        "--method",
        type=str,
        default="cosine",
        choices=["cosine", "splice"],
        help="cosine is fast and recommended for all-patch labeling; splice runs sparse decomposition.",
    )
    parser.add_argument("--solver", type=str, default="skl", choices=["skl", "admm"], help="Used only when --method splice.")
    parser.add_argument("--l1-penalty", type=float, default=0.02, help="Used only when --method splice.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for embedding files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files whose output JSONL already exists.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel processing.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index [0, num-shards).")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap on number of files to process (0 means all).")
    parser.add_argument("--max-pages-per-file", type=int, default=0, help="Optional cap on pages per file (0 means all).")
    parser.add_argument("--patch-limit-per-page", type=int, default=0, help="Optional cap on patches per page (0 means all).")
    parser.add_argument("--print-every", type=int, default=10, help="Progress print interval in files.")
    parser.add_argument("--positive-only", action="store_true", help="Drop negative concept scores.")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON output.")
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
            raise ImportError("safetensors is required for .safetensors files. Install with: pip install safetensors")
        return load_safetensors_file(str(path), device="cpu")
    raise ValueError(f"Unsupported file suffix: {path}")


def list_embedding_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Embeddings path not found: {path}")

    valid_suffixes = {".pt", ".pth", ".npy", ".safetensors"}
    files: List[Path] = []
    if recursive:
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                p = Path(root) / filename
                if p.suffix.lower() in valid_suffixes:
                    files.append(p)
    else:
        for p in path.iterdir():
            if p.is_file() and p.suffix.lower() in valid_suffixes:
                files.append(p)

    files.sort()
    if not files:
        raise ValueError(f"No embedding files found under: {path}")
    return files


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


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {tuple(vec.shape)}")
    norm = torch.linalg.norm(vec)
    if not torch.isfinite(norm) or norm.item() == 0:
        raise ValueError("Vector has invalid norm.")
    return vec / norm


def load_mean_vector(mean_path: str) -> torch.Tensor:
    obj = load_object(Path(mean_path))
    if isinstance(obj, dict):
        if "mean" in obj:
            vec = to_tensor(obj["mean"])
        else:
            tensor_values = [
                to_tensor(v)
                for v in obj.values()
                if isinstance(v, torch.Tensor) or (np is not None and isinstance(v, np.ndarray))
            ]
            if not tensor_values:
                raise ValueError(f"Could not find tensor in mean file: {mean_path}")
            vec = tensor_values[0]
    else:
        vec = to_tensor(obj)
    if vec.ndim > 1:
        vec = vec.reshape(-1, vec.shape[-1]).mean(dim=0)
    return normalize_vector(vec)


def load_dictionary(dictionary_path: str, vocab_size: int) -> torch.Tensor:
    dictionary = to_tensor(load_object(Path(dictionary_path)))
    if dictionary.ndim != 2:
        raise ValueError(f"Dictionary must be 2D, got shape {tuple(dictionary.shape)}")
    if dictionary.shape[0] != vocab_size:
        raise ValueError(
            f"Dictionary rows ({dictionary.shape[0]}) must match vocab size ({vocab_size})."
        )
    dictionary = F.normalize(dictionary, dim=1)
    dictionary = F.normalize(dictionary - dictionary.mean(dim=0, keepdim=True), dim=1)
    return dictionary


def extract_pages(obj: Any, base_id: str) -> List[Tuple[str, torch.Tensor]]:
    tensor: Optional[torch.Tensor] = None
    if isinstance(obj, dict):
        if "embeddings" in obj:
            tensor = to_tensor(obj["embeddings"])
        else:
            tensor_values = [
                to_tensor(v)
                for v in obj.values()
                if isinstance(v, torch.Tensor) or (np is not None and isinstance(v, np.ndarray))
            ]
            if len(tensor_values) == 1:
                tensor = tensor_values[0]
    elif isinstance(obj, torch.Tensor) or (np is not None and isinstance(obj, np.ndarray)):
        tensor = to_tensor(obj)

    if tensor is None:
        raise ValueError(f"Could not parse embeddings for {base_id}")

    pages: List[Tuple[str, torch.Tensor]] = []
    if tensor.ndim == 3:
        for idx in range(tensor.shape[0]):
            pages.append((f"{base_id}:{idx}", tensor[idx]))
    elif tensor.ndim == 2:
        pages.append((f"{base_id}:0", tensor))
    else:
        raise ValueError(f"Unsupported tensor ndim {tensor.ndim} for {base_id}")
    return pages


def top_concepts_from_scores(
    scores_row: torch.Tensor,
    vocab: List[str],
    topk: int,
    min_weight: float,
) -> List[Dict[str, Any]]:
    k = min(topk, scores_row.shape[0])
    vals, inds = torch.topk(scores_row, k=k, largest=True)
    out: List[Dict[str, Any]] = []
    for value, idx in zip(vals.tolist(), inds.tolist()):
        if value < min_weight:
            continue
        out.append({"concept": vocab[idx], "weight": round(float(value), 6)})
    return out


def output_path_for_file(embeddings_root: Path, input_file: Path, output_dir: Path) -> Path:
    if embeddings_root.is_file():
        return output_dir / f"{input_file.stem}.jsonl"
    rel = input_file.relative_to(embeddings_root).with_suffix(".jsonl")
    return output_dir / rel


def encode_scores_cosine(
    patch_vectors: torch.Tensor,
    dictionary_dev: torch.Tensor,
    mean_dev: torch.Tensor,
    batch_size: int,
    positive_only: bool,
    device: str,
) -> torch.Tensor:
    # patch_vectors: [num_patches, dim], normalized
    all_scores: List[torch.Tensor] = []
    for start in range(0, patch_vectors.shape[0], batch_size):
        end = min(start + batch_size, patch_vectors.shape[0])
        x = patch_vectors[start:end].to(device)
        centered = F.normalize(x - mean_dev, dim=1)
        scores = centered @ dictionary_dev.T
        if positive_only:
            scores = torch.clamp_min(scores, 0.0)
        all_scores.append(scores.detach().cpu())
    return torch.cat(all_scores, dim=0)


def encode_scores_splice(
    patch_vectors: torch.Tensor,
    model: "SPLICE",
    batch_size: int,
    positive_only: bool,
) -> torch.Tensor:
    all_scores: List[torch.Tensor] = []
    for start in range(0, patch_vectors.shape[0], batch_size):
        end = min(start + batch_size, patch_vectors.shape[0])
        x = patch_vectors[start:end].to(model.device)
        with torch.no_grad():
            weights = model.encode_image(x)
        if positive_only:
            weights = torch.clamp_min(weights, 0.0)
        all_scores.append(weights.detach().cpu())
    return torch.cat(all_scores, dim=0)


def process_file(
    input_file: Path,
    output_file: Path,
    vocab: List[str],
    dictionary_cpu: torch.Tensor,
    mean_cpu: torch.Tensor,
    args: argparse.Namespace,
    model: Optional["SPLICE"],
) -> Tuple[int, int]:
    obj = load_object(input_file)
    pages = extract_pages(obj, base_id=input_file.stem)
    if args.max_pages_per_file > 0:
        pages = pages[: args.max_pages_per_file]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    n_pages = 0
    n_patches = 0
    dict_dev = dictionary_cpu.to(args.device) if args.method == "cosine" else None
    mean_dev = mean_cpu.to(args.device) if args.method == "cosine" else None

    with open(tmp_file, "w") as handle:
        for page_id, page_tensor in pages:
            patch_vectors = page_tensor.reshape(-1, page_tensor.shape[-1]).float().cpu()
            patch_vectors = F.normalize(patch_vectors, dim=1)
            if args.patch_limit_per_page > 0:
                patch_vectors = patch_vectors[: args.patch_limit_per_page]

            if args.method == "cosine":
                assert dict_dev is not None and mean_dev is not None
                scores = encode_scores_cosine(
                    patch_vectors=patch_vectors,
                    dictionary_dev=dict_dev,
                    mean_dev=mean_dev,
                    batch_size=args.batch_size,
                    positive_only=args.positive_only,
                    device=args.device,
                )
            else:
                if model is None:
                    raise RuntimeError("SPLICE model is not initialized.")
                scores = encode_scores_splice(
                    patch_vectors=patch_vectors,
                    model=model,
                    batch_size=args.batch_size,
                    positive_only=args.positive_only,
                )

            for patch_idx in range(scores.shape[0]):
                row = scores[patch_idx]
                record = {
                    "id": f"{page_id}#patch{patch_idx}",
                    "page_id": page_id,
                    "patch_index": patch_idx,
                    "top_concepts": top_concepts_from_scores(
                        row,
                        vocab=vocab,
                        topk=args.topk,
                        min_weight=args.min_weight,
                    ),
                }
                handle.write(json.dumps(record) + "\n")

            n_pages += 1
            n_patches += scores.shape[0]

    tmp_file.replace(output_file)
    return n_pages, n_patches


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    embeddings_root = Path(args.embeddings_path)
    output_dir = Path(args.output_dir)
    vocab = load_vocab(args.vocab_path)
    dictionary_cpu = load_dictionary(args.dictionary_path, vocab_size=len(vocab))
    mean_cpu = load_mean_vector(args.mean_path)

    if mean_cpu.shape[0] != dictionary_cpu.shape[1]:
        raise ValueError(
            f"Mean dim ({mean_cpu.shape[0]}) does not match dictionary dim ({dictionary_cpu.shape[1]})."
        )

    files = list_embedding_files(embeddings_root, recursive=args.recursive)
    sharded_files = [p for i, p in enumerate(files) if (i % args.num_shards) == args.shard_index]
    if args.max_files > 0:
        sharded_files = sharded_files[: args.max_files]

    model: Optional["SPLICE"] = None
    if args.method == "splice":
        from splice.model import SPLICE

        model = SPLICE(
            image_mean=mean_cpu,
            dictionary=dictionary_cpu,
            clip=None,
            solver=args.solver,
            l1_penalty=args.l1_penalty,
            return_weights=True,
            return_cosine=False,
            device=args.device,
        )
        model.eval()

    total_files = len(sharded_files)
    done_files = 0
    skipped_files = 0
    failed_files = 0
    total_pages = 0
    total_patches = 0

    print(
        f"Starting patch labeling: files={total_files} method={args.method} "
        f"shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )

    for idx, input_file in enumerate(sharded_files, start=1):
        output_file = output_path_for_file(embeddings_root, input_file, output_dir)

        if args.skip_existing and output_file.exists():
            skipped_files += 1
            continue

        try:
            n_pages, n_patches = process_file(
                input_file=input_file,
                output_file=output_file,
                vocab=vocab,
                dictionary_cpu=dictionary_cpu,
                mean_cpu=mean_cpu,
                args=args,
                model=model,
            )
            done_files += 1
            total_pages += n_pages
            total_patches += n_patches
        except Exception as exc:
            failed_files += 1
            print(f"[ERROR] {input_file}: {exc}", flush=True)

        if idx % max(1, args.print_every) == 0:
            print(
                f"Progress {idx}/{total_files} | done={done_files} skipped={skipped_files} "
                f"failed={failed_files} pages={total_pages} patches={total_patches}",
                flush=True,
            )

    summary: Dict[str, Any] = {
        "embeddings_path": str(embeddings_root),
        "output_dir": str(output_dir),
        "method": args.method,
        "solver": args.solver if args.method == "splice" else None,
        "l1_penalty": args.l1_penalty if args.method == "splice" else None,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_input_files": len(files),
        "sharded_files": total_files,
        "done_files": done_files,
        "skipped_files": skipped_files,
        "failed_files": failed_files,
        "total_pages": total_pages,
        "total_patches": total_patches,
        "topk": args.topk,
        "min_weight": args.min_weight,
    }

    print("Done.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
