#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed query JSONL with ColPali and save one .safetensors per query_id."
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="vidore/colpali-v1.2")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "colpali_engine", "transformers"])
    parser.add_argument("--query-id-field", type=str, default="query_id")
    parser.add_argument("--query-text-field", type=str, default="query_text")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    text = text.strip("._")
    return text or "item"


def load_queries(path: str, id_field: str, text_field: str, max_queries: int | None) -> List[Tuple[str, str]]:
    queries: List[Tuple[str, str]] = []
    with open(path, "r") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = row.get(id_field, f"line-{line_idx}")
            query_text = row.get(text_field)
            if query_text is None:
                continue
            query_id = str(query_id)
            query_text = str(query_text).strip()
            if not query_text:
                continue
            queries.append((query_id, query_text))
            if max_queries is not None and len(queries) >= max_queries:
                break
    return queries


def extract_embeddings(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "embeddings"):
        return outputs.embeddings
    if isinstance(outputs, (tuple, list)) and outputs:
        if isinstance(outputs[0], torch.Tensor):
            return outputs[0]
    raise TypeError(f"Could not extract embeddings from model output type: {type(outputs)}")


class ColPaliEmbedder:
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        raise NotImplementedError


class ColPaliEngineEmbedder(ColPaliEmbedder):
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.device = self.model.device

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        batch = self.processor.process_queries(queries).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
        return extract_embeddings(outputs)


class TransformersEmbedder(ColPaliEmbedder):
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        self.model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.device = self.model.device

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        try:
            batch = self.processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(self.device)
        except TypeError:
            batch = self.processor(text=queries).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
        return extract_embeddings(outputs)


def init_embedder(args: argparse.Namespace, dtype: torch.dtype) -> ColPaliEmbedder:
    backend = args.backend

    if backend in {"auto", "colpali_engine"}:
        try:
            return ColPaliEngineEmbedder(args.model_name, args.device, dtype)
        except Exception as exc:
            if backend == "colpali_engine":
                raise
            print(f"[auto] colpali_engine backend unavailable: {exc}")

    if backend in {"auto", "transformers"}:
        try:
            return TransformersEmbedder(args.model_name, args.device, dtype)
        except Exception as exc:
            if backend == "transformers":
                raise
            print(f"[auto] transformers backend unavailable: {exc}")

    raise RuntimeError(
        "Could not initialize ColPali backend. Install either colpali_engine or transformers with ColPali support."
    )


def save_embedding(output_dir: Path, query_id: str, embedding: torch.Tensor) -> Path:
    filename = sanitize_filename(query_id) + ".safetensors"
    path = output_dir / filename
    save_file({"embeddings": embedding.contiguous().cpu()}, str(path))
    return path


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    queries = load_queries(
        path=args.input_jsonl,
        id_field=args.query_id_field,
        text_field=args.query_text_field,
        max_queries=args.max_queries,
    )
    if not queries:
        raise ValueError("No valid queries loaded.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = init_embedder(args, dtype)
    print(f"Loaded backend for model: {args.model_name}")
    print(f"Queries to embed: {len(queries)}")

    written = 0
    skipped = 0

    for start in range(0, len(queries), args.batch_size):
        chunk = queries[start : start + args.batch_size]
        chunk_ids = [qid for qid, _ in chunk]
        chunk_texts = [qtext for _, qtext in chunk]

        outputs = embedder.encode_queries(chunk_texts)
        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(1)

        if outputs.shape[0] != len(chunk_ids):
            raise ValueError(
                f"Batch size mismatch. Got embeddings batch={outputs.shape[0]}, expected={len(chunk_ids)}"
            )

        for i, query_id in enumerate(chunk_ids):
            out_path = out_dir / (sanitize_filename(query_id) + ".safetensors")
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            save_embedding(out_dir, query_id, outputs[i].detach())
            written += 1

        done = min(start + args.batch_size, len(queries))
        print(f"Embedded {done}/{len(queries)} queries...", flush=True)

    print(f"Done. Wrote {written} files to {out_dir}")
    if args.skip_existing:
        print(f"Skipped existing files: {skipped}")


if __name__ == "__main__":
    main()
