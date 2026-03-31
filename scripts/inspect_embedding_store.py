#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect embedding directory structure and sample tensor metadata.")
    parser.add_argument("--embeddings-path", type=str, required=True, help="Directory with embedding files.")
    parser.add_argument("--sample-files", type=int, default=10, help="How many embedding files to inspect.")
    return parser.parse_args()


def count_extensions(files: List[Path]) -> Dict[str, int]:
    counter: Counter = Counter()
    for file_path in files:
        suffix = file_path.suffix.lower() if file_path.suffix else "<none>"
        counter[suffix] += 1
    return dict(counter)


def inspect_json_file(path: Path) -> None:
    print(f"\nJSON metadata file: {path}")
    try:
        if path.suffix.lower() == ".jsonl":
            with open(path, "r") as handle:
                line = handle.readline().strip()
                if not line:
                    print("  Empty JSONL file.")
                    return
                obj = json.loads(line)
                print(f"  First JSONL row keys: {list(obj.keys())[:20]}")
        else:
            with open(path, "r") as handle:
                obj = json.load(handle)
            if isinstance(obj, dict):
                print(f"  Top-level keys: {list(obj.keys())[:20]}")
            elif isinstance(obj, list):
                print(f"  Top-level list length: {len(obj)}")
                if obj and isinstance(obj[0], dict):
                    print(f"  First item keys: {list(obj[0].keys())[:20]}")
            else:
                print(f"  Top-level type: {type(obj).__name__}")
    except Exception as exc:
        print(f"  Could not parse JSON metadata: {exc}")


def inspect_safetensors_file(path: Path) -> None:
    if safe_open is None:
        raise RuntimeError("safetensors is not installed. Run: pip install safetensors")

    print(f"\n--- {path}")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        print(f"keys ({len(keys)}): {keys[:20]}")
        if metadata:
            print(f"metadata keys: {list(metadata.keys())[:20]}")
        for key in keys[:10]:
            tensor = f.get_tensor(key)
            print(f"  {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def inspect_torch_file(path: Path) -> None:
    print(f"\n--- {path}")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        print(f"tensor: shape={tuple(obj.shape)} dtype={obj.dtype}")
        return
    if isinstance(obj, dict):
        print(f"dict keys: {list(obj.keys())[:20]}")
        for key, value in list(obj.items())[:10]:
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value).__name__}")
        return
    if isinstance(obj, list):
        print(f"list length: {len(obj)}")
        if obj:
            print(f"first element type: {type(obj[0]).__name__}")
        return
    print(f"type: {type(obj).__name__}")


def main() -> None:
    args = parse_args()
    root = Path(args.embeddings_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    files = sorted([p for p in root.rglob("*") if p.is_file()])
    if not files:
        raise ValueError(f"No files found under: {root}")

    print(f"Directory: {root}")
    print(f"Total files: {len(files)}")
    for ext, count in sorted(count_extensions(files).items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext}: {count}")

    json_files = [p for p in files if p.suffix.lower() in {".json", ".jsonl"}]
    for path in json_files[:5]:
        inspect_json_file(path)

    embedding_files = [p for p in files if p.suffix.lower() in {".safetensors", ".pt", ".pth"}]
    for path in embedding_files[: args.sample_files]:
        if path.suffix.lower() == ".safetensors":
            inspect_safetensors_file(path)
        else:
            inspect_torch_file(path)


if __name__ == "__main__":
    main()
