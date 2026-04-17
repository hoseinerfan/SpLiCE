#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PAGE_FILE_RE = re.compile(r"^(?P<doc_id>.+)_page(?P<page_idx>\d+)_labels_tfill\.jsonl$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export page-level patch class assignments from strict layout outputs. "
            "Each patch is assigned one exclusive class using overlay precedence: "
            "image > text > table > unassigned."
        )
    )
    p.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="Target doc id. Repeatable.",
    )
    p.add_argument(
        "--doc-ids-file",
        type=str,
        default="",
        help="Optional newline-delimited doc id file.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs",
        help="Root folder containing debug_<DOC>_linked_context_auto output directories.",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default="latest",
        help="Specific run tag under each doc context, or 'latest'.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Combined JSON output path.",
    )
    p.add_argument(
        "--image-token-start",
        type=int,
        default=0,
        help="Start patch index for page-image tokens.",
    )
    p.add_argument(
        "--image-token-count",
        type=int,
        default=1024,
        help="Number of page-image patches per page.",
    )
    return p.parse_args()


def load_doc_ids(args: argparse.Namespace) -> List[str]:
    seen = set()
    out: List[str] = []

    def add(raw: str) -> None:
        doc_id = str(raw).strip()
        if not doc_id or doc_id in seen:
            return
        seen.add(doc_id)
        out.append(doc_id)

    for doc_id in args.doc_id:
        add(doc_id)

    if args.doc_ids_file:
        with open(args.doc_ids_file, "r") as f:
            for line in f:
                add(line)

    if not out:
        raise ValueError("No doc ids provided. Use --doc-id or --doc-ids-file.")
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_run_dir(output_root: Path, doc_id: str, run_tag: str) -> Path:
    ctx = output_root / f"debug_{doc_id}_linked_context_auto"
    if not ctx.is_dir():
        raise FileNotFoundError(f"Doc context not found: {ctx}")

    if run_tag != "latest":
        run_dir = ctx / run_tag
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        return run_dir

    candidates: List[Path] = []
    for child in ctx.iterdir():
        if not child.is_dir():
            continue
        if any(child.glob(f"{doc_id}_page*_labels_tfill.jsonl")):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No layout run directories found under: {ctx}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_page_file(path: Path) -> Tuple[str, int]:
    m = PAGE_FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized page labels filename: {path.name}")
    return m.group("doc_id"), int(m.group("page_idx"))


def page_file_sort_key(path: Path) -> Tuple[int, str]:
    _, page_idx = parse_page_file(path)
    return page_idx, path.name


def infer_patch_class(top_concepts: Sequence[Dict[str, Any]]) -> str:
    names = {
        str(item.get("concept", "")).strip().lower()
        for item in top_concepts
        if isinstance(item, dict)
    }
    if "image_region" in names or "visual_region" in names:
        return "image"
    if "ocr_text" in names or "text_region" in names or "text" in names:
        return "text"
    if "table_all" in names or "table_structure" in names or "table_text" in names:
        return "table"
    return "unassigned"


def export_doc(
    output_root: Path,
    doc_id: str,
    run_tag: str,
    image_token_start: int,
    image_token_count: int,
) -> Dict[str, Any]:
    run_dir = resolve_run_dir(output_root=output_root, doc_id=doc_id, run_tag=run_tag)
    page_files = sorted(run_dir.glob(f"{doc_id}_page*_labels_tfill.jsonl"), key=page_file_sort_key)
    if not page_files:
        raise FileNotFoundError(f"No page label files found in: {run_dir}")

    page_records: Dict[str, Any] = {}
    total_counts: Counter = Counter()
    image_token_end = image_token_start + image_token_count

    for page_file in page_files:
        _, page_idx = parse_page_file(page_file)
        page_id = f"{doc_id}:{page_idx}"

        class_to_patches: Dict[str, List[int]] = {
            "text": [],
            "table": [],
            "image": [],
            "unassigned": [],
        }

        for row in load_jsonl(page_file):
            if str(row.get("page_id", "")) != page_id:
                continue
            patch_index = int(row.get("patch_index", -1))
            if patch_index < image_token_start or patch_index >= image_token_end:
                continue
            patch_class = infer_patch_class(row.get("top_concepts", []) or [])
            class_to_patches[patch_class].append(patch_index)

        for key in class_to_patches:
            class_to_patches[key].sort()
            total_counts[key] += len(class_to_patches[key])

        page_records[str(page_idx)] = {
            "page_id": page_id,
            "page_index": page_idx,
            "counts": {k: len(v) for k, v in class_to_patches.items()},
            "patches": class_to_patches,
        }

    return {
        "doc_id": doc_id,
        "run_dir": str(run_dir),
        "class_precedence": ["image", "text", "table", "unassigned"],
        "image_token_start": image_token_start,
        "image_token_count": image_token_count,
        "page_count": len(page_records),
        "counts": {k: int(total_counts[k]) for k in ["text", "table", "image", "unassigned"]},
        "pages": page_records,
    }


def main() -> None:
    args = parse_args()
    doc_ids = load_doc_ids(args)

    output_root = Path(args.output_root)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs: Dict[str, Any] = {}
    for doc_id in doc_ids:
        docs[doc_id] = export_doc(
            output_root=output_root,
            doc_id=doc_id,
            run_tag=args.run_tag,
            image_token_start=int(args.image_token_start),
            image_token_count=int(args.image_token_count),
        )

    out = {
        "format": "layout_patch_assignments_v1",
        "class_precedence": ["image", "text", "table", "unassigned"],
        "docs": docs,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote: {out_path}")
    print(f"docs: {len(docs)}")
    for doc_id in doc_ids:
        rec = docs[doc_id]
        print(
            f"{doc_id}: pages={rec['page_count']} "
            f"text={rec['counts']['text']} "
            f"table={rec['counts']['table']} "
            f"image={rec['counts']['image']} "
            f"unassigned={rec['counts']['unassigned']}"
        )


if __name__ == "__main__":
    main()
