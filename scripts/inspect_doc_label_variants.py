#!/usr/bin/env python3
import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inspect query artifacts and page-label variants for one doc/page. "
            "Useful for comparing regular, noisy/merged, layout, and backfilled outputs."
        )
    )
    p.add_argument("--doc-id", type=str, required=True)
    p.add_argument("--page-index", type=int, default=0)
    p.add_argument(
        "--output-root",
        type=str,
        default="/mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings",
        help="Root folder that contains debug_<DOC>_linked_context_auto directories.",
    )
    p.add_argument(
        "--sample-rows",
        type=int,
        default=6,
        help="How many sample patch rows to print per label file.",
    )
    p.add_argument(
        "--top-concepts",
        type=int,
        default=12,
        help="How many top concept frequencies to print per label file.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def first_nonempty_rows(rows: List[Dict], n: int) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        tc = r.get("top_concepts", [])
        if isinstance(tc, list) and len(tc) > 0:
            out.append(r)
        if len(out) >= n:
            break
    return out


def concept_counter(rows: List[Dict]) -> Counter:
    c = Counter()
    for r in rows:
        for item in r.get("top_concepts", []):
            if not isinstance(item, dict):
                continue
            concept = str(item.get("concept", "")).strip().lower()
            if concept:
                c[concept] += 1
    return c


def concept_weight_stats(rows: List[Dict]) -> Tuple[float, float]:
    top1_sum = 0.0
    top1_n = 0
    nonempty = 0
    for r in rows:
        tc = r.get("top_concepts", [])
        if isinstance(tc, list) and len(tc) > 0:
            nonempty += 1
            item = tc[0]
            if isinstance(item, dict):
                try:
                    top1_sum += float(item.get("weight", 0.0))
                except Exception:
                    top1_sum += 0.0
                top1_n += 1
    avg_top1 = (top1_sum / top1_n) if top1_n else 0.0
    nonempty_ratio = (nonempty / len(rows)) if rows else 0.0
    return avg_top1, nonempty_ratio


def print_query_artifacts(ctx: Path) -> None:
    print("\n=== Query Artifacts ===")

    queries_jsonl = ctx / "doc_seed_text_queries.jsonl"
    vocab_txt = ctx / "doc_seed_concepts_text_strict.txt"
    raw_vocab_txt = ctx / "doc_seed_concepts_text_raw.txt"

    if queries_jsonl.is_file():
        rows = load_jsonl(queries_jsonl)
        print(f"[doc_seed_text_queries.jsonl] rows={len(rows)} path={queries_jsonl}")
        for r in rows[:5]:
            qid = str(r.get("query_id", ""))
            qtext = str(r.get("query_text", ""))
            print(f"  - {qid}: {qtext}")
    else:
        print(f"[doc_seed_text_queries.jsonl] missing: {queries_jsonl}")

    if vocab_txt.is_file():
        lines = [x.strip() for x in vocab_txt.read_text().splitlines() if x.strip()]
        print(f"[doc_seed_concepts_text_strict.txt] count={len(lines)} path={vocab_txt}")
        print("  - first:", ", ".join(lines[:12]))
    else:
        print(f"[doc_seed_concepts_text_strict.txt] missing: {vocab_txt}")

    if raw_vocab_txt.is_file():
        lines = [x.strip() for x in raw_vocab_txt.read_text().splitlines() if x.strip()]
        print(f"[doc_seed_concepts_text_raw.txt] count={len(lines)} path={raw_vocab_txt}")
        print("  - first:", ", ".join(lines[:12]))


def latest_existing(paths: List[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return None
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


def resolve_variants(ctx: Path, page_idx: int) -> List[Tuple[str, Optional[Path]]]:
    candidates: List[Tuple[str, Optional[Path]]] = []

    candidates.append(("text_regular", latest_existing([ctx / "doc_seed_text_patch_labels.jsonl"])))
    candidates.append(("merged_noisy", latest_existing([ctx / "doc_seed_text_visual_merged.jsonl"])))
    candidates.append(("merged_ocr_filtered", latest_existing([ctx / "doc_seed_text_visual_merged_ocr_v6b.jsonl"])))
    candidates.append(("layout_v4", latest_existing([ctx / "doc_seed_layout_patch_labels_v4.jsonl"])))
    candidates.append(("layout_v4_strict", latest_existing([ctx / "doc_seed_layout_patch_labels_v4_strict.jsonl"])))

    # Backfilled/final variants for this page.
    pattern = str(ctx / f"doc_seed_layout_patch_labels_page{page_idx}_final_strict*.jsonl")
    final_paths = [Path(p) for p in glob.glob(pattern)]
    final_latest = latest_existing(final_paths)
    candidates.append(("final_backfilled_latest", final_latest))

    # Helpful custom variants users often create.
    extra_patterns = [
        str(ctx / f"doc_seed_layout_patch_labels_page{page_idx}_*.jsonl"),
        str(ctx / "doc_seed_layout_patch_labels_v*.jsonl"),
    ]
    extra_set = set()
    for pat in extra_patterns:
        for p in glob.glob(pat):
            pp = Path(p)
            if pp.is_file():
                extra_set.add(pp)
    # Keep a few newest extras to inspect.
    extras = sorted(extra_set, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    for i, p in enumerate(extras, start=1):
        candidates.append((f"extra_{i}", p))

    return candidates


def print_page_variant(
    name: str,
    path: Path,
    page_id: str,
    sample_rows: int,
    top_concepts: int,
) -> None:
    rows = load_jsonl(path)
    page_rows = [r for r in rows if str(r.get("page_id", "")) == page_id]

    print(f"\n--- {name} ---")
    print(f"path: {path}")
    print(f"rows_total={len(rows)} rows_page={len(page_rows)} page_id={page_id}")

    if not page_rows:
        return

    avg_top1, nonempty_ratio = concept_weight_stats(page_rows)
    print(f"page_nonempty_ratio={nonempty_ratio:.3f} page_avg_top1={avg_top1:.4f}")

    cc = concept_counter(page_rows)
    print("top_concepts_by_patch_freq:")
    for concept, cnt in cc.most_common(max(0, top_concepts)):
        print(f"  - {concept}: {cnt}")

    print("sample_rows:")
    for r in first_nonempty_rows(page_rows, max(0, sample_rows)):
        patch = r.get("patch_index", -1)
        tc = r.get("top_concepts", [])
        compact = []
        for item in tc[:8]:
            if isinstance(item, dict):
                c = str(item.get("concept", ""))
                try:
                    w = float(item.get("weight", 0.0))
                except Exception:
                    w = 0.0
                compact.append(f"{c}:{w:.3f}")
            else:
                compact.append(str(item))
        print(f"  - patch={patch} concepts=[{', '.join(compact)}]")


def main() -> None:
    args = parse_args()

    page_id = f"{args.doc_id}:{args.page_index}"
    ctx = Path(args.output_root) / f"debug_{args.doc_id}_linked_context_auto"

    print(f"doc_id={args.doc_id}")
    print(f"page_id={page_id}")
    print(f"ctx={ctx}")
    if not ctx.is_dir():
        raise FileNotFoundError(f"Context directory not found: {ctx}")

    print_query_artifacts(ctx)

    print("\n=== Page Label Variants ===")
    variants = resolve_variants(ctx, args.page_index)
    seen: set = set()
    for name, path in variants:
        if path is None:
            print(f"\n--- {name} ---")
            print("path: MISSING")
            continue
        # Avoid printing duplicates if same file got picked by multiple buckets.
        if str(path) in seen:
            continue
        seen.add(str(path))
        print_page_variant(
            name=name,
            path=path,
            page_id=page_id,
            sample_rows=args.sample_rows,
            top_concepts=args.top_concepts,
        )


if __name__ == "__main__":
    main()
