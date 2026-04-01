#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact quality audit for patch-level concept labels."
    )
    parser.add_argument("--labels-path", type=str, required=True, help="Patch label JSONL file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directory for JSONL files.")
    parser.add_argument("--sample-pages", type=int, default=20, help="Number of pages to sample for deep inspection.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling.")
    parser.add_argument(
        "--low-top1-threshold",
        type=float,
        default=0.10,
        help="Top-1 concept weight threshold for low-signal patch detection.",
    )
    parser.add_argument("--show-top-concepts", type=int, default=25, help="How many top concepts to print.")
    parser.add_argument("--show-flagged-pages", type=int, default=20, help="How many flagged pages to print.")
    parser.add_argument("--print-every-files", type=int, default=100, help="Progress print interval (files).")
    parser.add_argument(
        "--anchors",
        type=str,
        default="",
        help="Comma-separated concepts to track globally and in sampled pages.",
    )
    parser.add_argument("--vocab-path", type=str, default=None, help="Optional vocab file for seen/coverage reporting.")
    parser.add_argument("--output-summary-json", type=str, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--output-sample-jsonl", type=str, default=None, help="Optional sampled page report JSONL path.")
    parser.add_argument("--output-top-concepts-tsv", type=str, default=None, help="Optional top concepts TSV output path.")
    return parser.parse_args()


def list_jsonl_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"labels path not found: {path}")

    if recursive:
        files = sorted([p for p in path.rglob("*.jsonl") if p.is_file()])
    else:
        files = sorted([p for p in path.glob("*.jsonl") if p.is_file()])

    if not files:
        raise ValueError(f"No JSONL files found under: {path}")
    return files


def parse_anchors(anchors: str) -> Set[str]:
    if not anchors:
        return set()
    return {x.strip().lower() for x in anchors.split(",") if x.strip()}


def load_vocab(vocab_path: Optional[str]) -> Optional[Set[str]]:
    if not vocab_path:
        return None
    vocab: Set[str] = set()
    with open(vocab_path, "r") as handle:
        for line in handle:
            item = line.strip()
            if item:
                vocab.add(item.lower())
    return vocab


def infer_page_id(row: Dict) -> str:
    page_id = str(row.get("page_id", "")).strip()
    if page_id:
        return page_id
    rid = str(row.get("id", "")).strip()
    if "#patch" in rid:
        return rid.split("#patch", 1)[0]
    if rid:
        return rid
    return "unknown_page"


def parse_concepts(row: Dict) -> List[Tuple[str, float]]:
    raw = row.get("top_concepts", [])
    if not isinstance(raw, list):
        return []

    out: List[Tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept", "")).strip()
        if not concept:
            continue
        try:
            weight = float(item.get("weight", 0.0))
        except Exception:
            weight = 0.0
        out.append((concept, weight))

    if not out:
        return out

    # Keep best weight for duplicated concepts within the same patch.
    dedup: Dict[str, float] = {}
    for concept, weight in out:
        if concept not in dedup or weight > dedup[concept]:
            dedup[concept] = weight
    out = list(dedup.items())
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def iter_rows(files: Iterable[Path]) -> Iterable[Dict]:
    for path in files:
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def pct(num: float, den: float) -> float:
    return num / den if den else 0.0


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels_path)
    files = list_jsonl_files(labels_path, args.recursive)
    anchors = parse_anchors(args.anchors)
    vocab = load_vocab(args.vocab_path)

    print(f"Auditing files: {len(files)}")
    if anchors:
        print(f"Anchors tracked: {sorted(anchors)}")

    total_patches = 0
    malformed_rows = 0
    empty_patches = 0
    low_signal_patches = 0
    top1_sum = 0.0
    concepts_per_patch_sum = 0
    nonempty_patches = 0

    concept_patch_freq: Counter = Counter()
    concept_weight_sum: Counter = Counter()
    anchor_patch_hits: Counter = Counter()

    page_patches: Dict[str, int] = defaultdict(int)
    page_empty: Dict[str, int] = defaultdict(int)
    page_low: Dict[str, int] = defaultdict(int)
    page_top1_sum: Dict[str, float] = defaultdict(float)

    for idx, path in enumerate(files, start=1):
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total_patches += 1
                try:
                    row = json.loads(line)
                except Exception:
                    malformed_rows += 1
                    continue

                page_id = infer_page_id(row)
                page_patches[page_id] += 1

                concepts = parse_concepts(row)
                if not concepts:
                    empty_patches += 1
                    page_empty[page_id] += 1
                    continue

                nonempty_patches += 1
                concepts_per_patch_sum += len(concepts)
                top1 = float(concepts[0][1])
                top1_sum += top1
                page_top1_sum[page_id] += top1

                if top1 < args.low_top1_threshold:
                    low_signal_patches += 1
                    page_low[page_id] += 1

                for concept, weight in concepts:
                    concept_patch_freq[concept] += 1
                    concept_weight_sum[concept] += float(weight)
                    if anchors and concept.lower() in anchors:
                        anchor_patch_hits[concept.lower()] += 1

        if args.print_every_files > 0 and (idx % args.print_every_files == 0 or idx == len(files)):
            print(f"Progress: {idx}/{len(files)} files | patches={total_patches}")

    all_pages = sorted(page_patches.keys())
    sample_size = min(max(args.sample_pages, 0), len(all_pages))
    rng = random.Random(args.seed)
    sampled_pages = set(rng.sample(all_pages, sample_size)) if sample_size > 0 else set()

    sample_page_concept_freq: Dict[str, Counter] = {pid: Counter() for pid in sampled_pages}
    sample_page_concept_weight: Dict[str, Counter] = {pid: Counter() for pid in sampled_pages}
    sample_page_anchor_hits: Dict[str, Counter] = {pid: Counter() for pid in sampled_pages}

    if sampled_pages:
        for row in iter_rows(files):
            page_id = infer_page_id(row)
            if page_id not in sampled_pages:
                continue
            concepts = parse_concepts(row)
            for concept, weight in concepts:
                sample_page_concept_freq[page_id][concept] += 1
                sample_page_concept_weight[page_id][concept] += float(weight)
                if anchors and concept.lower() in anchors:
                    sample_page_anchor_hits[page_id][concept.lower()] += 1

    top_by_freq = concept_patch_freq.most_common(max(args.show_top_concepts, 0))
    top_by_weight = sorted(
        concept_weight_sum.items(),
        key=lambda x: x[1],
        reverse=True,
    )[: max(args.show_top_concepts, 0)]

    flagged_pages = []
    for pid in all_pages:
        n = page_patches[pid]
        if n <= 0:
            continue
        empty_rate = pct(page_empty[pid], n)
        low_rate = pct(page_low[pid], n)
        avg_top1 = pct(page_top1_sum[pid], n)
        score = empty_rate * 2.0 + low_rate + (args.low_top1_threshold - min(avg_top1, args.low_top1_threshold))
        flagged_pages.append(
            {
                "page_id": pid,
                "patches": n,
                "empty_rate": round(empty_rate, 6),
                "low_signal_rate": round(low_rate, 6),
                "avg_top1": round(avg_top1, 6),
                "flag_score": round(score, 6),
            }
        )
    flagged_pages.sort(key=lambda x: (x["flag_score"], x["empty_rate"], x["low_signal_rate"]), reverse=True)
    flagged_pages = flagged_pages[: max(args.show_flagged_pages, 0)]

    sampled_rows = []
    for pid in sorted(sampled_pages):
        n = page_patches[pid]
        row = {
            "page_id": pid,
            "patches": n,
            "empty_patches": page_empty[pid],
            "low_signal_patches": page_low[pid],
            "empty_rate": round(pct(page_empty[pid], n), 6),
            "low_signal_rate": round(pct(page_low[pid], n), 6),
            "avg_top1": round(pct(page_top1_sum[pid], n), 6),
            "top_concepts_by_freq": sample_page_concept_freq[pid].most_common(10),
            "top_concepts_by_weight": sorted(
                sample_page_concept_weight[pid].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
        if anchors:
            row["anchor_hits"] = dict(sample_page_anchor_hits[pid])
        sampled_rows.append(row)

    concepts_seen = {k.lower() for k in concept_patch_freq.keys()}
    vocab_size = len(vocab) if vocab is not None else None
    coverage = pct(len(concepts_seen), vocab_size) if vocab_size else None

    summary = {
        "files": len(files),
        "pages": len(all_pages),
        "patches": total_patches,
        "malformed_rows": malformed_rows,
        "empty_patches": empty_patches,
        "empty_patch_rate": round(pct(empty_patches, total_patches), 6),
        "low_signal_patches": low_signal_patches,
        "low_signal_patch_rate": round(pct(low_signal_patches, total_patches), 6),
        "nonempty_patches": nonempty_patches,
        "avg_top1_weight": round(pct(top1_sum, nonempty_patches), 6),
        "avg_concepts_per_nonempty_patch": round(pct(concepts_per_patch_sum, nonempty_patches), 6),
        "unique_concepts_seen": len(concepts_seen),
        "vocab_size": vocab_size,
        "concept_coverage": round(coverage, 6) if coverage is not None else None,
        "anchors_tracked": sorted(anchors),
        "anchor_patch_hits": dict(anchor_patch_hits),
        "top_concepts_by_patch_freq": top_by_freq,
        "top_concepts_by_weight_sum": [(k, round(v, 6)) for k, v in top_by_weight],
        "flagged_pages": flagged_pages,
        "sampled_pages": sampled_rows,
    }

    print("\n=== Patch Label Quality Audit ===")
    print(f"files: {summary['files']}")
    print(f"pages: {summary['pages']}")
    print(f"patches: {summary['patches']}")
    print(f"malformed_rows: {summary['malformed_rows']}")
    print(f"empty_patch_rate: {summary['empty_patch_rate']:.6f}")
    print(f"low_signal_patch_rate (top1 < {args.low_top1_threshold}): {summary['low_signal_patch_rate']:.6f}")
    print(f"avg_top1_weight: {summary['avg_top1_weight']:.6f}")
    print(f"avg_concepts_per_nonempty_patch: {summary['avg_concepts_per_nonempty_patch']:.6f}")
    print(f"unique_concepts_seen: {summary['unique_concepts_seen']}")
    if summary["vocab_size"] is not None:
        print(f"concept_coverage: {summary['concept_coverage']:.6f} ({summary['unique_concepts_seen']}/{summary['vocab_size']})")

    print("\nTop concepts by patch frequency:")
    for concept, cnt in summary["top_concepts_by_patch_freq"]:
        print(f"  {concept}: {cnt}")

    print("\nTop concepts by weight sum:")
    for concept, wsum in summary["top_concepts_by_weight_sum"]:
        print(f"  {concept}: {wsum:.6f}")

    if anchors:
        print("\nAnchor patch hits:")
        for concept in sorted(anchors):
            print(f"  {concept}: {summary['anchor_patch_hits'].get(concept, 0)}")

    if summary["flagged_pages"]:
        print(f"\nFlagged pages (top {len(summary['flagged_pages'])}):")
        for row in summary["flagged_pages"]:
            print(
                f"  {row['page_id']} | patches={row['patches']} "
                f"empty_rate={row['empty_rate']:.6f} low_rate={row['low_signal_rate']:.6f} avg_top1={row['avg_top1']:.6f}"
            )

    if args.output_summary_json:
        out = Path(args.output_summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nWrote summary: {out}")

    if args.output_sample_jsonl:
        out = Path(args.output_sample_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as handle:
            for row in sampled_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote sampled pages: {out}")

    if args.output_top_concepts_tsv:
        out = Path(args.output_top_concepts_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as handle:
            handle.write("rank\tconcept\tpatch_freq\tweight_sum\n")
            all_by_weight = dict(summary["top_concepts_by_weight_sum"])
            for rank, (concept, cnt) in enumerate(summary["top_concepts_by_patch_freq"], start=1):
                handle.write(
                    f"{rank}\t{concept}\t{cnt}\t{all_by_weight.get(concept, 0.0):.6f}\n"
                )
        print(f"Wrote top concepts TSV: {out}")


if __name__ == "__main__":
    main()
