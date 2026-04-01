#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a concept->source routing policy over patch-label sources on sampled pages."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source spec as name=/path/to/labels.jsonl_or_dir (repeatable).",
    )
    parser.add_argument(
        "--route",
        action="append",
        required=True,
        help="Route spec as concept=source_name (repeatable).",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="",
        help="Optional comma-separated concepts to evaluate. Defaults to all routed concepts.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan source directories for *.jsonl.")
    parser.add_argument(
        "--doc-ids-file",
        type=str,
        default=None,
        help="Optional doc id allowlist (one doc id per line) to limit processed files in directory sources.",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=20,
        help="Number of pages to sample for per-page diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.0,
        help="Concept considered active on a page if any source max weight > threshold.",
    )
    parser.add_argument("--print-top-failures", type=int, default=10, help="Worst pages per concept to print.")
    parser.add_argument("--output-summary-json", type=str, default=None)
    parser.add_argument("--output-sampled-pages-jsonl", type=str, default=None)
    return parser.parse_args()


def parse_name_path_specs(specs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --source spec '{spec}'. Use name=path.")
        name, path = spec.split("=", 1)
        name = name.strip()
        p = Path(path.strip())
        if not name:
            raise ValueError(f"Invalid --source spec '{spec}': empty name.")
        if not p.exists():
            raise FileNotFoundError(f"Source path for '{name}' not found: {p}")
        out[name] = p
    return out


def parse_routes(route_specs: List[str]) -> Dict[str, str]:
    route: Dict[str, str] = {}
    for spec in route_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --route spec '{spec}'. Use concept=source.")
        concept, src = spec.split("=", 1)
        concept = concept.strip().lower()
        src = src.strip()
        if not concept or not src:
            raise ValueError(f"Invalid --route spec '{spec}'.")
        route[concept] = src
    return route


def parse_concepts(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def read_doc_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    out: Set[str] = set()
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.add(line)
    return out


def list_jsonl_files(path: Path, recursive: bool, doc_ids: Optional[Set[str]]) -> List[Path]:
    if path.is_file():
        return [path]

    if recursive:
        files = sorted([p for p in path.rglob("*.jsonl") if p.is_file()])
    else:
        files = sorted([p for p in path.glob("*.jsonl") if p.is_file()])

    if doc_ids is not None:
        keep = []
        for p in files:
            doc = p.stem
            if doc in doc_ids:
                keep.append(p)
        files = keep
    return files


def infer_page_id(row: Dict) -> str:
    page_id = str(row.get("page_id", "")).strip()
    if page_id:
        return page_id
    rid = str(row.get("id", "")).strip()
    if "#patch" in rid:
        return rid.split("#patch", 1)[0]
    return rid if rid else "unknown_page"


def parse_top_concepts(row: Dict) -> List[Tuple[str, float]]:
    raw = row.get("top_concepts", [])
    if not isinstance(raw, list):
        return []
    out: List[Tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept", "")).strip().lower()
        if not concept:
            continue
        try:
            weight = float(item.get("weight", 0.0))
        except Exception:
            weight = 0.0
        out.append((concept, weight))
    return out


def pct(n: float, d: float) -> float:
    return n / d if d else 0.0


def main() -> None:
    args = parse_args()

    source_to_path = parse_name_path_specs(args.source)
    route = parse_routes(args.route)
    doc_ids = read_doc_ids(args.doc_ids_file)

    if args.concepts:
        concepts = parse_concepts(args.concepts)
    else:
        concepts = sorted(route.keys())

    # Validate routes
    for concept in concepts:
        if concept not in route:
            raise KeyError(f"No route for concept '{concept}'.")
        src = route[concept]
        if src not in source_to_path:
            raise KeyError(f"Route for concept '{concept}' points to unknown source '{src}'.")

    # source -> files
    source_files: Dict[str, List[Path]] = {}
    for src, path in source_to_path.items():
        files = list_jsonl_files(path, args.recursive, doc_ids)
        if not files:
            raise ValueError(f"No files found for source '{src}' at {path}")
        source_files[src] = files

    # source -> page -> concept -> max_weight
    source_page_concept_max: Dict[str, Dict[str, Dict[str, float]]] = {}
    source_pages: Dict[str, Set[str]] = {}

    for src, files in source_files.items():
        page_concept = defaultdict(lambda: defaultdict(float))
        pages = set()
        for f in files:
            with open(f, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    page_id = infer_page_id(row)
                    pages.add(page_id)
                    for concept, weight in parse_top_concepts(row):
                        if concept not in concepts:
                            continue
                        if weight > page_concept[page_id][concept]:
                            page_concept[page_id][concept] = weight
        source_page_concept_max[src] = page_concept
        source_pages[src] = pages

    all_pages = sorted(set().union(*source_pages.values()))
    if not all_pages:
        raise ValueError("No pages found across sources.")

    rng = random.Random(args.seed)
    sample_n = min(max(args.sample_pages, 0), len(all_pages))
    sampled_pages = sorted(rng.sample(all_pages, sample_n)) if sample_n > 0 else []

    per_concept_summary = {}
    sampled_rows = []

    for concept in concepts:
        routed_src = route[concept]

        active_pages = []
        routed_nonzero = 0
        routed_best = 0
        routed_scores = []
        best_scores = []
        margins = []
        failures = []  # lower margin is worse

        for page_id in all_pages:
            src_scores = {}
            for src in source_to_path:
                src_scores[src] = float(source_page_concept_max[src].get(page_id, {}).get(concept, 0.0))

            best_src = max(src_scores, key=lambda s: src_scores[s])
            best_score = src_scores[best_src]
            routed_score = src_scores[routed_src]

            if best_score > args.active_threshold:
                active_pages.append(page_id)
                routed_scores.append(routed_score)
                best_scores.append(best_score)
                margins.append(routed_score - best_score)
                if routed_score > 0:
                    routed_nonzero += 1
                if routed_src == best_src:
                    routed_best += 1
                else:
                    failures.append(
                        {
                            "page_id": page_id,
                            "routed_source": routed_src,
                            "best_source": best_src,
                            "routed_score": round(routed_score, 6),
                            "best_score": round(best_score, 6),
                            "margin": round(routed_score - best_score, 6),
                            "source_scores": {k: round(v, 6) for k, v in src_scores.items()},
                        }
                    )

        failures.sort(key=lambda x: x["margin"])  # most negative first

        summary = {
            "concept": concept,
            "routed_source": routed_src,
            "all_pages": len(all_pages),
            "active_pages": len(active_pages),
            "active_rate": round(pct(len(active_pages), len(all_pages)), 6),
            "routed_nonzero_on_active": routed_nonzero,
            "routed_nonzero_rate_on_active": round(pct(routed_nonzero, len(active_pages)), 6),
            "routed_best_on_active": routed_best,
            "routed_best_rate_on_active": round(pct(routed_best, len(active_pages)), 6),
            "avg_routed_score_on_active": round(sum(routed_scores) / max(len(routed_scores), 1), 6),
            "avg_best_score_on_active": round(sum(best_scores) / max(len(best_scores), 1), 6),
            "avg_margin_on_active": round(sum(margins) / max(len(margins), 1), 6),
            "worst_failures": failures[: max(args.print_top_failures, 0)],
        }
        per_concept_summary[concept] = summary

    # sampled page diagnostics
    for page_id in sampled_pages:
        rec = {"page_id": page_id, "concepts": []}
        for concept in concepts:
            routed_src = route[concept]
            src_scores = {
                src: float(source_page_concept_max[src].get(page_id, {}).get(concept, 0.0))
                for src in source_to_path
            }
            best_src = max(src_scores, key=lambda s: src_scores[s])
            rec["concepts"].append(
                {
                    "concept": concept,
                    "routed_source": routed_src,
                    "best_source": best_src,
                    "routed_score": round(src_scores[routed_src], 6),
                    "best_score": round(src_scores[best_src], 6),
                    "source_scores": {k: round(v, 6) for k, v in src_scores.items()},
                }
            )
        sampled_rows.append(rec)

    print("=== Routed Policy Evaluation ===")
    print(f"sources: {list(source_to_path.keys())}")
    print(f"concepts: {concepts}")
    print(f"pages_total: {len(all_pages)}")
    print(f"sampled_pages: {len(sampled_pages)}")
    print(f"active_threshold: {args.active_threshold}")
    print("")

    for concept in concepts:
        s = per_concept_summary[concept]
        print(f"[{concept}] route->{s['routed_source']}")
        print(
            f"  active_pages={s['active_pages']} ({s['active_rate']:.3f}), "
            f"best_rate={s['routed_best_rate_on_active']:.3f}, "
            f"nonzero_rate={s['routed_nonzero_rate_on_active']:.3f}, "
            f"avg_margin={s['avg_margin_on_active']:.6f}"
        )
        if s["worst_failures"]:
            w = s["worst_failures"][0]
            print(
                f"  worst_failure: page={w['page_id']} best={w['best_source']}({w['best_score']}) "
                f"routed={w['routed_source']}({w['routed_score']}) margin={w['margin']}"
            )
        print("")

    output = {
        "sources": {k: str(v) for k, v in source_to_path.items()},
        "routes": route,
        "concepts": concepts,
        "pages_total": len(all_pages),
        "active_threshold": args.active_threshold,
        "per_concept_summary": per_concept_summary,
        "sampled_pages": sampled_rows,
    }

    if args.output_summary_json:
        out = Path(args.output_summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as handle:
            json.dump(output, handle, indent=2)
        print(f"Wrote summary: {out}")

    if args.output_sampled_pages_jsonl:
        out = Path(args.output_sampled_pages_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as handle:
            for row in sampled_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote sampled pages: {out}")


if __name__ == "__main__":
    main()
