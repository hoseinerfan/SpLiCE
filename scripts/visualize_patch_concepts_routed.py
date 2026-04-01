#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from PIL import Image
except ImportError:
    Image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize patch concepts using per-concept source routing."
    )
    parser.add_argument(
        "--labels",
        action="append",
        required=True,
        help="Source spec as name=/abs/path/to/labels.jsonl (repeatable).",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Route spec as concept=source_name (repeatable).",
    )
    parser.add_argument(
        "--route-file",
        type=str,
        default=None,
        help="Optional route file (TSV/CSV): concept<tab|,>source_name, one per line.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="",
        help="Optional concept list to visualize (comma-separated). Defaults to routed concepts.",
    )
    parser.add_argument("--page-id", type=str, required=True, help="Target page id (example: <docid>:0).")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=0, help="Patch grid size. If 0, inferred.")
    parser.add_argument("--image-token-start", type=int, default=0)
    parser.add_argument("--image-token-count", type=int, default=0)
    parser.add_argument("--top-patches", type=int, default=20)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Global minimum concept score for a patch to be considered a hit.",
    )
    parser.add_argument(
        "--concept-min-score",
        action="append",
        default=[],
        help="Per-concept threshold as concept=value (repeatable).",
    )
    parser.add_argument(
        "--concept-min-score-file",
        type=str,
        default=None,
        help="Optional thresholds file: concept<tab|,>value per line.",
    )
    parser.add_argument("--page-image", type=str, default=None, help="Optional page image path for overlays.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--cmap", type=str, default="magma")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--default-source",
        type=str,
        default="",
        help="Fallback source name if a concept has no explicit route.",
    )
    return parser.parse_args()


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")[:140] or "item"


def parse_name_path_specs(specs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --labels spec '{spec}'. Use name=path.")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = Path(path.strip())
        if not name:
            raise ValueError(f"Invalid --labels spec '{spec}': empty source name.")
        if not path.is_file():
            raise FileNotFoundError(f"Label file for source '{name}' not found: {path}")
        out[name] = path
    if not out:
        raise ValueError("No label sources provided.")
    return out


def parse_route_specs(route_specs: List[str]) -> Dict[str, str]:
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


def parse_route_file(path: Path) -> Dict[str, str]:
    route: Dict[str, str] = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                left, right = line.split("\t", 1)
            elif "," in line:
                left, right = line.split(",", 1)
            else:
                raise ValueError(f"Invalid route line (expected concept<tab|,>source): {line}")
            concept = left.strip().lower()
            src = right.strip()
            if concept and src:
                route[concept] = src
    return route


def parse_concept_threshold_specs(specs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --concept-min-score spec '{spec}'. Use concept=value.")
        concept, value = spec.split("=", 1)
        concept = concept.strip().lower()
        value = value.strip()
        if not concept:
            raise ValueError(f"Invalid --concept-min-score spec '{spec}': empty concept.")
        try:
            thr = float(value)
        except Exception:
            raise ValueError(f"Invalid threshold in spec '{spec}'.")
        out[concept] = thr
    return out


def parse_concept_threshold_file(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                left, right = line.split("\t", 1)
            elif "," in line:
                left, right = line.split(",", 1)
            else:
                raise ValueError(f"Invalid threshold line (expected concept<tab|,>value): {line}")
            concept = left.strip().lower()
            try:
                thr = float(right.strip())
            except Exception:
                raise ValueError(f"Invalid threshold value in line: {line}")
            if concept:
                out[concept] = thr
    return out


def parse_concepts(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def load_page_rows(labels_jsonl: Path, page_id: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(labels_jsonl, "r") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("page_id", "")) == page_id:
                rows.append(record)
    rows.sort(key=lambda x: int(x.get("patch_index", -1)))
    return rows


def infer_grid(
    n_patches: int,
    grid_size: int,
    image_token_start: int,
    image_token_count: int,
) -> Tuple[int, int]:
    if grid_size > 0:
        g = grid_size
        count = image_token_count if image_token_count > 0 else g * g
    elif image_token_count > 0:
        root = int(round(math.sqrt(image_token_count)))
        if root * root != image_token_count:
            raise ValueError(
                f"--image-token-count must be a perfect square when --grid-size is 0. Got {image_token_count}."
            )
        g = root
        count = image_token_count
    else:
        g = int(math.floor(math.sqrt(n_patches)))
        if g <= 0:
            raise ValueError(f"Cannot infer grid from n_patches={n_patches}")
        count = g * g

    if image_token_start < 0:
        raise ValueError("--image-token-start must be >= 0.")
    if image_token_start + count > n_patches:
        raise ValueError(
            f"Grid token range [{image_token_start}, {image_token_start + count}) exceeds n_patches={n_patches}."
        )
    return g, count


def build_patch_score_map(rows: List[Dict]) -> Dict[int, Dict[str, float]]:
    patch_to_scores: Dict[int, Dict[str, float]] = {}
    for row in rows:
        patch_idx = int(row["patch_index"])
        concept_scores: Dict[str, float] = {}
        for item in row.get("top_concepts", []):
            concept = str(item.get("concept", "")).lower()
            weight = float(item.get("weight", 0.0))
            if concept:
                concept_scores[concept] = weight
        patch_to_scores[patch_idx] = concept_scores
    return patch_to_scores


def concept_grid_and_hits(
    concept: str,
    patch_to_scores: Dict[int, Dict[str, float]],
    grid_size: int,
    image_token_start: int,
    image_token_count: int,
    min_score: float,
) -> Tuple["np.ndarray", List[Tuple[float, int, int, int]]]:
    assert np is not None
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    hits: List[Tuple[float, int, int, int]] = []
    for patch_idx in range(image_token_start, image_token_start + image_token_count):
        score = float(patch_to_scores.get(patch_idx, {}).get(concept, 0.0))
        if score < min_score:
            continue
        rel = patch_idx - image_token_start
        row = rel // grid_size
        col = rel % grid_size
        grid[row, col] = score
        hits.append((score, patch_idx, row, col))
    hits.sort(reverse=True, key=lambda x: x[0])
    return grid, hits


def save_single_heatmap(
    out_path: Path,
    grid: "np.ndarray",
    title: str,
    cmap: str,
    dpi: int,
    top_hits: List[Tuple[float, int, int, int]],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    for score, patch_idx, row, col in top_hits[:5]:
        ax.scatter([col], [row], s=30, c="cyan", edgecolors="black", linewidths=0.5)
        ax.text(col + 0.2, row + 0.2, f"{patch_idx}", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_overlay_heatmap(
    out_path: Path,
    page_image_path: Path,
    grid: "np.ndarray",
    title: str,
    cmap: str,
    dpi: int,
    alpha: float,
    top_hits: List[Tuple[float, int, int, int]],
    grid_size: int,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for --page-image overlays. Install with: pip install pillow")
    image = Image.open(page_image_path).convert("RGB")
    image_np = np.asarray(image)
    h, w = image_np.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    im = ax.imshow(grid, cmap=cmap, alpha=alpha, extent=(0, w, h, 0), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    for score, patch_idx, row, col in top_hits[:10]:
        cx = (col + 0.5) * (w / grid_size)
        cy = (row + 0.5) * (h / grid_size)
        ax.scatter([cx], [cy], s=20, c="cyan", edgecolors="black", linewidths=0.5)
        ax.text(cx + 3, cy + 3, f"{patch_idx}", color="white", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_combined_panel(
    out_path: Path,
    concepts: List[str],
    labels: Dict[str, str],
    concept_to_grid: Dict[str, "np.ndarray"],
    cmap: str,
    dpi: int,
) -> None:
    n = len(concepts)
    n_cols = min(3, n)
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.8 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, concept in enumerate(concepts):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        grid = concept_to_grid[concept]
        im = ax.imshow(grid, cmap=cmap)
        ax.set_title(f"{concept} [{labels[concept]}]\nmax={float(grid.max()):.3f}")
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if np is None:
        raise RuntimeError("numpy is required. Install with: pip install numpy")
    if plt is None:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")

    source_to_labels = parse_name_path_specs(args.labels)
    route = parse_route_specs(args.route)
    if args.route_file:
        route.update(parse_route_file(Path(args.route_file)))
    if not route:
        raise ValueError("No routes provided. Use --route and/or --route-file.")

    concept_thresholds = parse_concept_threshold_specs(args.concept_min_score)
    if args.concept_min_score_file:
        concept_thresholds.update(parse_concept_threshold_file(Path(args.concept_min_score_file)))

    if args.concepts:
        concepts = parse_concepts(args.concepts)
    else:
        concepts = sorted(route.keys())

    concept_to_source: Dict[str, str] = {}
    for concept in concepts:
        src = route.get(concept)
        if src is None:
            if args.default_source:
                src = args.default_source
            else:
                raise KeyError(f"No source route for concept '{concept}'.")
        if src not in source_to_labels:
            raise KeyError(f"Source '{src}' for concept '{concept}' not found in --labels map.")
        concept_to_source[concept] = src

    # Load page rows once per used source.
    used_sources = sorted(set(concept_to_source.values()))
    source_rows: Dict[str, List[Dict]] = {}
    source_patch_scores: Dict[str, Dict[int, Dict[str, float]]] = {}
    n_patches = -1
    for src in used_sources:
        rows = load_page_rows(source_to_labels[src], args.page_id)
        if not rows:
            raise ValueError(f"No rows found for page-id '{args.page_id}' in source '{src}' file {source_to_labels[src]}")
        source_rows[src] = rows
        source_patch_scores[src] = build_patch_score_map(rows)
        max_patch = max(int(r["patch_index"]) for r in rows)
        n_patches = max(n_patches, max_patch + 1)

    if n_patches <= 0:
        raise ValueError(f"Could not infer patch count for page '{args.page_id}'.")

    grid_size, image_token_count = infer_grid(
        n_patches=n_patches,
        grid_size=args.grid_size,
        image_token_start=args.image_token_start,
        image_token_count=args.image_token_count,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_image_path = Path(args.page_image) if args.page_image else None
    if page_image_path is not None and not page_image_path.is_file():
        raise FileNotFoundError(f"page-image not found: {page_image_path}")

    concept_to_grid: Dict[str, "np.ndarray"] = {}
    report: Dict[str, object] = {
        "page_id": args.page_id,
        "sources": {k: str(v) for k, v in source_to_labels.items()},
        "global_min_score": float(args.min_score),
        "concept_min_scores": concept_thresholds,
        "n_patches": n_patches,
        "grid_size": grid_size,
        "image_token_start": args.image_token_start,
        "image_token_count": image_token_count,
        "concepts": [],
    }

    for concept in concepts:
        src = concept_to_source[concept]
        thr = float(concept_thresholds.get(concept, args.min_score))
        grid, hits = concept_grid_and_hits(
            concept=concept,
            patch_to_scores=source_patch_scores[src],
            grid_size=grid_size,
            image_token_start=args.image_token_start,
            image_token_count=image_token_count,
            min_score=thr,
        )
        concept_to_grid[concept] = grid
        top_hits = hits[: args.top_patches]

        report["concepts"].append(
            {
                "concept": concept,
                "source": src,
                "min_score": thr,
                "max_weight": float(grid.max()) if hits else 0.0,
                "nonzero_patches": int(np.count_nonzero(grid)),
                "top_hits": [
                    {
                        "patch_index": int(patch_idx),
                        "row": int(row),
                        "col": int(col),
                        "weight": float(score),
                    }
                    for score, patch_idx, row, col in top_hits
                ],
            }
        )

        safe_concept = sanitize_filename(concept)
        safe_src = sanitize_filename(src)
        base = f"{sanitize_filename(args.page_id)}__{safe_concept}__src_{safe_src}"
        title = f"{args.page_id}\n{concept} [{src}] | max={float(grid.max()):.4f}"

        heatmap_path = out_dir / f"{base}__heatmap.png"
        save_single_heatmap(
            out_path=heatmap_path,
            grid=grid,
            title=title,
            cmap=args.cmap,
            dpi=args.dpi,
            top_hits=top_hits,
        )

        if page_image_path is not None:
            overlay_path = out_dir / f"{base}__overlay.png"
            save_overlay_heatmap(
                out_path=overlay_path,
                page_image_path=page_image_path,
                grid=grid,
                title=f"{args.page_id}\n{concept} [{src}] overlay",
                cmap=args.cmap,
                dpi=args.dpi,
                alpha=args.overlay_alpha,
                top_hits=top_hits,
                grid_size=grid_size,
            )

    panel_path = out_dir / f"{sanitize_filename(args.page_id)}__combined_panel_routed.png"
    save_combined_panel(
        out_path=panel_path,
        concepts=concepts,
        labels=concept_to_source,
        concept_to_grid=concept_to_grid,
        cmap=args.cmap,
        dpi=args.dpi,
    )

    summary_path = out_dir / f"{sanitize_filename(args.page_id)}__summary_routed.json"
    with open(summary_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    hits_tsv_path = out_dir / f"{sanitize_filename(args.page_id)}__top_hits_routed.tsv"
    with open(hits_tsv_path, "w") as handle:
        handle.write("concept\tsource\tpatch_index\trow\tcol\tweight\n")
        for item in report["concepts"]:
            concept = item["concept"]
            source = item["source"]
            for hit in item["top_hits"]:
                handle.write(
                    f"{concept}\t{source}\t{hit['patch_index']}\t{hit['row']}\t{hit['col']}\t{hit['weight']:.6f}\n"
                )

    print(f"Page: {args.page_id}")
    print(f"Patches: {n_patches}")
    print(f"Grid: {grid_size}x{grid_size} (start={args.image_token_start}, count={image_token_count})")
    print(f"Sources used: {used_sources}")
    print(f"Wrote outputs to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Top hits TSV: {hits_tsv_path}")


if __name__ == "__main__":
    main()
