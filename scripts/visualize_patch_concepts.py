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
        description="Visualize concept scores over page patches as heatmaps."
    )
    parser.add_argument("--labels-jsonl", type=str, required=True, help="Patch-label JSONL file.")
    parser.add_argument("--page-id", type=str, required=True, help="Target page id (example: <docid>:0).")
    parser.add_argument("--concepts", type=str, required=True, help="Comma-separated concept list.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write heatmaps and reports.")
    parser.add_argument("--grid-size", type=int, default=0, help="Patch grid size per side. If 0, inferred.")
    parser.add_argument(
        "--image-token-start",
        type=int,
        default=0,
        help="Starting patch index used for image grid mapping.",
    )
    parser.add_argument(
        "--image-token-count",
        type=int,
        default=0,
        help="Number of patch tokens used in image grid. If 0, uses grid-size^2.",
    )
    parser.add_argument("--top-patches", type=int, default=10, help="Top patches to report per concept.")
    parser.add_argument("--page-image", type=str, default=None, help="Optional page image path for overlay.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Heatmap alpha for image overlay.")
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    return parser.parse_args()


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")[:120] or "concept"


def parse_concepts(raw: str) -> List[str]:
    concepts = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not concepts:
        raise ValueError("No concepts provided.")
    return concepts


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
) -> Tuple["np.ndarray", List[Tuple[float, int, int, int]]]:
    assert np is not None
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    hits: List[Tuple[float, int, int, int]] = []
    for patch_idx in range(image_token_start, image_token_start + image_token_count):
        score = float(patch_to_scores.get(patch_idx, {}).get(concept, 0.0))
        if score <= 0:
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
    page_id: str,
    concept: str,
    cmap: str,
    dpi: int,
    top_hits: List[Tuple[float, int, int, int]],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap=cmap)
    ax.set_title(f"{page_id}\n{concept} | max={float(grid.max()):.4f}")
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
    page_id: str,
    concept: str,
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
    im = ax.imshow(
        grid,
        cmap=cmap,
        alpha=alpha,
        extent=(0, w, h, 0),
        interpolation="nearest",
    )
    ax.set_title(f"{page_id}\n{concept} overlay")
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
    page_id: str,
    concepts: List[str],
    concept_to_grid: Dict[str, "np.ndarray"],
    cmap: str,
    dpi: int,
) -> None:
    n = len(concepts)
    n_cols = min(3, n)
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
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
        ax.set_title(f"{concept}\nmax={float(grid.max()):.3f}")
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].axis("off")

    fig.suptitle(page_id)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if np is None:
        raise RuntimeError("numpy is required. Install with: pip install numpy")
    if plt is None:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")

    labels_jsonl = Path(args.labels_jsonl)
    if not labels_jsonl.is_file():
        raise FileNotFoundError(f"labels-jsonl not found: {labels_jsonl}")

    concepts = parse_concepts(args.concepts)
    rows = load_page_rows(labels_jsonl=labels_jsonl, page_id=args.page_id)
    if not rows:
        raise ValueError(f"No rows found for page-id '{args.page_id}' in {labels_jsonl}")

    max_patch = max(int(r["patch_index"]) for r in rows)
    n_patches = max_patch + 1
    grid_size, image_token_count = infer_grid(
        n_patches=n_patches,
        grid_size=args.grid_size,
        image_token_start=args.image_token_start,
        image_token_count=args.image_token_count,
    )

    patch_to_scores = build_patch_score_map(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_to_grid: Dict[str, "np.ndarray"] = {}
    report: Dict[str, object] = {
        "labels_jsonl": str(labels_jsonl),
        "page_id": args.page_id,
        "n_rows": len(rows),
        "n_patches": n_patches,
        "grid_size": grid_size,
        "image_token_start": args.image_token_start,
        "image_token_count": image_token_count,
        "concepts": [],
    }

    page_image_path = Path(args.page_image) if args.page_image else None
    if page_image_path is not None and not page_image_path.is_file():
        raise FileNotFoundError(f"page-image not found: {page_image_path}")

    for concept in concepts:
        grid, hits = concept_grid_and_hits(
            concept=concept,
            patch_to_scores=patch_to_scores,
            grid_size=grid_size,
            image_token_start=args.image_token_start,
            image_token_count=image_token_count,
        )
        concept_to_grid[concept] = grid

        top_hits = hits[: args.top_patches]
        concept_rec = {
            "concept": concept,
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
        report["concepts"].append(concept_rec)

        safe = sanitize_filename(concept)
        heatmap_path = out_dir / f"{sanitize_filename(args.page_id)}__{safe}__heatmap.png"
        save_single_heatmap(
            out_path=heatmap_path,
            grid=grid,
            page_id=args.page_id,
            concept=concept,
            cmap=args.cmap,
            dpi=args.dpi,
            top_hits=top_hits,
        )

        if page_image_path is not None:
            overlay_path = out_dir / f"{sanitize_filename(args.page_id)}__{safe}__overlay.png"
            save_overlay_heatmap(
                out_path=overlay_path,
                page_image_path=page_image_path,
                grid=grid,
                page_id=args.page_id,
                concept=concept,
                cmap=args.cmap,
                dpi=args.dpi,
                alpha=args.overlay_alpha,
                top_hits=top_hits,
                grid_size=grid_size,
            )

    panel_path = out_dir / f"{sanitize_filename(args.page_id)}__combined_panel.png"
    save_combined_panel(
        out_path=panel_path,
        page_id=args.page_id,
        concepts=concepts,
        concept_to_grid=concept_to_grid,
        cmap=args.cmap,
        dpi=args.dpi,
    )

    summary_path = out_dir / f"{sanitize_filename(args.page_id)}__summary.json"
    with open(summary_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    hits_tsv_path = out_dir / f"{sanitize_filename(args.page_id)}__top_hits.tsv"
    with open(hits_tsv_path, "w") as handle:
        handle.write("concept\tpatch_index\trow\tcol\tweight\n")
        for item in report["concepts"]:
            concept = item["concept"]
            for hit in item["top_hits"]:
                handle.write(
                    f"{concept}\t{hit['patch_index']}\t{hit['row']}\t{hit['col']}\t{hit['weight']:.6f}\n"
                )

    print(f"Page: {args.page_id}")
    print(f"Patches: {n_patches}")
    print(f"Grid: {grid_size}x{grid_size} (start={args.image_token_start}, count={image_token_count})")
    print(f"Wrote outputs to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Top hits TSV: {hits_tsv_path}")


if __name__ == "__main__":
    main()
