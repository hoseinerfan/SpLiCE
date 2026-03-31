# HPC Workflow (MMQA + ColPali Embeddings)

This workflow is for running SpLiCE-style concept labeling on precomputed embeddings in an HPC environment.

## 1) Pull and install on HPC

```bash
git clone <YOUR_GITHUB_REPO_URL> SpLiCE
cd SpLiCE
pip install -e .
```

Optional sanity check for embedding stores:

```bash
python scripts/inspect_embedding_store.py \
  --embeddings-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev \
  --sample-files 10
```

## 2) Prepare MMQA queries for runtime embedding

Extract query IDs/text from your MMQA dev file:

```bash
python scripts/filter_mmqa_queries.py \
  --input-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl \
  --output-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev_queries_filtered.jsonl \
  --dedupe-text
```

Notes:
- If your ID field is not automatically detected, add `--id-field <field_name>`.
- If your query text field is not automatically detected, add `--query-field <field_name>`.
- If you have a keep list of query IDs, add `--allowlist-file <path>`.

## 3) Build concept dictionary files (required once)

SpLiCE labeling needs:
- `colpali_concepts.txt`
- `colpali_concept_dictionary.pt`

### 3a) Build concept vocabulary from filtered queries

```bash
python scripts/build_concept_vocab_from_queries.py \
  --input-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/data/m3-docvqa/multimodalqa/MMQA_dev_queries_filtered.jsonl \
  --output-concepts-txt /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts.txt \
  --output-counts-tsv /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts_counts.tsv \
  --top-unigrams 10000 \
  --top-bigrams 5000
```

### 3b) Convert concepts to query JSONL for your ColPali runtime embedder

```bash
python scripts/concepts_to_queries_jsonl.py \
  --concepts-txt /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts.txt \
  --output-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_queries.jsonl
```

### 3c) Run your existing ColPali query-embedding job on concept queries

Use either your existing runtime embedding pipeline, or the helper script:

```bash
python scripts/embed_colpali_queries.py \
  --input-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_queries.jsonl \
  --output-dir /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_query_embeddings \
  --model-name vidore/colpali-v1.2 \
  --backend auto \
  --query-id-field query_id \
  --query-text-field query_text \
  --batch-size 64 \
  --dtype bfloat16 \
  --device cuda:0 \
  --skip-existing
```

Expected output:
- one embedding file per concept query ID (same ID from JSONL), `.safetensors`/`.pt`/`.npy`
- each file contains a query embedding tensor compatible with your page embedding space (`dim=128`)

### 3d) Build dictionary tensor from concept embeddings

```bash
python scripts/build_dictionary_from_concept_embeddings.py \
  --embeddings-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_query_embeddings \
  --concept-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_queries.jsonl \
  --output-dictionary-pt /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_dictionary.pt \
  --output-vocab-txt /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts.txt \
  --recursive \
  --normalize \
  --expected-dim 128 \
  --summary-json /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_dictionary_summary.json
```

## 4) Label precomputed page/document embeddings

Run concept labeling on your page embeddings:

```bash
python scripts/label_precomputed_embeddings.py \
  --embeddings-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev \
  --dictionary-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_dictionary.pt \
  --vocab-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts.txt \
  --output-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev_labels.jsonl \
  --layout batch \
  --recursive \
  --topk 10 \
  --l1-penalty 0.25 \
  --device cuda
```

Important:
- `--dictionary-path` must be a tensor of shape `[num_concepts, embedding_dim]`.
- `--vocab-path` must have one concept string per line with exactly `num_concepts` lines.
- Embedding dimension must match the dictionary dimension.
- If `--mean-path` is omitted, the script estimates mean from your embeddings.
- `.safetensors` embeddings are supported directly.
- For ColPali page stores with tensor shape `[num_pages, 1030, 128]`, use `--layout batch` to label each page.

## 5) Label query embeddings generated at runtime

When your runtime ColPali pipeline writes query embeddings to disk, run:

```bash
python scripts/label_precomputed_embeddings.py \
  --embeddings-path /path/to/runtime_query_embeddings_dir \
  --dictionary-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_dictionary.pt \
  --vocab-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts.txt \
  --output-jsonl /path/to/runtime_query_labels.jsonl \
  --layout single \
  --recursive \
  --topk 10 \
  --l1-penalty 0.25 \
  --device cuda
```

For one-file-per-query stores, use `--layout single` and keep filenames as query IDs.

## 6) Output format

Each line in output JSONL contains:
- `id`
- `top_concepts` (list of `{concept, weight}`)
- `l0_norm`
- `cosine_similarity`

## 7) Build concept-overlap retrieval rankings

Once you have both:
- page labels JSONL (e.g., `colpali-v1.2_m3-docvqa_dev_labels.jsonl`)
- query labels JSONL (e.g., `MMQA_dev_query_labels.jsonl`)

generate a retrieval-ready ranking file:

```bash
python scripts/concept_overlap_retrieval.py \
  --query-labels-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/MMQA_dev_query_labels.jsonl \
  --page-labels-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev_labels.jsonl \
  --output-ranking-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/MMQA_dev_concept_overlap_rankings.jsonl \
  --topk-pages 100 \
  --max-shared-concepts 5 \
  --min-score 0.0
```

Each output row contains:
- `query_id`
- `query_top_concepts`
- `top_pages`: list of `{page_id, score, shared_concepts}`
- `confidence`: `high` or `low`
- `fallback_recommended`: boolean
- `fallback_reasons`: list of heuristic trigger labels
