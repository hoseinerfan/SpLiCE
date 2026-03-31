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

## 3) Label precomputed page/document embeddings

Run concept labeling on your page embeddings:

```bash
python scripts/label_precomputed_embeddings.py \
  --embeddings-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev \
  --dictionary-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_dictionary.pt \
  --vocab-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts.txt \
  --output-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali-v1.2_m3-docvqa_dev_labels.jsonl \
  --layout single \
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

## 4) Label query embeddings generated at runtime

When your runtime ColPali pipeline writes query embeddings to disk, run:

```bash
python scripts/label_precomputed_embeddings.py \
  --embeddings-path /path/to/runtime_query_embeddings.pt \
  --dictionary-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concept_dictionary.pt \
  --vocab-path /mmfs1/scratch/jacks.local/aerfanshekooh/custom/embeddings/colpali_concepts.txt \
  --output-jsonl /path/to/runtime_query_labels.jsonl \
  --layout batch \
  --topk 10 \
  --l1-penalty 0.25 \
  --device cuda
```

## 5) Output format

Each line in output JSONL contains:
- `id`
- `top_concepts` (list of `{concept, weight}`)
- `l0_norm`
- `cosine_similarity`
