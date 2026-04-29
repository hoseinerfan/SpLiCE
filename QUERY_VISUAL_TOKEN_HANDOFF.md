# Query Visual Token Handoff

Date: 2026-04-28

## Scope

This note captures the current state of the query-token visual/non-visual labeling module used in SpLiCE, including:

- the strict token-labeling logic
- the lexicon-expansion logic
- the evaluation protocol
- real example rows chosen for discussion
- the cluster output files currently being used
- the relationship between the label exports and the later page-query plots

## Main reported result

The user previously reported the following advisor-facing summary for query-token classification:

- Early strict setup:
  - IG + Occlusion attribution signals
  - curated visual lexicon
  - normalization and cleanup
  - dev coverage on gold visual queries: `66.28%`
- After expanding and cleaning the visual lexicon using attribution-supported tokens mined from visual queries in train:
  - dev coverage on gold visual queries: `90.85%`

Main conclusion:

- the main gain came from improving the visual lexicon rather than changing the attribution method itself

## Important file choices

There are multiple exported query-label files on cluster:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v2.jsonl`
- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v6_fulltrainlex_v2.jsonl`
- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v7_phraseaug.jsonl`

Working decision from this chat:

- use `union_relaxed_v2` for the real examples below, because that is the file later consumed by the page-query patch-dot plotting workflow

Important nuance:

- `union_relaxed_v2` is the chosen file for the examples and plot alignment in this handoff
- it is not necessarily the same file that should be treated as the final lexicon-expanded strict result behind the `90.85%` number
- if another chat needs the most likely strict lexicon-expanded export to inspect directly, start with:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v6_fulltrainlex_v2.jsonl`

## `union_relaxed_v2` provenance

The following parts of `union_relaxed_v2` were recovered exactly from the saved JSONL metadata and later cluster inspection:

- attribution file:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/attribution/dev_attr_visual_needed.jsonl`
- context files:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/context/context_ig_visual_needed.json`
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/context/context_occ_visual_needed.json`
- methods used:
  - `ig`, `occlusion`
- `require_method_overlap = False`
- `require_phrase_source_overlap = False`

What is not serialized in the output:

- the exact runtime visual-lexicon path or contents

However, the cluster evidence now strongly constrains the lexicon interpretation:

- `union_relaxed_v2` was created on `2026-04-20`
- the explicitly named `subsetlex` variants were created on `2026-04-22`
- the explicitly named `fulltrainlex` variants were created on `2026-04-23`
- the observed LGBT example rules out later lexicons that contain `lgbt` as a visual term, because `lgbt` was in `important_token_overlap` but was not labeled visual
- a later probe test over discriminating tokens returned `matches = 0`, which is most consistent with the conservative built-in exporter default rather than the larger older 321-token external lexicon family

Best current conclusion:

- the exact runtime lexicon path for `union_relaxed_v2` was not preserved
- behaviorally, `union_relaxed_v2` matches the exporter's conservative built-in default visual lexicon
- it should not be described as one of the later `subsetlex`, `fulltrainlex`, or phrase-augmentation variants

## Core code pointers

Primary exporter:

- [scripts/export_query_visual_binary_labels.py](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/export_query_visual_binary_labels.py)

Lexicon mining:

- [scripts/mine_visual_lexicon_from_attribution.py](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/mine_visual_lexicon_from_attribution.py)

Useful related scripts:

- [scripts/export_query_token_visual_labels.py](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/export_query_token_visual_labels.py)
- [scripts/export_visual_needed_merged_real.py](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/export_visual_needed_merged_real.py)
- [scripts/export_visual_needed_merged_recall.py](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/export_visual_needed_merged_recall.py)

## Built-in default visual lexicon

The exact built-in default lexicon in the exporter is:

- `logo`
- `poster`
- `image`
- `photo`
- `picture`
- `face`
- `hair`
- `beard`
- `mustache`
- `glasses`
- `eyeglasses`
- `wear`
- `wearing`
- `wears`
- `holding`
- `holds`
- `color`
- `colour`
- `shape`
- `symbol`
- `flag`
- `jersey`
- `bald`
- `sideburns`
- `visual`

Source:

- [scripts/export_query_visual_binary_labels.py:124](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/export_query_visual_binary_labels.py:124)

## Strict query-token pseudo-algorithm

This is the code-faithful version of the strict method.

```text
For each query q:
  1. Extract top positive attribution tokens from IG.
  2. Extract top positive attribution tokens from Occlusion.
  3. Normalize each attribution token by:
       - removing tokenizer prefixes such as ## / ▁ / Ġ
       - lowercasing
       - removing surrounding punctuation
  4. Filter attribution tokens by:
       - minimum token length
       - stopword removal
       - artifact-token removal
  5. Form the influential attribution set:
       Influential(q) = union or intersection of method token sets
     In the stronger setup described by the user, this was treated as a union.
  6. Tokenize the query text itself and apply the same token normalization
     function to each query token.
  7. For each normalized query token t:
       if t is in Influential(q)
          and t matches the visual lexicon:
             label(t) = visual_needed
       else:
             label(t) = non_visual_needed
  8. Return:
       - token_labels
       - visual_token_indices
       - non_visual_token_indices
```

### Normalization nuance

The same core normalization function is applied to both attribution tokens and query tokens:

- [scripts/export_query_visual_binary_labels.py:247](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/export_query_visual_binary_labels.py:247)

However:

- attribution tokens are normalized and then filtered for length, stopwords, and artifacts before they are used to build the influential set
- query tokens are normalized too, but they are not removed from the query-token sequence; instead the normalized token is checked against:
  - the already filtered influential set
  - the visual lexicon

So the comparison in the strict rule is essentially:

```text
normalized query token ∈ influential attribution token set
```

plus a visual-lexicon match.

## Lexicon-expansion pseudo-algorithm

This part is based on:

- [scripts/mine_visual_lexicon_from_attribution.py](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/mine_visual_lexicon_from_attribution.py)

```text
Input:
  - attribution JSONL from the training split
  - attribution methods (IG, Occlusion)
  - initial curated visual lexicon
  - visual vs non-visual query labels

Output:
  - expanded visual lexicon

Procedure:

  1. For each training query q:
       - decide whether q is visual or non-visual
         using gold qtypes or predicted labels

  2. For each attribution method:
       - extract top positive tokens
       - normalize them
       - remove short tokens, stopwords, and tokenization artifacts

  3. For each query q:
       - take the union of the method-specific token sets
       - treat that as the attribution-supported candidate set for q

  4. For each candidate token t:
       - count how many visual queries contain t
       - count how many non-visual queries contain t

  5. Compute token statistics such as:
       - positive query frequency
       - negative query frequency
       - precision
       - smoothed visual/non-visual lift
       - support-adjusted score

  6. Keep token t only if it passes thresholds on:
       - minimum positive frequency
       - minimum precision
       - minimum lift
       - minimum score

  7. Optionally exclude tokens already present in the default seed lexicon,
     so the output focuses on extension tokens only.

  8. Write the surviving tokens as the mined visual-lexicon extension and
     save ranked token statistics for inspection.
```

Important nuance:

- this is contrastive mining, not just frequency mining
- a token is kept only if it appears sufficiently often in visual queries and is not too common in non-visual queries
- in this note, "initial curated visual lexicon" means the starting/base lexicon before train-mined extension terms were added; for `union_relaxed_v2`, the recovered evidence is most consistent with the built-in default list recorded above

## Evaluation protocol

The strict module was evaluated at the query level, not at the token level.

```text
1. Select the dev queries whose gold qtypes require image evidence.
2. Run the strict query-token classifier.
3. Count a visual query as covered if at least one strict visual token is recovered.
4. Report:

   coverage =
     (# gold visual queries with at least one recovered strict visual token)
     /
     (# gold visual queries)
```

Why query-level evaluation was used:

- MMQA does not provide gold token-level labels for which query tokens are visual
- the practical goal of the module was to avoid leaving a visual query with no visual anchor tokens at all

The positive gold qtype set is defined in:

- [scripts/export_query_visual_binary_labels.py:155](/Users/hoseinerfan/Desktop/SPLICE/SpLiCE/scripts/export_query_visual_binary_labels.py:155)

## Not part of the main strict result

The main strict result did not rely on:

- auxiliary phrase-level labels
- attribution-based augmentation / fallback labels

Those exist in the exporter for analysis and extended views, but not as the core strict metric reported in the advisor-facing summary.

## Real examples from `union_relaxed_v2`

File used:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v2.jsonl`

### 1. LGBT example

- `query_id`: `e783cba0b3df36372d11823e378e5437`
- `query_text`: `Which completely bald person who wears thick glasses is among the members of LGBT billionaires?`
- `gold_qtype`: `ImageListQ`
- `pred_label_name`: `visual_needed`
- `important_token_overlap`:
  - `among`, `bald`, `billionaires`, `completely`, `glasses`, `lgbt`, `person`, `thick`, `wears`
- `visual_token_indices`:
  - `[2, 5, 7]`
- `strict_visual_tokens`:
  - `bald`, `wears`, `glasses`
- `visual_phrases`:
  - `bald person wears`
  - `glasses among`

Interpretation:

- this is a real multihop visual query
- the recovered visual anchors are appearance-oriented tokens, not the category phrase `LGBT billionaires`

### 2. Mustache example

- `query_id`: `46a4103ba65b176fba9ed85889775f8d`
- `query_text`: `Which candidate has a mustache among the candidates in Delaware's Mini-Tuesday?`
- `gold_qtype`: `ImageListQ`
- `pred_label_name`: `visual_needed`
- `important_token_overlap`:
  - `among`, `candidate`, `candidates`, `mini`, `mustache`, `tuesday`
- `visual_token_indices`:
  - `[4]`
- `strict_visual_tokens`:
  - `mustache`
- `visual_phrases`:
  - `mustache among candidates`
  - `mustache among`

Interpretation:

- this is a clean success case where one appearance token is enough to cover the query

### 3. Failure case

- `query_id`: `3b29528f6d900ff20bfebd2b938b851f`
- `query_text`: `Which African American artist was a musical guest at the 2003 Sanremo Music Festival?`
- `gold_qtype`: `ImageListQ`
- `pred_label_name`: `visual_needed`
- `important_token_overlap`:
  - `2003`, `african`, `american`, `artist`, `guest`, `music`, `musical`, `san`
- `visual_token_indices`:
  - `[]`
- `strict_visual_tokens`:
  - `[]`
- `visual_phrases`:
  - `[]`

Interpretation:

- this is a real strict failure case
- the query is visual in type, but no strict visual anchor token is recovered

## Meaning of `visual_token_indices`

`visual_token_indices` records the positions of the query tokens that were labeled as visual.

For the LGBT query:

```text
Which completely bald person who wears thick glasses is among the members of LGBT billionaires?
```

approximate token positions are:

- `0` `Which`
- `1` `completely`
- `2` `bald`
- `3` `person`
- `4` `who`
- `5` `wears`
- `6` `thick`
- `7` `glasses`
- `8` `is`
- `9` `among`
- `10` `the`
- `11` `members`
- `12` `of`
- `13` `LGBT`
- `14` `billionaires`
- `15` `?`

So:

- `visual_token_indices = [2, 5, 7]`

means:

- `bald`
- `wears`
- `glasses`

were labeled as `visual_needed`.

## Plot alignment check

For the LGBT page-query plot folder:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/page_query_heatmaps/e783cba0b3df36372d11823e378e5437_dev_query_visual_binary_labels_union_relaxed_v2/ret4_top4_allplots`

the following was confirmed from the summaries:

- `patch_dots_summary.json` uses
  - `splice_query_token_labels = .../dev_query_visual_binary_labels_union_relaxed_v2.jsonl`
- `heatmaps_summary.json` does not use the query-label file; it only records
  - `query_token_filter = full`

Therefore:

- the patch-dot query-axis coloring in that folder matches `union_relaxed_v2`
- the heatmaps are not driven by `union_relaxed_v2`

Patch-label nuance:

- that plot folder currently used
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far.jsonl`
- not the newer complete merged patch file
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl`

## Cluster commands

### List all query-label exports

```bash
find /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary \
  -type f -name 'dev_query_visual_binary_labels*.jsonl' | sort
```

### Extract the three real examples from `union_relaxed_v2`

```bash
LABELS=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v2.jsonl

python - "$LABELS" <<'PY'
import json, sys

path = sys.argv[1]
targets = [
    ("e783cba0b3df36372d11823e378e5437", "LGBT"),
    ("46a4103ba65b176fba9ed85889775f8d", "mustache"),
    ("3b29528f6d900ff20bfebd2b938b851f", "failure_case"),
]

rows = {}
with open(path) as f:
    for line in f:
        row = json.loads(line)
        qid = row.get("query_id")
        for target_qid, _ in targets:
            if qid == target_qid:
                rows[qid] = row

for qid, name in targets:
    print("=" * 100)
    print("name:", name)
    print("query_id:", qid)
    row = rows.get(qid)
    if row is None:
        print("MISSING")
        continue

    print("query_text:", row.get("query_text", ""))
    print("gold_qtype:", row.get("gold_qtype", ""))
    print("pred_label_name:", row.get("pred_label_name", ""))
    print("important_token_overlap:", row.get("important_token_overlap", []))
    print("visual_token_indices:", row.get("visual_token_indices", []))
    print("visual_token_indices_augmented:", row.get("visual_token_indices_augmented", []))
    print("strict_visual_tokens:", [x["token"] for x in row.get("token_labels", []) if x.get("label") == "visual_needed"])
    print("augmented_visual_tokens:", [x["token"] for x in row.get("token_labels", []) if x.get("augmented_label") == "visual_needed"])
    print("visual_phrases:", [x.get("norm_phrase", "") for x in row.get("phrase_labels", []) if x.get("label") == "visual_needed"])
PY
```

### Verify plot folder wiring

```bash
DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/page_query_heatmaps/e783cba0b3df36372d11823e378e5437_dev_query_visual_binary_labels_union_relaxed_v2/ret4_top4_allplots

python - "$DIR" <<'PY'
import json, os, sys

dir_path = sys.argv[1]
patch_summary = os.path.join(dir_path, "patch_dots_summary.json")
heatmap_summary = os.path.join(dir_path, "heatmaps_summary.json")

print("patch_dots_summary exists =", os.path.exists(patch_summary))
print("heatmaps_summary exists =", os.path.exists(heatmap_summary))

if os.path.exists(patch_summary):
    row = json.load(open(patch_summary))
    print("\\nPATCH-DOT SUMMARY")
    print("qid =", row.get("qid"))
    print("query =", row.get("query"))
    print("splice_query_token_labels =", row.get("splice_query_token_labels"))
    print("splice_patch_labels_jsonl =", row.get("splice_patch_labels_jsonl"))

if os.path.exists(heatmap_summary):
    row = json.load(open(heatmap_summary))
    print("\\nHEATMAP SUMMARY")
    print("qid =", row.get("qid"))
    print("query =", row.get("query"))
    print("query_token_filter =", row.get("query_token_filter"))
PY
```

## Key takeaways for the next chat

- use this handoff as the primary context file
- if the goal is advisor-facing strict explanations, be explicit that:
  - the strict module uses token-level attribution overlap plus lexicon matching
  - phrase labels and augmentation are not part of the main strict result
- if the goal is real examples aligned with the later visual-reranker plots:
  - use `union_relaxed_v2`
- if the goal is the most likely lexicon-expanded strict export behind the `90.85%` number:
  - inspect `union_relaxed_v6_fulltrainlex_v2` first
