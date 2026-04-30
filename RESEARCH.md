# Research Notes

## Query Token Weighting

- Query-token weighting **training path is implemented**.
  - `examples/train/hydra_train.py` can build and append a `QueryTokenWeightHead`.
  - `conf/train/contrastive.yaml` includes `query_token_weight_head` config knobs.
  - `pylate/models/QueryTokenWeightHead.py` is present with tests in `tests/test_query_token_weight_head.py`.
  - Training runs confirm small NanoBEIR bump from weighting (0.594 -> 0.614 mean ndcg@10)

- Query-token weighting is **applied at encode time for queries**.
  - In `pylate/models/colbert.py`, query embeddings are multiplied by `query_token_weights` in `prepare_token_embeddings_for_scoring()`.
  - This means any retrieval/eval path that uses `model.encode(..., is_query=True)` receives weighted query token vectors.
  - But no retrieval path (especially XTR + WARP which do E2E retrieval) is wired in yet.

## Query Length - Candidate Set Size Exploration
Want to explore the relationship between query length and candidate set size (driver of latency in the search pipeline).

Data: BrowseComp+'s ~100K document corpus, AgentIR's 5.24K queries that look like:

```
Reasoning: Thus we have a clear answer: the manager who was replaced is A.J. Hinch.

But to be thorough, we should confirm that his number was retired by the Tigers? Wait, but maybe the question incorrectly states "interim manager who had his number retired by the Detroit Tigers." Actually the interim manager is "Kirk Gibson", whose number might indeed be retired by the Tigers? But I'm not sure. Let's search for "Tigers retired #29" again with maybe a site like MLB.com.

Query: "#29" "Detroit Tigers" "retired" "number"
```

Models: a fixed set of maybe 3-5 representative ColBERT/XTR models.

Expansion Methods (on top of only the Query: {query} part of the AgentIR "query"): 

- None
- RM3/PRF
- HyDE
- Doc2Query
- add back reasoning

Length variants: 

- bin naturally by length
- pruning
  - truncation
  - idf
  - learned importance weight (I've got a recent work testing this)

"Index": Cluster centroids gathered the same way PLAID/WARP would.

Measurements:
For (model, query, expansion, truncation): measure candidate set size (num centroids and num documents) for num_probes.

Plot:
query length vs candidate set size

### Implementation checklist

- [x] Verify centroid-path feasibility without full retrieval (fast-plaid centroids + IVF postings are accessible from index artifacts).
- [x] Build corpus length diagnostics for BrowseComp+ (histogram + tail stats from corpus-wide scan).
- [x] Validate long-doc encoding feasibility:
  - [x] single-doc encode works with default doc length.
  - [x] `document_length=100000` fails in fp32 (V100/H100 OOM) and succeeds on H100 with bf16.
- [x] Build standalone dynamic corpus encoder (`scripts/dynamic_encode_hf_corpus.py`) with:
  - [x] dynamic cost-budget batching with OOM backoff.
  - [x] sort order toggle (`smallest`/`biggest` first).
  - [x] buffered shard writes (doc vectors are not split across shards).
  - [x] intelligent resume from existing shards (`output_dir`) + auto shard index continuation.
  - [x] `torch.compile` enabled by default (with safe fallback).
  - [x] sample mode for fast dry runs.
  - [x] live progress postfix including current batch size and buffer status.
  - [ ] Implement this into main `evaluate_indexes.py` for future use.
- [ ] Run full BrowseComp+ encode (biggest -> smallest, bf16, H100), producing final doc embedding shards. (In Progress)
- [x] Add AgentIR query loader/parser (`Reasoning` + `Query` fields) and query-only extraction.
- [ ] Implement query expansion variants (none / RM3-PRF / HyDE / Doc2Query / add reasoning).
  - [x] none (`plain`) implemented.
  - [x] add reasoning (`query_plus_reasoning`) implemented.
  - [x] RM3/PRF-style lexical expansion implemented and tuned on first-10 sample.
  - [x] HyDE via DSPy implemented + Hydra LM config wiring.
    - end-to-end generation run/test.
  - [ ] Doc2Query implementation.
- [ ] Implement query-length pruning variants (natural bins / truncation / idf / learned importance).
- [ ] Implement centroid candidate analysis runner:
  - [ ] load centroids + postings.
  - [ ] query-to-centroid matmul for multiple `num_probes`.
  - [ ] emit per-query candidate counts (centroids + docs).
- [ ] Generate final analysis plots: query length vs candidate set size by model / expansion / pruning / probe.
