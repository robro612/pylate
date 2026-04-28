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
