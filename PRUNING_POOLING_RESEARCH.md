# Token Pruning and Pooling Research

## Overview

Exploration of token pruning and pooling strategies for ColBERT document representations to reduce memory footprint and improve retrieval efficiency. Note that these are all done only to document tokens.


## Methods

### Pruning
- [ ] top-k idf
   - [ ] global idf (at the global level, remove tokens with low idf)
   - [ ] document-wise idf (at the document level, remove tokens with low global idf)
- [ ] attention score
- [ ] first-k tokens
- [ ] stopwords
    - prune all stopwords that are 1 token in the tokenizer
- [ ] document-wise random sampling
    - sample without replacement tokens from each doc
- [ ] Compactor

### Pooling
- [x] hierarchical clustering
    - already implemented in `pylate` (`pool_factor` determines vectors per cluster. Also has protected tokens to not pool the first few tokens.)
- [ ] spherical clustering
    - [ ] implement

### Hybrid
Compose any of the above methods in a pipeline.