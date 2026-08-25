# Chimera

[Chimera](https://github.com/iidyc/Chimera) is a single-GPU multi-vector
retrieval system: RaBitQ-style residual quantisation of document tokens, a CAGRA
graph over the cluster centroids for probing, and an AVX-512 CPU rerank of the
candidate pool with full-bit codes. `pylate.indexes.Chimera` wraps its pybind11
`ChimeraIndex` as an end-to-end PyLate index, so it goes through the same
`retrieve.ColBERT` path and the same `scripts/benchmark_indexes.py` harness as
PLAID/FastPlaid and tachiom.

```bash
# once per environment, on an L40S/A100/H100 node
srunl40s --mem=64G --cpus-per-task=16 ./scripts/sync_env.sh cu130

# then, like any other index
UV_PROJECT_ENVIRONMENT=.venv-cu130 uv run --no-sync \
    python scripts/benchmark_indexes.py model=lateon index=chimera \
    datasets=[beir/nfcorpus/test]
```

## Installation

Chimera is the `chimera` extra, built from the `../Chimera` checkout via
`[tool.uv.sources]` — the same shape as tachiom and fast-plaid. Upstream has no
Python packaging at all (a bare CMake project; you were meant to build it in a
conda env and put `build/` on `PYTHONPATH`), so the checkout carries a
`pyproject.toml` on branch `pylate-packaging`. See `third_party/README.md`.

### No conda

Upstream's `setup/setup_chimera_env.sh` creates a conda environment and installs
`libcuvs`, `librmm`, nvcc, cmake and pybind11 into it. We do not use it.
`.venv-cu130` already carries the RAPIDS **pip** wheels — `libcuvs`, `libraft`,
`librmm`, `rapids_logger`, each shipping headers and a CMake config — pulled in
by the `cu130` extra for `scripts/gpu_cluster.py`, and `/usr/local/cuda-13.1`
supplies nvcc. The packaging commit teaches `CMakeLists.txt` to find them there
when no conda toolchain is configured; the conda path still works and still
takes precedence.

### Build isolation is off, deliberately

`[tool.uv] no-build-isolation-package = ["chimera-retrieval"]`. The extension
resolves cuVS/RAFT/RMM from the *target* environment's site-packages and bakes
those directories into its RPATH, which is what makes `import chimera` resolve
`libcuvs.so` with nobody exporting `LD_LIBRARY_PATH`. A PEP 517 isolated build
environment is deleted after the build, so an RPATH into it would dangle. It
also means the pybind11 module is compiled against the interpreter that will
import it. `scikit-build-core` and `pybind11` are therefore listed in the
`chimera` extra rather than left to `[build-system] requires`, and
`sync_env.sh` holds `chimera-retrieval` back through phase 1 so they are in
place before it builds — the same two-phase shape as `LIBTORCH` for the Rust
forks.

### cu130 only

`compute_full_bit_scores` uses AVX-512 intrinsics unconditionally, and the V100
nodes behind `cu126` are Intel Xeon E5-2698 v4 (Broadwell) without AVX-512 —
Chimera cannot run there at all, so `sync_env.sh` skips it for those targets.
Two further things tie the artifact to the node class that built it:

* `CMakeLists.txt` sets `-march=native`. The login node is an EPYC 7713 (Zen 3,
  no AVX-512) and the compile fails there outright; L40S nodes are EPYC 9534
  (Zen 4) and have it. A Zen-4 build will not run on the Intel A100/H100 nodes.
* `CMAKE_CUDA_ARCHITECTURES` defaults to `native`, read off the building node's
  GPU. `sync_env.sh` passes it explicitly, because getting it wrong does not
  fail the build — it ships PTX the driver cannot JIT and dies at the first
  kernel launch with *"the provided PTX was compiled with an unsupported
  toolchain"*. Set `CHIMERA_CUDA_ARCH` (89 = L40S, 80 = A100, 90 = H100) to
  build for a different class.

Upstream's own `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard sat *below*
`enable_language(CUDA)`, which under CMP0104 has already filled the variable in
— so it never fired and every build that did not pass `-D` got the PTX failure.
The second commit on `pylate-packaging` moves it up.

## Compile-time query shape

`cpp/include/chimera/config.cuh` fixes two numbers into the binary. Upstream
made them source edits; the packaging branch turns them into build options and
exports the resolved values as `chimera.PADDED_DIM` / `chimera.Q_DOCLEN`, which
is what the wrapper reads — so it validates against the binary it actually
loaded rather than assuming.

| constant | this environment | meaning |
|---|---|---|
| `PADDED_DIM` | 128 | dimension after rotation; embeddings must be 64-128 |
| `Q_DOCLEN` | **48** | query tokens, exactly |

`sync_env.sh` builds `Q_DOCLEN=48` (override with `CHIMERA_Q_DOCLEN`, a multiple
of 16) because it has to cover the longest `query_length` in
`conf/eval/config.yaml` — 48, for trec-covid, scidocs and scifact — and the
wrapper raises on a longer query rather than truncating, since silently dropping
tokens would change what the benchmark measures.

Queries shorter than `Q_DOCLEN` are zero-padded. That is score-preserving for
MaxSim — a zero query vector contributes 0 to every document — and measurably
so: on nfcorpus (`query_length` 32) a `Q_DOCLEN=48` build returns *exactly* the
same ndcg@10 0.3783 and r@100 0.3372 as a `Q_DOCLEN=32` build. It is not free
in throughput, though, because each padded token still runs its own CAGRA probe
and drags documents into the candidate pool:

| build | nfcorpus QPS | ndcg@10 | r@100 |
|---|---|---|---|
| `Q_DOCLEN=32` (exact) | 122 | 0.3783 | 0.3372 |
| `Q_DOCLEN=48` (16 padded) | 90 | 0.3783 | 0.3372 |

So **a QPS number is only comparable when `Q_DOCLEN` matches the dataset's
`query_length`.** trec-covid, scidocs and scifact are exact at 48 and pay
nothing. For a speed comparison on the 32-token corpora — nfcorpus, lotte,
msmarco — rebuild with `CHIMERA_Q_DOCLEN=32`, or PLAID and tachiom get a ~26%
head start they did not earn. Quality comparisons are unaffected either way.

Changing `Q_DOCLEN` or `PADDED_DIM` invalidates every index already built: the
rotation and code layout depend on both. The wrapper records them in the index's
`params.json` and refuses to search a mismatched index rather than returning
nonsense.

`nprobe` becomes CAGRA's search `k`, which RAFT's topk kernel caps at 1024. The
wrapper raises at construction instead of letting the search die several seconds
in with `RAFT failure ... topk must be lower than or equal to 1024`.

## Upstream behaviours the wrapper works around

### Searching a freshly built index is unreliable

`ChimeraIndex.build` returns a live index, and searching that object drops the
correct answer often enough to move a benchmark. Measured on 2000 self-queries
(a document's own tokens as the query, which must rank it first) — same process,
same search options, identical index contents:

| index object | queries with >1 distinct top-10 across repeats | self-doc missed top-1 |
|---|---|---|
| `build()`-returned | 5/5 | 54/500 (10.8%) |
| `load()`-ed | 0/5 | 0/500 |

The data `save()` writes is correct; only the in-memory post-build search state
is not. `Chimera._build` therefore drops the built object after saving and lets
`_ensure_loaded()` read the artifact back — a reload, in exchange for
determinism. `tests/test_chimera.py::test_search_is_deterministic` guards it.
Revisit if upstream fixes `initialize_search_state()` on the build path.

### Scores are not returned

The pybind11 `search` returns document ids only. The late-interaction scores
exist — `collaborative_document_scoring` builds a
`priority_queue<pair<float, size_t>>` — but they are dropped when the heap is
drained. Documents come back in descending-score order, so `Chimera.__call__`
synthesises strictly decreasing placeholders and sets
`scores_are_synthetic = True`.

Rank metrics (NDCG, recall, MAP, MRR) are exact. Anything that reads score
*values* — score distributions, run fusion, calibration — is not, and run files
under `results/runfiles/` carry the placeholders. Surfacing the real scores
means widening `chimera_index::search` and the binding to return the heap's
`.first`; the wrapper already picks up a `search_scored`/`search_with_scores`
method automatically if a build exposes one, so that patch needs no change here.

## Memory, and what the fork fixes

`ChimeraIndex.build` takes the whole corpus in one call — there is no
incremental path, where WARP and PLAID stream shard by shard and tachiom reads
shards natively in Rust. So the build peak cannot be amortised, and two commits
on `pylate-packaging` are what make a large collection possible at all.

**The corpus was held twice.** `python/bindings.cpp` copied the NumPy array into
a `std::vector<float>` before calling `build`, because `build` took a
`const std::vector<float>&`. Nothing in the build retains or mutates it, so the
copy was pure duplication of the largest object in the process. `build` now has
a borrowing `const float*` overload and the binding passes `embeddings.data()`.

**The doc-major 1-bit codes were dead.** `encode_embeddings` produces the 1-bit
codes in document order; `reorder_by_cluster` copies them into the cluster-major
array that the GPU probe scan reads. The originals were kept, written to
`doc_1bit.bin` and read back at load, but no search path touched them. They are
no longer retained or persisted. `doc_1bit.bin` still exists — it also carries
the header, the rotator and the full-bit factors — so `IndexHeader` gained a
magic and format version, and an index written by an older build now fails
loudly instead of parsing its factors out of the wrong bytes.

Measured peak host RSS building from the lotte shards:

| tokens | before | after |
|---|---|---|
| 13.2M | 16.2 GiB | 9.9 GiB |
| 32.2M | 38.9 GiB | 23.5 GiB |
| slope | 1.20 GiB / M tokens | **0.72 GiB / M tokens** |

lotte/pooled/dev/search is 354.6M tokens: ~424 GiB before, **~254 GiB after**,
against 349 GiB on an L40S node. That is the difference between an OOM kill and
a build with headroom. Reproduce with `scripts/analysis/chimera_scale.py`.

The remaining 169 GiB at that scale is the float32 array itself, forced by
Chimera's f32-only API — the fp16 cache doubles on the way in. If it ever runs
tight, the pointer overload means `np.memmap` needs no further C++ change.

## Other differences from PLAID / tachiom

* **No stage profiling.** The C++ search is opaque from Python, so
  `search.profile=true` reports one derived `overhead/dispatch` span rather than
  a breakdown. There is no equivalent of `crates/stage-profile` here.
* **No update, delete, or embedding reconstruction.** `add_documents` builds
  once and raises on a second call; `get_documents_embeddings` has no backing
  API.
* **Denser codes, and a second representation.** Per token on nfcorpus:

  | file | B/token | |
  |---|---|---|
  | `doc_full.bin` | 80.0 | rerank codes, `PADDED_DIM x (1+ex_bits)/8` |
  | `cluster_1bit.bin` | 28.0 | 1-bit codes cluster-major + doc id, for the GPU probe |
  | `ivf.bin` | 4.1 | posting lists |
  | `centroids.carga` | 5.1 | centroids + CAGRA graph (scales with clusters) |
  | `doc_1bit.bin` | 4.0 | header, rotator, full-bit factors |
  | **total** | **121.2** | vs tachiom 78.0, plaid 47.0 |

  The gap is `ex_bits`: at the default 4 the rerank code is 5 bits/dim, against
  2 bits/dim for both PLAID (`nbits: 2`) and tachiom (`pq_subspaces: 32`). The
  `cluster_1bit.bin` 28 B/token has no analogue in either — it is the price of
  a GPU-resident coarse stage.

## Reference numbers

`beir/nfcorpus/test`, `lightonai/LateOn`, 3633 docs / 866k tokens, one L40S.
Chimera rows are `Q_DOCLEN=48`, so their QPS carries the ~26% padding penalty
described above; at a matched `Q_DOCLEN=32` the `ex_bits=4` row is 122 QPS.

| index | build | disk | QPS | NDCG@10 | R@100 |
|---|---|---|---|---|---|
| plaid (`nbits=2`) | 22.9s | 39 MB | 79 | 0.3804 | 0.3424 |
| tachiom (tac, m32) | 54.9s | 64 MB | 168 | 0.3789 | 0.3377 |
| chimera `ex_bits=1` | 2.6s | 61 MB | 91 | 0.3486 | 0.3088 |
| chimera `ex_bits=2` | 2.9s | 74 MB | 92 | 0.3762 | 0.3342 |
| chimera `ex_bits=4` (default) | 3.3s | 100 MB | 91 | 0.3783 | 0.3372 |

Reading it honestly: **at this scale Chimera does not win.** It is 2.6x PLAID's
disk at slightly worse quality, and slower than tachiom. `ex_bits=2` is the
sensible operating point — 99.4% of `ex_bits=4`'s NDCG for 74% of the disk —
and `ex_bits=1` is where quality actually breaks (-0.030 NDCG). QPS is flat
across all three, so `ex_bits` is a disk/quality knob, not a speed one, and the
rerank is not the bottleneck here.

The one unambiguous win is build time: 3s against PLAID's 23s and tachiom's 55s,
which at lotte scale is PLAID's 121 min and tachiom's 346 min against a
projected ~22 min.

nfcorpus is 3633 documents, so its QPS is dominated by fixed overhead rather
than by the candidate pipeline. Nothing here settles the scaling claim — that
needs trec-covid (171k docs, 29.3M tokens) and lotte/pooled (2.4M docs, 355M
tokens), both of which are now cached and within memory budget.

`k_full_bit` is the recall knob — the same lever as tachiom's
`k_docs_to_score` — and upstream's default of 300 is starved: it gives ndcg@10
0.3634 / r@100 0.2903 at 347 QPS against 2000's 0.3783 / 0.3372 at 122. Raising
`nprobe` instead does almost nothing. `conf/eval/index/chimera.yaml` defaults to
2000.
