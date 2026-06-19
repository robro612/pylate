#!/usr/bin/env python3
"""Model the on-disk size, per component, of tachiom and PLAID (fast-plaid) indices.

Pure arithmetic — no deps, no GPU. Lets you do capacity planning ("how big is a
10M-doc index at M=64?") and see the centroid/residual split without building.

Component models are derived from the actual on-disk layouts (verified against the
LoTTE/LateOn-regularized indices, which the defaults reproduce — run with no args):

TACHIOM  (single tachiom.bin; see tachiom.rs / vectorium multivec_two_level_pq.rs)
  per token (T = N * avg_vectors):
    pq_codes      = T * M                      (u8 x M)
    coarse_ids    = T * 4                       (u32, one coarse centroid id / token)
    norms         = T * 4   (only if with_norms)
    inverted_list = T * 4   (approx; u32 doc-id postings, ~1/token upper bound, data-dependent)
  per centroid (K = centroid_frac * T):
    hnsw_vectors  = K * d * 2                   (f16, kannolo HNSW dataset)
    hnsw_graph    = K * 8 * hnsw_m              (layer-0 = 2*hnsw_m u32 neighbours; dominant)
    encoder_f32   = K * d * 4                   (two-level PQ coarse_centroids, f32 -- REDUNDANT
                                                 with hnsw f16; drop via --tachiom-dedup-f32)
    offsets       = K * 8                        (usize)
  fixed:
    pq_codebook   = 256 * d * 4                  (M*KSUB*dsub*f32 = 256*d*4; ~0.13MB, negligible)

PLAID  (fast-plaid; per-shard {i}.* + merged_*; see freeze() / load.py)
  per token:
    residuals     = T * (d * nbits / 8)          (uint8 packed; merged_residuals)
    codes         = T * 8                         (int64 centroid id/token; merged_codes)
    ivf           = T * 4                         (int32 inverted file of embedding ids)
    [if NOT frozen: residuals + codes are stored TWICE (per-shard + merged) -> x2]
  per centroid (K_c, default ~ 2^round(log2(16*sqrt(T))), ColBERT heuristic):
    centroids     = K_c * d * 2                   (fp16 centroids.npy)
    ivf_lengths   = K_c * 4
  small/fixed: avg_residual, bucket_cutoffs/weights, doclens (~N*4) -- negligible.
"""
import argparse, math

KSUB = 256  # PQ centroids per subspace (8-bit), both backends


def _grp(d):
    return sum(d.values())


def tachiom_model(T, d, M, centroid_frac, hnsw_m, with_norms, dedup_f32):
    K = centroid_frac * T
    residuals = {
        "pq_codes (T*M)": T * M,
        "coarse_ids (T*4)": T * 4,
    }
    if with_norms:
        residuals["norms (T*4)"] = T * 4
    ivf = {
        "inverted_lists (~T*4)": T * 4,
        "offsets (K*8)": K * 8,
    }
    centroids = {
        "hnsw_vectors f16 (K*d*2)": K * d * 2,
        "hnsw_graph (K*8*hnsw_m)": K * 8 * hnsw_m,
    }
    if not dedup_f32:
        centroids["encoder_coarse f32 (K*d*4)"] = K * d * 4
    fixed = {"pq_codebook (256*d*4)": 256 * d * 4}
    return {"residuals": residuals, "centroids": centroids, "ivf": ivf, "fixed": fixed}


def plaid_centroid_heuristic(T):
    # ColBERT/PLAID: num_partitions ~ 16 * sqrt(n_embeddings), rounded to a power of 2.
    return int(2 ** round(math.log2(16 * math.sqrt(T))))


def plaid_model(T, N, d, nbits, n_centroids, frozen):
    res_per_tok = d * nbits / 8.0
    dup = 1 if frozen else 2  # un-frozen keeps per-shard + merged copies of residuals+codes
    residuals = {
        f"residuals (T*{res_per_tok:.0f}){'' if frozen else ' x2 unfrozen'}": T * res_per_tok * dup,
        f"codes int64 (T*8){'' if frozen else ' x2 unfrozen'}": T * 8 * dup,
    }
    ivf = {"ivf int32 (T*4)": T * 4, "ivf_lengths (Kc*4)": n_centroids * 4}
    centroids = {"centroids fp16 (Kc*d*2)": n_centroids * d * 2}
    fixed = {"doclens (~N*4)": N * 4}
    return {"residuals": residuals, "centroids": centroids, "ivf": ivf, "fixed": fixed}


def fmt(b):
    gib = b / 1024**3
    return f"{b/1024**2:10.1f} MiB  {gib:8.3f} GiB"


def report(name, groups, total_ref=None):
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    total = 0
    for gname, comps in groups.items():
        gtot = _grp(comps)
        total += gtot
        print(f"\n  [{gname}]  {fmt(gtot)}")
        for cname, b in comps.items():
            print(f"    {cname:<34} {fmt(b)}")
    print(f"\n  {'TOTAL':<34} {fmt(total)}")
    res = _grp(groups.get("residuals", {}))
    cen = _grp(groups.get("centroids", {}))
    if res:
        print(f"  centroids/residuals ratio: {cen/res:.3f}   "
              f"(centroids = {100*cen/total:.1f}% of total)")
    if total_ref is not None:
        print(f"  [measured on disk: {total_ref/1024**3:.2f} GiB  "
              f"-> model is {100*total/total_ref:.0f}% of measured]")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # corpus (defaults = LoTTE pooled/dev/search with LateOn-regularized)
    ap.add_argument("--n-docs", type=float, default=2_428_854)
    ap.add_argument("--avg-vectors", type=float, default=114.97,
                    help="avg token vectors per doc (T = n_docs * avg_vectors)")
    ap.add_argument("--total-tokens", type=float, default=None,
                    help="override T directly instead of n_docs*avg_vectors")
    ap.add_argument("--dim", type=int, default=128)
    # tachiom
    ap.add_argument("--pq-subspaces", type=int, default=32, help="tachiom M (PQ bytes/token)")
    ap.add_argument("--centroid-frac", type=float, default=0.01, help="tachiom K = frac * T")
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--tachiom-norms", action="store_true", help="index stores per-token norms")
    ap.add_argument("--tachiom-dedup-f32", action="store_true",
                    help="model dropping the redundant f32 coarse-centroid copy")
    # plaid
    ap.add_argument("--plaid-nbits", type=int, default=2)
    ap.add_argument("--plaid-centroids", type=int, default=None,
                    help="PLAID coarse centroid count (default: ColBERT heuristic)")
    ap.add_argument("--plaid-frozen", action="store_true",
                    help="index.freeze() applied (drops per-shard duplicate residuals/codes)")
    ap.add_argument("--only", choices=["tachiom", "plaid"], default=None)
    args = ap.parse_args()

    T = args.total_tokens if args.total_tokens else args.n_docs * args.avg_vectors
    N = args.n_docs
    print(f"N={N:,.0f} docs   avg_vectors={T/N:.2f}   T={T:,.0f} tokens   dim={args.dim}")

    if args.only != "plaid":
        g = tachiom_model(T, args.dim, args.pq_subspaces, args.centroid_frac,
                          args.hnsw_m, args.tachiom_norms, args.tachiom_dedup_f32)
        ref = 13_972_036_991 if (abs(T-279_230_440) < 1e6 and args.pq_subspaces == 32
                                 and not args.tachiom_dedup_f32 and not args.tachiom_norms) else None
        report(f"TACHIOM  (M={args.pq_subspaces}, centroid_frac={args.centroid_frac}, "
               f"K={args.centroid_frac*T:,.0f}"
               f"{', dedup_f32' if args.tachiom_dedup_f32 else ''})", g, ref)

    if args.only != "tachiom":
        kc = args.plaid_centroids or plaid_centroid_heuristic(T)
        g = plaid_model(T, N, args.dim, args.plaid_nbits, kc, args.plaid_frozen)
        report(f"PLAID  (nbits={args.plaid_nbits}, K_c={kc:,}, "
               f"{'frozen' if args.plaid_frozen else 'UNFROZEN (~2x)'})", g)


if __name__ == "__main__":
    main()
