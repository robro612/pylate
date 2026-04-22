from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn.functional as F

from pylate.scores import ScopedBatchScores, XTRKDScores, XTRScores


DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _summarize(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(t.mean().item()),
        "p50": float(t.quantile(0.5).item()),
        "p95": float(t.quantile(0.95).item()),
        "max": float(t.max().item()),
    }


def _print_summary(name: str, values: list[float]) -> None:
    s = _summarize(values)
    print(
        f"{name:<28} mean={s['mean']:.6e} p50={s['p50']:.6e} "
        f"p95={s['p95']:.6e} max={s['max']:.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose behavioral significance of old-vs-new XTR scoring parity."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Computation dtype for synthetic tensors.",
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nway", type=int, default=3)
    parser.add_argument("--q-len", type=int, default=12)
    parser.add_argument("--d-len", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--new-topk-half-fp32",
        action="store_true",
        help="Enable fp32->fp16 cast for topk in ScopedBatchScores.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="xtr_parity_significance.csv",
        help="Output CSV path for aggregate metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device. Use cpu for quick smoke checks.",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda, but CUDA is unavailable on this environment."
            )
        device = "cuda"
    else:
        device = "cpu"
    dtype = DTYPE_MAP[args.dtype]
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        print(
            "Warning: CPU low-precision behavior may differ from GPU; "
            "prefer --device cuda for fidelity."
        )

    old_xtr = XTRScores(k=args.k)
    new_xtr = ScopedBatchScores(
        mode="xtr",
        xtr_k=args.k,
        xtr_topk_cast_half_for_fp32=args.new_topk_half_fp32,
    )
    old_kd = XTRKDScores(k=args.k)
    new_kd = ScopedBatchScores(
        mode="xtr",
        xtr_k=args.k,
        xtr_topk_cast_half_for_fp32=args.new_topk_half_fp32,
    )

    # Contrastive-style metrics on full (B, B*N) scores
    full_abs_diffs: list[float] = []
    ce_diffs: list[float] = []
    top1_match: list[float] = []
    pos_top1_agreement: list[float] = []
    pos_rank_exact_match: list[float] = []
    pos_rank_abs_delta: list[float] = []

    # KD-style metrics on (B, N) scores
    kd_abs_diffs: list[float] = []
    kd_kl_old_new: list[float] = []
    kd_top1_match: list[float] = []

    for trial in range(args.trials):
        g = torch.Generator(device=device)
        g.manual_seed(1000 + trial)

        B, N, Q, D, H = (
            args.batch_size,
            args.nway,
            args.q_len,
            args.d_len,
            args.hidden,
        )
        queries = F.normalize(
            torch.randn(B, Q, H, device=device, dtype=dtype, generator=g), p=2, dim=-1
        )
        docs = F.normalize(
            torch.randn(B, N, D, H, device=device, dtype=dtype, generator=g), p=2, dim=-1
        )
        q_mask = torch.ones(B, Q, device=device, dtype=torch.bool)
        d_mask = torch.ones(B, N, D, device=device, dtype=torch.bool)

        # Full/global scores for contrastive-style behavior.
        s_old = old_xtr(queries, docs, queries_mask=q_mask, documents_mask=d_mask).float()
        s_new = new_xtr(
            queries,
            docs,
            queries_mask=q_mask,
            documents_mask=d_mask,
            scoring_scope="global",
            return_scope="global",
        ).float()

        full_abs_diffs.append((s_old - s_new).abs().mean().item())

        labels = torch.arange(B, device=device) * N
        ce_old = F.cross_entropy(s_old, labels)
        ce_new = F.cross_entropy(s_new, labels)
        ce_diffs.append((ce_old - ce_new).abs().item())

        top1_match.append((s_old.argmax(dim=1) == s_new.argmax(dim=1)).float().mean().item())
        pos_top1_agreement.append(
            ((s_old.argmax(dim=1) == labels) == (s_new.argmax(dim=1) == labels))
            .float()
            .mean()
            .item()
        )

        ranks_old = torch.argsort(s_old, dim=1, descending=True)
        ranks_new = torch.argsort(s_new, dim=1, descending=True)
        pos_rank_old = (ranks_old == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        pos_rank_new = (ranks_new == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        pos_rank_exact_match.append((pos_rank_old == pos_rank_new).float().mean().item())
        pos_rank_abs_delta.append(
            (pos_rank_old.float() - pos_rank_new.float()).abs().mean().item()
        )

        # KD/local-return scores.
        kd_old = old_kd(queries, docs, queries_mask=q_mask, documents_mask=d_mask).float()
        kd_new = new_kd(
            queries,
            docs,
            queries_mask=q_mask,
            documents_mask=d_mask,
            scoring_scope="global",
            return_scope="local",
        ).float()

        kd_abs_diffs.append((kd_old - kd_new).abs().mean().item())
        p_old = F.softmax(kd_old, dim=1)
        p_new = F.softmax(kd_new, dim=1)
        kd_kl_old_new.append(
            (p_old * (p_old.clamp_min(1e-12).log() - p_new.clamp_min(1e-12).log()))
            .sum(dim=1)
            .mean()
            .item()
        )
        kd_top1_match.append(
            (kd_old.argmax(dim=1) == kd_new.argmax(dim=1)).float().mean().item()
        )

    print("=== XTR old-vs-new significance diagnostics ===")
    print(f"dtype={args.dtype} trials={args.trials} B={args.batch_size} N={args.nway} Q={args.q_len} D={args.d_len} H={args.hidden} k={args.k}")
    print(f"new_topk_half_fp32={args.new_topk_half_fp32}")
    _print_summary("full_score_abs_mean", full_abs_diffs)
    _print_summary("contrastive_ce_abs_diff", ce_diffs)
    _print_summary("top1_match_rate", top1_match)
    _print_summary("pos_top1_agreement", pos_top1_agreement)
    _print_summary("pos_rank_exact_match", pos_rank_exact_match)
    _print_summary("pos_rank_abs_delta", pos_rank_abs_delta)
    _print_summary("kd_score_abs_mean", kd_abs_diffs)
    _print_summary("kd_kl(old||new)", kd_kl_old_new)
    _print_summary("kd_top1_match_rate", kd_top1_match)

    out_path = args.out_csv
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rows = [
        ("full_score_abs_mean", _summarize(full_abs_diffs)),
        ("contrastive_ce_abs_diff", _summarize(ce_diffs)),
        ("top1_match_rate", _summarize(top1_match)),
        ("pos_top1_agreement", _summarize(pos_top1_agreement)),
        ("pos_rank_exact_match", _summarize(pos_rank_exact_match)),
        ("pos_rank_abs_delta", _summarize(pos_rank_abs_delta)),
        ("kd_score_abs_mean", _summarize(kd_abs_diffs)),
        ("kd_kl(old||new)", _summarize(kd_kl_old_new)),
        ("kd_top1_match_rate", _summarize(kd_top1_match)),
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "metric",
                "dtype",
                "trials",
                "B",
                "N",
                "Q",
                "D",
                "H",
                "k",
                "new_topk_half_fp32",
                "mean",
                "p50",
                "p95",
                "max",
            ]
        )
        for metric, s in rows:
            writer.writerow(
                [
                    metric,
                    args.dtype,
                    args.trials,
                    args.batch_size,
                    args.nway,
                    args.q_len,
                    args.d_len,
                    args.hidden,
                    args.k,
                    int(args.new_topk_half_fp32),
                    s["mean"],
                    s["p50"],
                    s["p95"],
                    s["max"],
                ]
            )
    print(f"\nSaved diagnostics CSV to: {out_path}")


if __name__ == "__main__":
    main()
