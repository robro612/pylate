from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

from ranx import Qrels, Run, compare
from rich.console import Console
from rich.table import Table


DEFAULT_METRICS = [
    "map",
    "ndcg@10",
    "ndcg@100",
    "recall@10",
    "recall@100",
    "hit_rate@5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate runfiles on an ir_datasets dataset and run statistical "
            "tests between runs using ranx."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Full ir_datasets dataset ID (e.g., 'beir/nfcorpus/test').",
    )
    parser.add_argument(
        "--runfiles",
        nargs="+",
        help="Paths to runfiles saved by ranx (e.g., JSON, TREC).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=(
            "List of metrics to evaluate. Defaults to the metrics used in "
            "eval_model_irds.py."
        ),
    )
    parser.add_argument(
        "--stat_test",
        type=str,
        default="student",
        help=(
            "Statistical test to use in ranx compare() "
            "(e.g., 'student', 'fisher'). Default: student."
        ),
    )
    parser.add_argument(
        "--max_p",
        type=float,
        default=0.01,
        help="P-value threshold for statistical significance. Default: 0.01.",
    )
    return parser.parse_args()


def expand_runfiles(runfiles: List[str]) -> List[str]:
    expanded: List[str] = []
    for runfile in runfiles:
        if any(ch in runfile for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(runfile))
            if not matches:
                console = Console()
                console.print(
                    f"[yellow]Warning:[/] No matches found for glob pattern: {runfile}"
                )
            expanded.extend(matches)
        else:
            expanded.append(runfile)
    return expanded


def load_runs(runfiles: List[str]) -> List[Run]:
    runs: List[Run] = []
    for runfile in runfiles:
        run_path = Path(runfile)
        console = Console()
        console.print(f"[cyan]Loading runfile:[/] {run_path}")
        run = Run.from_file(run_path.as_posix())
        # Use filename as a stable label if the run lacks a name.
        if not getattr(run, "name", None):
            run.name = run_path.stem
        runs.append(run)
    return runs


def main() -> None:
    args = parse_args()

    console = Console(width=200)
    console.print(f"[bold]Dataset:[/] {args.dataset}")
    console.print(f"[bold]Metrics:[/] {', '.join(args.metrics)}")
    console.print(f"[bold]Stat test:[/] {args.stat_test} (max_p={args.max_p})")

    qrels = Qrels.from_ir_datasets(args.dataset)
    if not args.runfiles:
        runfiles_arg = [f"results/{args.dataset.replace('/', '_')}/runs/*.json"]
    else:
        runfiles_arg = args.runfiles

    console.print(f"[bold]Runfiles:[/] {runfiles_arg}")

    runfiles = expand_runfiles(runfiles_arg)
    console.print(f"[bold]Expanded runfiles:[/] {runfiles}")

    if not runfiles:
        console.print("[red]Error:[/] No runfiles to evaluate.")
        raise SystemExit(1)
    runs = load_runs(runfiles)

    table = Table(title="Loaded Runfiles")
    table.add_column("Name", style="bold")
    table.add_column("Path")
    for runfile, run in zip(runfiles, runs):
        table.add_row(getattr(run, "name", Path(runfile).stem), runfile)
    console.print(table)

    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=args.metrics,
        stat_test=args.stat_test,
        max_p=args.max_p,
    )

    console.print("\n[bold]Statistical Comparison Report[/bold]\n")
    print(report)


if __name__ == "__main__":
    main()

