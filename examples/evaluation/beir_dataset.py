"""Evaluation script for BEIR datasets with a PLAID index."""

from __future__ import annotations

import argparse
import json
import os

from pylate import evaluation, indexes, models, retrieve
from pylate.models.compression import PoolingConfig, PoolingStrategy, IDFPruningConfig, IDFPruningStrategy

if __name__ == "__main__":
    query_len = {
        "quora": 32,
        "climate-fever": 64,
        "nq": 32,
        "msmarco": 32,
        "hotpotqa": 32,
        "nfcorpus": 32,
        "scifact": 48,
        "trec-covid": 48,
        "fiqa": 32,
        "arguana": 64,
        "scidocs": 48,
        "dbpedia-entity": 32,
        "webis-touche2020": 32,
        "fever": 32,
        "cqadupstack/android": 32,
        "cqadupstack/english": 32,
        "cqadupstack/gaming": 32,
        "cqadupstack/gis": 32,
        "cqadupstack/mathematica": 32,
        "cqadupstack/physics": 32,
        "cqadupstack/programmers": 32,
        "cqadupstack/stats": 32,
        "cqadupstack/tex": 32,
        "cqadupstack/unix": 32,
        "cqadupstack/webmasters": 32,
        "cqadupstack/wordpress": 32,
    }

    # Parse dataset_name from command line arguments
    parser = argparse.ArgumentParser(description="Dataset name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        nargs="+",
        default=["nfcorpus"],
        help="Dataset names to evaluate on.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        nargs="+",
        default=["lightonai/GTE-ModernColBERT-v1"],
        help="Model names or paths.",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="plaid",
        help="Index type to use (default: 'plaid')",
        choices=["flat", "plaid"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to write evaluation results.",
    )
    parser.add_argument(
        "--save_runfile",
        action="store_true",
        help="Save the runfile (run.json) for each run.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def sanitize_name(model_name: str) -> str:
        return "_".join(model_name.strip("/").split("/")[-2:])

    overall_results_path = os.path.join(args.output_dir, "overall_results.jsonl")

    def print_token_stats(embeddings: list, name: str) -> None:
        num_tokens = sum(len(embedding) for embedding in embeddings)
        data = {
            "num_tokens": num_tokens,
            "num_documents": len(documents),
            "avg_tokens_per_document": num_tokens / len(documents),
        }
        print(
            f"{name} - Number of tokens: {data['num_tokens']}, Number of documents: {data['num_documents']}, Average number of tokens per document: {data['avg_tokens_per_document']}"
        )
        return data

    for dataset_name in args.dataset_name:
        if "cqadupstack" in dataset_name:
            from beir import util

            util.download_and_unzip(
                url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
                out_dir="./evaluation_datasets/",
            )
            documents, queries, qrels = evaluation.load_custom_dataset(
                f"evaluation_datasets/{dataset_name}",
                split="test",
            )
            dataset_dir = sanitize_name(dataset_name)
        else:
            documents, queries, qrels = evaluation.load_beir(
                dataset_name=dataset_name,
                split="dev" if "msmarco" in dataset_name else "test",
            )
            dataset_dir = sanitize_name(dataset_name)
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)

        for model_name in args.model_name_or_path:
            print("=" * 80)
            print(f"Model: {model_name}")
            print("=" * 80)
            model = models.ColBERT(
                model_name_or_path=model_name,
                document_length=300,
                query_length=query_len.get(dataset_name),
            )

            documents_embeddings, artifacts = model.encode(
                sentences=[document["text"] for document in documents],
                batch_size=1000,
                is_query=False,
                show_progress_bar=True,
                convert_to_tensor=True,
                return_extra_artifacts={"input_ids": True, "attention_scores": False},
            )

            pruning_configs = [
                IDFPruningConfig(
                    mode="global",
                    top_k=k,
                    protected_tokens=1,
                    ignore_token_ids=model.tokenizer.all_special_ids,
                    use_tfidf=False,
                    track_pruned_tokens=False,
                ) for k in [5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
            ] + [
                IDFPruningConfig(
                    mode="document",
                    top_k=k,
                    protected_tokens=1,
                    ignore_token_ids=model.tokenizer.all_special_ids,
                    use_tfidf=False,
                    track_pruned_tokens=False,
                ) for k in [5, 10, 20, 30, 40, 50, 75, 100]
            ]
            pruning_strategies = [IDFPruningStrategy(config) for config in pruning_configs]

            pooling_configs = [
                PoolingConfig(
                    pool_factor=k,
                    protected_tokens=1,
                    clustering_method="hierarchical",
                    show_progress_bar=True,
                ) for k in [2, 3, 4, 5]
            ]
            pooling_strategies = [PoolingStrategy(config) for config in pooling_configs]
            compression_strategies = [*pruning_strategies, *pooling_strategies]

            queries_embeddings = model.encode(
                sentences=list(queries.values()),
                is_query=True,
                show_progress_bar=True,
                batch_size=512,
                convert_to_tensor=True,
            )

            experiments = {
                "baseline": None,
                # **{strategy.name: strategy for strategy in compression_strategies}
            }

            for name, compression_strategy in experiments.items():
                if compression_strategy is None:
                    compressed_embs = documents_embeddings
                else:
                    compressed_embs, _compressed_artifacts = compression_strategy.compress(
                        embeddings=documents_embeddings,
                        artifacts=artifacts,
                    )

                results = {}
                token_stats = print_token_stats(compressed_embs, name)
                results.update(token_stats)

                match args.index_type:
                    case "flat":
                        index = indexes.Flat(
                            override=True,
                            index_name=f"{dataset_dir}_{sanitize_name(model_name)}",
                        )
                    case "plaid":
                        index = indexes.PLAID(
                            override=True,
                            index_name=f"{dataset_dir}_{sanitize_name(model_name)}",
                        )
                    case _:
                        raise ValueError(f"Invalid index type: {args.index_type}")

                retriever = retrieve.ColBERT(index=index)

                index.add_documents(
                    documents_ids=[document["id"] for document in documents],
                    documents_embeddings=compressed_embs,
                )

                scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=20)

                # Remove query_id from scores, needed for FiQA dataset
                for (query_id, _query), query_scores in zip(queries.items(), scores):
                    for score in list(query_scores):
                        if score["id"] == query_id:
                            query_scores.remove(score)

                evaluation_scores, run = evaluation.evaluate(
                    scores=scores,
                    qrels=qrels,
                    queries=list(queries.keys()),
                    metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
                    return_run=True,
                )

                print(f"Evaluation scores for {name}:")
                print({metric: round(float(value), 3) for metric, value in evaluation_scores.items()})
                results.update(evaluation_scores)

                model_dir = sanitize_name(model_name)
                run_dir = os.path.join(args.output_dir, dataset_dir, model_dir)
                os.makedirs(run_dir, exist_ok=True)

                with open(os.path.join(run_dir, "evaluation_results.json"), "w") as f:
                    json.dump(results, f, indent=2)

                if args.save_runfile:
                    run.save(os.path.join(run_dir, "run.json"))

                summary = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "experiment": name,
                    "metrics": evaluation_scores,
                    "token_stats": token_stats,
                }
                with open(overall_results_path, "a") as f:
                    f.write(json.dumps(summary) + "\n")
