"""Evaluation script for BEIR datasets with a PLAID index."""

from __future__ import annotations

import argparse

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
        default="nfcorpus",
        help="Name of the dataset to evaluate on (default: 'fiqa')",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="plaid",
        help="Index type to use (default: 'plaid')",
        choices=["flat", "plaid"],
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name

    model_name = "lightonai/GTE-ModernColBERT-v1"
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )

    if "cqadupstack" in dataset_name:
        # Download dataset if not already downloaded
        from beir import util

        data_path = util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
        dataset_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
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

    experiments = {
        "baseline" : None,
        **{strategy.name : strategy for strategy in compression_strategies}
    }
    experiment_results = {}

    for name, compression_strategy in experiments.items():
        if compression_strategy is None:
            compressed_embs = documents_embeddings
        else:
            compressed_embs, compressed_artifacts = compression_strategy.compress(
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
                    index_name=f"{dataset_name}_{model_name.split('/')[-1]}",
                )
            case "plaid":
                index = indexes.PLAID(
                    override=True,
                    index_name=f"{dataset_name}_{model_name.split('/')[-1]}",
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
        for (query_id, query), query_scores in zip(queries.items(), scores):
            for score in query_scores:
                if score["id"] == query_id:
                    # Remove the query_id from the score
                    query_scores.remove(score)

        evaluation_scores = evaluation.evaluate(
            scores=scores,
            qrels=qrels,
            queries=list(queries.keys()),
            # queries=queries,
            metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
        )

        print(f"Evaluation scores for {name}:")
        print(evaluation_scores)
        results.update(evaluation_scores)
        experiment_results[name] = results

    import pandas as pd
    df = pd.DataFrame.from_dict(experiment_results, orient="index")
    df.to_csv(f"evaluation_results_{dataset_name}_{model_name.split('/')[-1]}_{args.index_type}.tsv", sep="\t", index=True)
