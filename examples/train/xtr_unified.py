"""Unified training script for both Knowledge Distillation and Contrastive training."""
from __future__ import annotations

import os
import torch

def is_main_process():
    return (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )


if not is_main_process():
    os.environ["WANDB_DISABLED"] = "true"

from argparse import ArgumentParser
from typing import Literal, Optional

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils, scores

def parse_arguments():
    """Parse command-line arguments with organized groups."""
    parser = ArgumentParser(
        description="Unified training script for Knowledge Distillation and Contrastive training"
    )

    # Training Method Group
    method_group = parser.add_argument_group("Training Method")
    method_group.add_argument(
        "--training_method",
        type=str,
        choices=["distillation", "contrastive", "xtr_mixedbread"],
        default="distillation",
        help="Training method: 'distillation' or 'contrastive' or 'xtr_mixedbread'",
    )
    method_group.add_argument(
        "--auxiliary_loss_weight",
        type=float,
        default=None,
        help="Weight for auxiliary loss",
    )
    # Model Group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Base model name or path",
    )
    model_group.add_argument(
        "--query_length",
        type=int,
        default=32,
        help="Query length for ColBERT model",
    )
    model_group.add_argument(
        "--doc_length",
        type=int,
        default=300,
        help="Document length for ColBERT model",
    )

    # Dataset Group
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--distillation_train_dataset",
        type=str,
        default="lightonai/ms-marco-en-bge-gemma",
        help="Train dataset path (for distillation)",
    )
    dataset_group.add_argument(
        "--contrastive_dataset",
        type=str,
        default="sentence-transformers/msmarco-bm25",
        help="Triplet dataset path (for contrastive)",
    )
    dataset_group.add_argument(
        "--train_dataset_subset",
        type=str,
        default=None,
        help="Dataset subset/name for contrastive/kd training (optional, omit for datasets without subsets like bclavie/msmarco-10m-triplets)",
    )
    dataset_group.add_argument(
        "--nways",
        type=int,
        default=32,
        help="Number of negatives per query for distillation training, defaults to 32",
    )
    dataset_group.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.01,
        help="Test split ratio for contrastive dataset",
    )

    # XTR Parameters Group
    xtr_group = parser.add_argument_group("XTR Parameters")
    xtr_group.add_argument(
        "--use_colbert",
        action="store_true",
        help="Use ColBERT score function instead of XTR (baseline)",
    )
    xtr_group.add_argument(
        "--k_prime",
        type=int,
        default=128,
        help="k_prime parameter for XTR",
    )
    xtr_group.add_argument(
        "--k_prime_start",
        type=int,
        default=None,
        help="Starting k_prime for annealing (defaults to k_prime if not set)",
    )
    xtr_group.add_argument(
        "--k_prime_anneal_steps",
        type=int,
        default=1000,
        help="Number of steps over which to anneal k_prime",
    )
    xtr_group.add_argument(
        "--k_prime_schedule",
        type=str,
        default="constant",
        choices=["linear", "exponential", "constant"],
        help="Schedule type for k_prime annealing",
    )
    xtr_group.add_argument(
        "--impute_scores_instead_of_zero",
        action="store_true",
        help="Impute scores instead of zero for XTR",
    )
    xtr_group.add_argument(
        "--use_normalizer_Z",
        action="store_true",
        help="Use normalizer Z for XTR",
    )
    xtr_group.add_argument(
        "--Z_clamp_value",
        type=float,
        default=1.0,
        help="Clamp value for normalizer Z",
    )
    xtr_group.add_argument(
        "--start_normalizer_Z_at_step",
        type=int,
        default=0,
        help="Step at which to start using normalizer Z",
    )
    xtr_group.add_argument(
        "--score_fn",
        type=str,
        default=None,
        help="Name of score functions to use from pylate.scores (e.g., 'xtr_kd_training_scores', 'xtr_contrastive_training_scores', 'xtr_contrastive_training_scores_multiple_negatives'). If not provided, defaults based on training_method.",
    )
    xtr_group.add_argument(
        "--score_all_docs_at_once",
        action="store_true",
        help="(contrastive) Score all documents (pos+negs) in a single call (needed for XTR global retrieval-pool thresholding).",
    )
    xtr_group.add_argument(
        "--positive_document_index",
        type=int,
        default=0,
        help="(contrastive) Index of the positive document among the provided document groups (default: 0).",
    )

    # Training Hyperparameters Group
    training_group = parser.add_argument_group("Training Hyperparameters")
    training_group.add_argument(
        "--lr",
        type=float,
        default=3e-6,
        help="Learning rate",
    )
    training_group.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per device",
    )
    training_group.add_argument(
        "--grad_acc_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    training_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    training_group.add_argument(
        "--num_train_steps",
        type=int,
        default=-1,
        help="Number of training steps (if positive, overrides num_train_epochs)",
    )
    training_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.005,
        help="Warmup ratio for the learning rate scheduler",
    )
    training_group.add_argument(
        "--kd_minmax_normalize",
        action="store_true",
        help="Normalize the scores using min-max normalization (for distillation)",
    )

    # Evaluation Group
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--use_triplet_evaluator",
        action="store_true",
        help="Use ColBERTTripletEvaluator (requires eval dataset for contrastive)",
    )
    eval_group.add_argument(
        "--use_nanobeir_evaluator",
        action="store_true",
        help="Use NanoBEIREvaluator",
    )
    eval_group.add_argument(
        "--nanobeir_datasets",
        type=str,
        nargs="*",
        default=None,
        help="NanoBEIR dataset names to evaluate on (default: all)",
    )
    eval_group.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluation steps interval",
    )

    # Output Group
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--save_steps",
        type=int,
        default=10_000,
        help="Save checkpoint steps interval",
    )
    output_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the default run name",
    )

    return parser.parse_args()


def load_distillation_datasets(train_dataset_path: str, name: Optional[str] = None):
    """Load datasets for knowledge distillation training."""
    if name is None:
        name = "train"
    train = load_dataset(path=train_dataset_path, name=name)
    queries = load_dataset(path=train_dataset_path, name="queries")
    documents = load_dataset(path=train_dataset_path, name="documents")
    return train, queries, documents


def load_contrastive_datasets(
    dataset_path: str, dataset_name: str | None, eval_split_ratio: float
):
    """Load datasets for contrastive training.
    
    Args:
        dataset_path: Path to the dataset
        dataset_name: Optional subset/name of the dataset. If None, loads dataset without subset.
        eval_split_ratio: Ratio for train/test split
    """
    if dataset_name:
        dataset = load_dataset(dataset_path, dataset_name, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")
    splits = dataset.train_test_split(test_size=eval_split_ratio)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    print(f"Loaded {len(train_dataset)} train samples and {len(eval_dataset)} eval samples")
    return train_dataset, eval_dataset


def create_k_prime_scheduler(
    k_prime: int,
    k_prime_start: int,
    k_prime_anneal_steps: int,
    k_prime_schedule: str,
):
    """Create k_prime scheduler function."""

    def k_prime_scheduler(step: int) -> int:
        """Compute k_prime based on current training step."""
        if k_prime_schedule == "constant":
            return k_prime
        elif k_prime_schedule == "linear":
            if step >= k_prime_anneal_steps:
                return k_prime
            progress = step / k_prime_anneal_steps
            k_prime_current = k_prime_start + (k_prime - k_prime_start) * progress
            return int(k_prime_current)
        elif k_prime_schedule == "exponential":
            if step >= k_prime_anneal_steps:
                return k_prime
            # Exponential annealing: 1 - (1 - progress)^2
            progress = step / k_prime_anneal_steps
            exponential_progress = 1 - (1 - progress) ** 2
            k_prime_current = k_prime_start + (k_prime - k_prime_start) * exponential_progress
            return int(k_prime_current)
        else:
            raise ValueError(f"Unknown schedule type: {k_prime_schedule}")

    return k_prime_scheduler


def create_score_function(
    training_method: Literal["distillation", "contrastive", "xtr_mixedbread"],
    use_colbert: bool,
    k_prime: int,
    k_prime_start: int,
    k_prime_anneal_steps: int,
    k_prime_schedule: str,
    use_normalizer_Z: bool,
    Z_clamp_value: float,
    start_normalizer_Z_at_step: int,
    impute_scores_instead_of_zero: bool,
    score_all_docs_at_once: bool = False,
    score_fn: Optional[str] = None,
):
    """Create the appropriate score function based on training method and configuration."""
    # XTR loss doesn't use a score function - it computes scores internally
    if training_method == "xtr_mixedbread":
        return None
    
    if use_colbert:
        if training_method == "distillation":
            return scores.colbert_kd_scores
        else:
            return scores.colbert_scores
    else:
        k_prime_scheduler_fn = create_k_prime_scheduler(
            k_prime=k_prime,
            k_prime_start=k_prime_start,
            k_prime_anneal_steps=k_prime_anneal_steps,
            k_prime_schedule=k_prime_schedule,
        )
        if score_fn is not None:
            # Dynamically get the score function from the scores module
            if not hasattr(scores, score_fn):
                available_fns = [name for name in dir(scores) if not name.startswith('_') and callable(getattr(scores, name))]
                raise ValueError(
                    f"Score function '{score_fn}' not found in pylate.scores. "
                    f"Available functions: {', '.join(available_fns)}"
                )
            base_score_fn = getattr(scores, score_fn)
        else:
            if training_method == "distillation":
                base_score_fn = scores.xtr_kd_training_scores
            else:
                base_score_fn = (
                    scores.xtr_contrastive_training_scores_multiple_negatives
                    if score_all_docs_at_once
                    else scores.xtr_contrastive_training_scores
                )

        scheduled_xtr_kwargs = dict(
            score_fn=base_score_fn,
            k_prime_scheduler=k_prime_scheduler_fn,
            use_normalizer_Z=use_normalizer_Z,
            Z_clamp_value=Z_clamp_value,
            start_normalizer_Z_at_step=start_normalizer_Z_at_step,
        )
        if training_method == "contrastive":
            scheduled_xtr_kwargs["impute_scores_instead_of_zero"] = impute_scores_instead_of_zero

        return scores.ScheduledXTRScore(**scheduled_xtr_kwargs)


def create_loss_function(
    training_method: Literal["distillation", "contrastive", "xtr_mixedbread"],
    model: models.ColBERT,
    score_fn,
    kd_minmax_normalize: bool = False,
    k_prime: int = 100,
    auxiliary_loss_weight: Optional[float] = None,
    score_all_docs_at_once: bool = False,
    positive_document_index: int = 0,
):
    """Create the appropriate loss function based on training method."""
    match training_method:
        case "distillation":
            return losses.Distillation(
                model=model,
                score_metric=score_fn,
                normalize_scores=kd_minmax_normalize,
            )
        case "contrastive":
            return losses.Contrastive(
                model=model,
                score_metric=score_fn,
                do_auxiliary_loss=(k_prime, auxiliary_loss_weight) if auxiliary_loss_weight is not None else None,
                score_all_docs_at_once=score_all_docs_at_once,
                positive_document_index=positive_document_index,
            )
        case "xtr_mixedbread":
            return losses.XTR(
                model=model,
                k_prime=k_prime,
            )
        case _:
            raise ValueError(f"Unknown training method: {training_method}")


def create_evaluators(
    training_method: Literal["distillation", "contrastive", "xtr_mixedbread"],
    use_triplet_evaluator: bool,
    use_nanobeir_evaluator: bool,
    eval_dataset=None,
    nanobeir_datasets=None,
    batch_size: int = 128,
):
    """Create evaluators based on configuration."""
    evaluators = []

    if use_nanobeir_evaluator:
        evaluator_kwargs = {}
        if nanobeir_datasets:
            evaluator_kwargs["dataset_names"] = nanobeir_datasets
        evaluator_kwargs["batch_size"] = batch_size
        evaluators.append(evaluation.NanoBEIREvaluator(**evaluator_kwargs))

    if use_triplet_evaluator:
        if training_method in ("contrastive", "xtr_mixedbread") and eval_dataset is not None:
            evaluators.append(
                evaluation.ColBERTTripletEvaluator(
                    anchors=eval_dataset["query"],
                    positives=eval_dataset["positive"],
                    negatives=eval_dataset["negative"],
                    batch_size=batch_size,
                )
            )
        elif training_method == "distillation":
            print(
                "Warning: Triplet evaluator requires eval_dataset. "
                "Skipping triplet evaluator for distillation."
            )

    # If no evaluators specified, use default based on training method
    if not evaluators:
        if training_method == "distillation":
            evaluators.append(evaluation.NanoBEIREvaluator(batch_size=batch_size))
        elif training_method in ("contrastive", "xtr_mixedbread") and eval_dataset is not None:
            evaluators.append(
                evaluation.ColBERTTripletEvaluator(
                    anchors=eval_dataset["query"],
                    positives=eval_dataset["positive"],
                    negatives=eval_dataset["negative"],
                    batch_size=batch_size,
                )
            )

    # Return single evaluator, list of evaluators, or None
    # SentenceTransformerTrainer accepts both single evaluator and list
    if len(evaluators) == 1:
        return evaluators[0]
    elif len(evaluators) > 1:
        return evaluators
    else:
        return None


def build_run_name(
    training_method: Literal["distillation", "contrastive", "xtr_mixedbread"],
    model_name: str,
    use_colbert: bool,
    lr: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    k_prime: int,
    k_prime_start: int,
    k_prime_schedule: str,
    k_prime_anneal_steps: int,
    use_normalizer_Z: bool,
    Z_clamp_value: float,
    start_normalizer_Z_at_step: int,
    impute_scores_instead_of_zero: bool,
    kd_minmax_normalize: bool = False,
    run_name_override: str = None,
):
    """Build run name from configuration."""
    if run_name_override:
        return run_name_override

    score_type = "ColBERT" if use_colbert else "XTR"

    run_name_parts = [
        # f"model={model_name.split('/')[-1]}",
        f"score={score_type}",
        # f"lr={lr}",
        f"batch_size={batch_size}",
        f"acc={gradient_accumulation_steps}",
    ]

    if training_method == "distillation":
        run_name_parts.append(f"minmax_norm={kd_minmax_normalize}")

    if not use_colbert:
        if k_prime_start != k_prime and k_prime_schedule != "constant":
            run_name_parts.append(
                f"k_prime={k_prime_schedule}[{k_prime_start}_{k_prime}_{k_prime_anneal_steps}]"
            )
        else:
            run_name_parts.append(f"k_prime={k_prime}")

        if use_normalizer_Z:
            run_name_parts.append(f"use_normalizer_Z=True")
            run_name_parts.append(f"Z_clamp_value={Z_clamp_value}")
            if start_normalizer_Z_at_step != 0:
                run_name_parts.append(f"start_normalizer_Z_at_step={start_normalizer_Z_at_step}")
        else:
            run_name_parts.append(f"use_normalizer_Z=False")

    if impute_scores_instead_of_zero:
        run_name_parts.append(f"impute_scores=True")

    run_name = f"{score_type}-{training_method}-[{']['.join(run_name_parts)}]"
    return run_name


def main():
    """Main training function."""
    args = parse_arguments()

    # Extract arguments
    training_method = args.training_method
    auxiliary_loss_weight = args.auxiliary_loss_weight
    model_name = args.model_name.strip("/")
    query_length = args.query_length
    doc_length = args.doc_length
    distillation_train_dataset_path = args.distillation_train_dataset
    train_dataset_subset = args.train_dataset_subset
    contrastive_dataset_path = args.contrastive_dataset
    eval_split_ratio = args.eval_split_ratio
    use_colbert = args.use_colbert
    k_prime = args.k_prime
    k_prime_start = args.k_prime_start if args.k_prime_start is not None else k_prime
    k_prime_anneal_steps = args.k_prime_anneal_steps
    k_prime_schedule = args.k_prime_schedule
    use_normalizer_Z = args.use_normalizer_Z
    Z_clamp_value = args.Z_clamp_value
    start_normalizer_Z_at_step = args.start_normalizer_Z_at_step
    impute_scores_instead_of_zero = args.impute_scores_instead_of_zero
    score_fn = args.score_fn
    score_all_docs_at_once = args.score_all_docs_at_once
    positive_document_index = args.positive_document_index
    lr = args.lr
    batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_acc_steps
    num_train_epochs = args.num_train_epochs
    num_train_steps = args.num_train_steps
    warmup_ratio = args.warmup_ratio
    kd_minmax_normalize = args.kd_minmax_normalize
    use_triplet_evaluator = args.use_triplet_evaluator
    use_nanobeir_evaluator = args.use_nanobeir_evaluator
    nanobeir_datasets = args.nanobeir_datasets
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    run_name_override = args.run_name
    nways = args.nways

    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Training Method: {training_method}")
    if auxiliary_loss_weight is not None:
        print(f"auxiliary Loss Weight: {auxiliary_loss_weight}")
    print(f"Model: {model_name}")
    print(f"Query Length: {query_length}, Doc Length: {doc_length}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}, Gradient Accumulation Steps: {gradient_accumulation_steps}")
    if num_train_steps > 0:
        print(f"Training Steps: {num_train_steps} (overrides epochs = {num_train_epochs})")
    else:
        print(f"Epochs: {num_train_epochs}")
    print(f"Use ColBERT: {use_colbert}")
    if not use_colbert:
        print(f"k_prime: {k_prime}, k_prime_start: {k_prime_start}")
        print(f"k_prime_schedule: {k_prime_schedule}, anneal_steps: {k_prime_anneal_steps}")
        print(f"use_normalizer_Z: {use_normalizer_Z}, Z_clamp: {Z_clamp_value}")
        if score_fn:
            print(f"Score Function: {score_fn} (custom)")
    if training_method == "distillation":
        subset_display = train_dataset_subset if train_dataset_subset else "train (default)"
        print(f"Train Dataset: {distillation_train_dataset_path} (subset: {subset_display})")
        print(f"KD MinMax Normalize: {kd_minmax_normalize}")
    else:  # contrastive or xtr_mixedbread
        subset_display = f"/{train_dataset_subset}" if train_dataset_subset else " (no subset)"
        print(f"Train Dataset: {contrastive_dataset_path}{subset_display}")
        if training_method == "xtr_mixedbread":
            print(f"k_prime: {k_prime}")
    print(f"Evaluators: Triplet={use_triplet_evaluator}, NanoBEIR={use_nanobeir_evaluator}")
    print("=" * 80)

    # Load datasets
    if training_method == "distillation":
        train, queries, documents = load_distillation_datasets(distillation_train_dataset_path, name=train_dataset_subset)
        train.set_transform(
            utils.KDProcessing(queries=queries, documents=documents, n_ways=nways).transform
        )
        train_dataset = train
        eval_dataset = None
    else:  # contrastive or xtr_mixedbread - both use contrastive dataset format
        train_dataset, eval_dataset = load_contrastive_datasets(
            contrastive_dataset_path, dataset_name=train_dataset_subset, eval_split_ratio=eval_split_ratio
        )

    # Build run name
    run_name = build_run_name(
        training_method=training_method,
        model_name=model_name,
        use_colbert=use_colbert,
        lr=lr,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        k_prime=k_prime,
        k_prime_start=k_prime_start,
        k_prime_schedule=k_prime_schedule,
        k_prime_anneal_steps=k_prime_anneal_steps,
        use_normalizer_Z=use_normalizer_Z,
        Z_clamp_value=Z_clamp_value,
        start_normalizer_Z_at_step=start_normalizer_Z_at_step,
        kd_minmax_normalize=kd_minmax_normalize,
        run_name_override=run_name_override,
        impute_scores_instead_of_zero=impute_scores_instead_of_zero,
    )

    output_dir = f"output/{run_name}"
    print(f"Output Directory: {output_dir}")
    print(f"Run Name: {run_name}")

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
        document_length=doc_length,
        do_query_expansion=True,
        attend_to_expansion_tokens=True,
    )

    # Create score function (returns None for xtr_mixedbread)
    score_fn_obj = create_score_function(
        training_method=training_method,
        use_colbert=use_colbert,
        k_prime=k_prime,
        k_prime_start=k_prime_start,
        k_prime_anneal_steps=k_prime_anneal_steps,
        k_prime_schedule=k_prime_schedule,
        use_normalizer_Z=use_normalizer_Z,
        Z_clamp_value=Z_clamp_value,
        start_normalizer_Z_at_step=start_normalizer_Z_at_step,
        impute_scores_instead_of_zero=impute_scores_instead_of_zero,
        score_all_docs_at_once=score_all_docs_at_once,
        score_fn=score_fn,
    )

    # Create loss function
    train_loss = create_loss_function(
        training_method=training_method,
        model=model,
        score_fn=score_fn_obj,
        kd_minmax_normalize=kd_minmax_normalize,
        k_prime=k_prime,
        auxiliary_loss_weight=auxiliary_loss_weight,
        score_all_docs_at_once=score_all_docs_at_once,
        positive_document_index=positive_document_index,
    )

    # print(f"\n\n\nOVERRIDING TRAIN LOSS TO TEST MULTIPLE SCORE FUNCTIONS (MULTIPLE K_PRIME VALUES). REMOVE THIS BEFORE DOING ANYTHING ELSE, ROHAN!\n\n\n")
    # input("Press Enter to acknowledge you read this...")

    # score_fns = [
    #     create_score_function(
    #         training_method="contrastive",
    #         use_colbert=False,
    #         k_prime=k,
    #         k_prime_start=k,
    #         k_prime_anneal_steps=k_prime_anneal_steps,
    #         k_prime_schedule="constant",
    #         use_normalizer_Z=False,
    #         Z_clamp_value=Z_clamp_value,
    #         start_normalizer_Z_at_step=start_normalizer_Z_at_step,
    #         impute_scores_instead_of_zero=impute_scores_instead_of_zero,
    #         score_fn="xtr_contrastive_training_scores",
    #     )
    #     for k in [64, 128, 256, 512, 1024, 2048]
    # ]

    # score_metric = [
    #     (score_fn, 1.0) for score_fn in score_fns
    # ]

    # print(f"{score_metric=}")
    # train_loss = losses.Contrastive(
    #     model=model,
    #     score_metric=score_metric,
    # )

    # Create evaluators
    evaluator = create_evaluators(
        training_method=training_method,
        use_triplet_evaluator=use_triplet_evaluator,
        use_nanobeir_evaluator=use_nanobeir_evaluator,
        eval_dataset=eval_dataset,
        nanobeir_datasets=nanobeir_datasets,
        batch_size=batch_size,
    )

    # Configure training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=num_train_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_on_start=True,
        save_steps=save_steps,
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        disable_tqdm=not is_main_process(),
        log_on_each_node=False,
        torch_compile=True,
    )        

    # Add eval batch size for contrastive and xtr_mixedbread training
    if training_method in ("contrastive", "xtr_mixedbread"):
        training_args.per_device_eval_batch_size = batch_size

    # Initialize trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "loss": train_loss,
        "data_collator": utils.ColBERTCollator(tokenize_fn=model.tokenize),
    }

    if training_method in ("contrastive", "xtr_mixedbread"):
        trainer_kwargs["eval_dataset"] = eval_dataset

    if evaluator is not None:
        trainer_kwargs["evaluator"] = evaluator

    trainer = SentenceTransformerTrainer(**trainer_kwargs)

    # Add k_prime callback for XTR (only for scheduled score functions, not for xtr_mixedbread)
    if not use_colbert and training_method != "xtr_mixedbread" and score_fn_obj is not None:
        trainer.add_callback(scores.KPrimeSchedulerCallback(score_fn_obj))

    # Start training
    trainer.train()

    # Save final model
    model.save_pretrained(f"{output_dir}/final")
    print(f"Training completed! Final model saved to {output_dir}/final")


if __name__ == "__main__":
    main()

