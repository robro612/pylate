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
from typing import Literal

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
        choices=["distillation", "contrastive"],
        default="distillation",
        help="Training method: 'distillation' or 'contrastive'",
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
        "--train_dataset",
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
        "--contrastive_dataset_name",
        type=str,
        default="triplet",
        help="Dataset name/split for contrastive training",
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


def load_distillation_datasets(train_dataset_path: str):
    """Load datasets for knowledge distillation training."""
    train = load_dataset(path=train_dataset_path, name="train")
    queries = load_dataset(path=train_dataset_path, name="queries")
    documents = load_dataset(path=train_dataset_path, name="documents")
    return train, queries, documents


def load_contrastive_datasets(
    dataset_path: str, dataset_name: str, eval_split_ratio: float
):
    """Load datasets for contrastive training."""
    dataset = load_dataset(dataset_path, dataset_name, split="train")
    splits = dataset.train_test_split(test_size=eval_split_ratio)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
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
    training_method: Literal["distillation", "contrastive"],
    use_colbert: bool,
    k_prime: int,
    k_prime_start: int,
    k_prime_anneal_steps: int,
    k_prime_schedule: str,
    use_normalizer_Z: bool,
    Z_clamp_value: float,
    start_normalizer_Z_at_step: int,
):
    """Create the appropriate score function based on training method and configuration."""
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

        if training_method == "distillation":
            base_score_fn = scores.xtr_kd_training_scores
        else:
            base_score_fn = scores.xtr_contrastive_training_scores

        return scores.ScheduledXTRScore(
            score_fn=base_score_fn,
            k_prime_scheduler=k_prime_scheduler_fn,
            use_normalizer_Z=use_normalizer_Z,
            Z_clamp_value=Z_clamp_value,
            start_normalizer_Z_at_step=start_normalizer_Z_at_step,
        )


def create_loss_function(
    training_method: Literal["distillation", "contrastive"],
    model: models.ColBERT,
    score_fn,
    kd_minmax_normalize: bool = False,
):
    """Create the appropriate loss function based on training method."""
    if training_method == "distillation":
        return losses.Distillation(
            model=model,
            score_metric=score_fn,
            normalize_scores=kd_minmax_normalize,
        )
    else:
        return losses.Contrastive(
            model=model,
            score_metric=score_fn,
        )


def create_evaluators(
    training_method: Literal["distillation", "contrastive"],
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
        if training_method == "contrastive" and eval_dataset is not None:
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
        elif training_method == "contrastive" and eval_dataset is not None:
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
    training_method: Literal["distillation", "contrastive"],
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

    run_name = f"{score_type}-{training_method}-[{']['.join(run_name_parts)}]"
    return run_name


def main():
    """Main training function."""
    args = parse_arguments()

    # Extract arguments
    training_method = args.training_method
    model_name = args.model_name.strip("/")
    query_length = args.query_length
    doc_length = args.doc_length
    train_dataset_path = args.train_dataset
    contrastive_dataset_path = args.contrastive_dataset
    contrastive_dataset_name = args.contrastive_dataset_name
    eval_split_ratio = args.eval_split_ratio
    use_colbert = args.use_colbert
    k_prime = args.k_prime
    k_prime_start = args.k_prime_start if args.k_prime_start is not None else k_prime
    k_prime_anneal_steps = args.k_prime_anneal_steps
    k_prime_schedule = args.k_prime_schedule
    use_normalizer_Z = args.use_normalizer_Z
    Z_clamp_value = args.Z_clamp_value
    start_normalizer_Z_at_step = args.start_normalizer_Z_at_step
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

    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Training Method: {training_method}")
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
    if training_method == "distillation":
        print(f"Train Dataset: {train_dataset_path}")
        print(f"KD MinMax Normalize: {kd_minmax_normalize}")
    else:
        print(f"Contrastive Dataset: {contrastive_dataset_path}/{contrastive_dataset_name}")
    print(f"Evaluators: Triplet={use_triplet_evaluator}, NanoBEIR={use_nanobeir_evaluator}")
    print("=" * 80)

    # Load datasets
    if training_method == "distillation":
        train, queries, documents = load_distillation_datasets(train_dataset_path)
        train.set_transform(
            utils.KDProcessing(queries=queries, documents=documents).transform
        )
        train_dataset = train
        eval_dataset = None
    else:
        train_dataset, eval_dataset = load_contrastive_datasets(
            contrastive_dataset_path, contrastive_dataset_name, eval_split_ratio
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

    # Create score function
    score_fn = create_score_function(
        training_method=training_method,
        use_colbert=use_colbert,
        k_prime=k_prime,
        k_prime_start=k_prime_start,
        k_prime_anneal_steps=k_prime_anneal_steps,
        k_prime_schedule=k_prime_schedule,
        use_normalizer_Z=use_normalizer_Z,
        Z_clamp_value=Z_clamp_value,
        start_normalizer_Z_at_step=start_normalizer_Z_at_step,
    )

    # Create loss function
    train_loss = create_loss_function(
        training_method=training_method,
        model=model,
        score_fn=score_fn,
        kd_minmax_normalize=kd_minmax_normalize,
    )

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

    # Add eval batch size for contrastive training
    if training_method == "contrastive":
        training_args.per_device_eval_batch_size = batch_size

    # Initialize trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "loss": train_loss,
        "data_collator": utils.ColBERTCollator(tokenize_fn=model.tokenize),
    }

    if training_method == "contrastive":
        trainer_kwargs["eval_dataset"] = eval_dataset

    if evaluator is not None:
        trainer_kwargs["evaluator"] = evaluator

    trainer = SentenceTransformerTrainer(**trainer_kwargs)

    # Add k_prime callback for XTR
    if not use_colbert:
        trainer.add_callback(scores.KPrimeSchedulerCallback(score_fn))

    # Start training
    trainer.train()

    # Save final model
    model.save_pretrained(f"{output_dir}/final")
    print(f"Training completed! Final model saved to {output_dir}/final")


if __name__ == "__main__":
    main()

