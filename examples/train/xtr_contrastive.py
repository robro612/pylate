from __future__ import annotations

import torch
from argparse import ArgumentParser
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

from pylate import evaluation, losses, models, utils, scores


parser = ArgumentParser()
parser.add_argument("--use_normalizer_Z", action="store_true")
parser.add_argument("--Z_clamp_value", type=float, default=1.0)
parser.add_argument("--start_normalizer_Z_at_step", type=int, default=0)
parser.add_argument("--k_prime", type=int, default=100)
parser.add_argument("--lr", type=float, default=3e-6)
parser.add_argument("--k_prime_start", type=int, default=None, help="Starting k_prime for annealing (defaults to k_prime if not set)")
parser.add_argument("--k_prime_anneal_steps", type=int, default=1000, help="Number of steps over which to anneal k_prime")
parser.add_argument("--k_prime_schedule", type=str, default="constant", choices=["linear", "exponential", "constant"], help="Schedule type for k_prime annealing")
parser.add_argument("--use_colbert", action="store_true", help="Use ColBERT score function instead of XTR (baseline)")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
parser.add_argument("--run_name", type=str, default=None, help="Override the default run name")
args = parser.parse_args()

use_normalizer_Z = args.use_normalizer_Z
Z_clamp_value = args.Z_clamp_value
k_prime = args.k_prime
lr = args.lr
start_normalizer_Z_at_step = args.start_normalizer_Z_at_step
k_prime_start = args.k_prime_start if args.k_prime_start is not None else k_prime
k_prime_anneal_steps = args.k_prime_anneal_steps
k_prime_schedule = args.k_prime_schedule
use_colbert = args.use_colbert
run_name_override = args.run_name
print(f"use_normalizer_Z: {use_normalizer_Z}, k_prime: {k_prime}, k_prime_start: {k_prime_start}, k_prime_anneal_steps: {k_prime_anneal_steps}, schedule: {k_prime_schedule}, use_colbert: {use_colbert}")

# Define model parameters for contrastive training
# model_name = "bert-base-uncased"  # Choose the pre-trained model you want to use as base
model_name = args.model_name
batch_size = args.batch_size  # Larger batch size often improves results, but requires more memory

num_train_epochs = 1  # Adjust based on your requirements
# Set the run name for logging and output directory

if run_name_override:
    run_name = run_name_override
else:
    # Build run name with bracket format
    score_type = "ColBERT" if use_colbert else "XTR"
    run_name_parts = [
        f"model=ModernBERT-base",
        f"score={score_type}",
        f"lr={lr}",
        f"batch_size={batch_size}",
    ]
    
    if not use_colbert:
        if k_prime_start and k_prime_start != k_prime:
            run_name_parts.append(f"k_prime={k_prime_schedule}[{k_prime_start}_{k_prime}_{k_prime_anneal_steps}]")
        else:
            run_name_parts.append(f"k_prime={k_prime}")
    
    if use_normalizer_Z:
        run_name_parts.append(f"use_normalizer_Z=True")
        run_name_parts.append(f"Z_clamp_value={Z_clamp_value}")
        run_name_parts.append(f"start_normalizer_Z_at_step={start_normalizer_Z_at_step}")
    
    run_name = f"contrastive-[{']['.join(run_name_parts)}]"

output_dir = f"output/{run_name}"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
model = models.ColBERT(
    model_name_or_path=model_name,
    query_length=32,
    document_length=300,
    do_query_expansion=True,
    attend_to_expansion_tokens=True,
)

# Load dataset
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
# Split the dataset (this dataset does not have a validation set, so we split the training set)
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# Define score function based on use_colbert flag
if use_colbert:
    # Use ColBERT score function (baseline)
    score_fn = scores.colbert_scores
else:
    # Define k_prime scheduler function
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

    # Create the scheduled score function
    score_fn = scores.ScheduledXTRContrastiveScore(
        k_prime_scheduler=k_prime_scheduler,
        use_normalizer_Z=use_normalizer_Z,
        Z_clamp_value=Z_clamp_value,
        start_normalizer_Z_at_step=start_normalizer_Z_at_step,
    )

# Define the loss function
train_loss = losses.Contrastive(
    model=model,
    score_metric=score_fn,
)

# Initialize the evaluator
triplet_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)

# Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
training_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    bf16=True,
    fp16=False,
    learning_rate=lr,
    run_name=run_name,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=1000,
)

# Callback to update k_prime scheduler with current training step (only needed for XTR)
class KPrimeSchedulerCallback(TrainerCallback):
    """Callback to update k_prime scheduler with current training step."""
    
    def __init__(self, scheduled_score_fn):
        self.scheduled_score_fn = scheduled_score_fn
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update the step in the scheduled score function."""
        if hasattr(self.scheduled_score_fn, 'update_step'):
            self.scheduled_score_fn.update_step(state.global_step)
        return control

# Initialize the trainer for the contrastive training
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=triplet_evaluator,
    data_collator=utils.ColBERTCollator(model.tokenize),
)

# Add the callback to update k_prime (only needed for XTR, not ColBERT)
if not use_colbert:
    trainer.add_callback(KPrimeSchedulerCallback(score_fn))

# Start the training process
trainer.train()
