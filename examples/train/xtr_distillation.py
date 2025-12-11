"""Script to reproduce the training of GTE-ModernColBERT using Knowledge Distillation on MS MARCO with Gemma reranker."""
from __future__ import annotations

import torch
import os

def is_main_process():
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

if not is_main_process():
    os.environ["WANDB_DISABLED"] = "true"

from argparse import ArgumentParser
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils, scores




parser = ArgumentParser()
parser.add_argument("--use_normalizer_Z", action="store_true")
parser.add_argument("--Z_clamp_value", type=float, default=1.0)
parser.add_argument("--start_normalizer_Z_at_step", type=int, default=0)
parser.add_argument("--k_prime", type=int, default=128)
parser.add_argument("--k_prime_start", type=int, default=None, help="Starting k_prime for annealing (defaults to k_prime if not set)")
parser.add_argument("--k_prime_anneal_steps", type=int, default=1000, help="Number of steps over which to anneal k_prime")
parser.add_argument("--k_prime_schedule", type=str, default="constant", choices=["linear", "exponential", "constant"], help="Schedule type for k_prime annealing")
parser.add_argument("--lr", type=float, default=3e-6)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--grad_acc_steps", type=int, default=1)
parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
parser.add_argument("--train_dataset", type=str, default="lightonai/ms-marco-en-bge-gemma", help="Train dataset to use")
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=10_000)
parser.add_argument("--run_name", type=str, default=None, help="Override the default run name")
parser.add_argument("--kd_minmax_normalize", action="store_true", help="Normalize the scores using min-max normalization")
parser.add_argument("--use_colbert", action="store_true", help="Use ColBERT score function instead of XTR (baseline)")
parser.add_argument("--query_length", type=int, default=32, help="Query length for ColBERT model")
parser.add_argument("--doc_length", type=int, default=300, help="Document length for ColBERT model")
parser.add_argument("--warmup_ratio", type=float, default=0.005, help="Warmup ratio for the learning rate scheduler")
args = parser.parse_args()

use_normalizer_Z = args.use_normalizer_Z
Z_clamp_value = args.Z_clamp_value
start_normalizer_Z_at_step = args.start_normalizer_Z_at_step
k_prime = args.k_prime
k_prime_start = args.k_prime_start if args.k_prime_start is not None else k_prime
k_prime_anneal_steps = args.k_prime_anneal_steps
k_prime_schedule = args.k_prime_schedule
lr = args.lr
batch_size = args.batch_size
gradient_accumulation_steps = args.grad_acc_steps
kd_minmax_normalize = args.kd_minmax_normalize
use_colbert = args.use_colbert
query_length = args.query_length
doc_length = args.doc_length
warmup_ratio = args.warmup_ratio
model_name = args.model_name.strip("/")
train_dataset = args.train_dataset
num_train_epochs = args.num_train_epochs
eval_steps = args.eval_steps
save_steps = args.save_steps
run_name_override = args.run_name

print(
    f"use_normalizer_Z: {use_normalizer_Z}\n"
    f"k_prime: {k_prime}\n"
    f"k_prime_start: {k_prime_start}\n"
    f"k_prime_anneal_steps: {k_prime_anneal_steps}\n"
    f"schedule: {k_prime_schedule}\n"
    f"lr: {lr}\n"
    f"batch_size: {batch_size}\n"
    f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
    f"model_name: {model_name}\n"
    f"train_dataset: {train_dataset}\n"
    f"num_train_epochs: {num_train_epochs}\n"
    f"eval_steps: {eval_steps}\n"
    f"save_steps: {save_steps}\n"
    f"run_name_override: {run_name_override}\n"
    f"warmup_ratio: {warmup_ratio}\n"
    f"query_length: {query_length}\n"
    f"doc_length: {doc_length}\n"
    f"use_colbert: {use_colbert}\n"
    f"kd_minmax_normalize: {kd_minmax_normalize}\n"
)

# Load the datasets required for knowledge distillation (train, queries, documents)
train = load_dataset(
    path=train_dataset,
    name="train",
)

queries = load_dataset(
    path=train_dataset,
    name="queries",
)

documents = load_dataset(
    path=train_dataset,
    name="documents",
)

# Set the transformation to load the documents/queries texts using the corresponding ids on the fly
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents).transform,
)

# Set the run name for logging and output directory
if run_name_override:
    run_name = run_name_override
else:
    # Build run name with bracket format
    score_type = "ColBERT" if use_colbert else "XTR"
    run_name_parts = [
        f"model={model_name.split('/')[-1]}",
        f"score={score_type}",
        f"lr={lr}",
        f"batch_size={batch_size}",
        f"acc={gradient_accumulation_steps}",
        f"minmax_norm={kd_minmax_normalize}",
    ]
    
    if not use_colbert:
        if k_prime_start != k_prime:
            run_name_parts.append(f"k_prime={k_prime_schedule}[{k_prime_start}_{k_prime}_{k_prime_anneal_steps}]")
        else:
            run_name_parts.append(f"k_prime={k_prime}")

        if use_normalizer_Z:
            run_name_parts.append(f"use_normalizer_Z=True")
            run_name_parts.append(f"Z_clamp_value={Z_clamp_value}")
            run_name_parts.append(
                f"start_normalizer_Z_at_step={start_normalizer_Z_at_step}"
            )
        else:
            run_name_parts.append(f"use_normalizer_Z=False")

    run_name = f"{'ColBERT' if use_colbert else 'XTR'}-KD-[{']['.join(run_name_parts)}]"

output_dir = f"output/{run_name}"

# Initialize the ColBERT model from the base model
model = models.ColBERT(
    model_name_or_path=model_name,
    query_length=query_length,
    document_length=doc_length,
    do_query_expansion=True,
    attend_to_expansion_tokens=True,
)

dev_evaluator = evaluation.NanoBEIREvaluator()
# Configure the training arguments (e.g., epochs, batch size, learning rate)
training_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_strategy="steps",
    eval_steps=eval_steps,
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

# Define score function based on use_colbert flag
if use_colbert:
    # Use ColBERT score function (baseline)
    score_fn = scores.colbert_kd_scores
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
    score_fn = scores.ScheduledXTRScore(
        score_fn=scores.xtr_kd_training_scores,
        k_prime_scheduler=k_prime_scheduler,
        use_normalizer_Z=use_normalizer_Z,
        Z_clamp_value=Z_clamp_value,
        start_normalizer_Z_at_step=start_normalizer_Z_at_step,
    )

# Use the Distillation loss function for training
train_loss = losses.Distillation(
    model=model,
    score_metric=score_fn,
    normalize_scores=kd_minmax_normalize,
)

# dev_evaluator = evaluation.NanoBEIREvaluator()

# Initialize the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)

# Add the callback to update k_prime (only needed for XTR, not ColBERT)
if not use_colbert:
    trainer.add_callback(scores.KPrimeSchedulerCallback(score_fn))

# Start the training process

trainer.train()
model.save_pretrained(f"{output_dir}/final")
