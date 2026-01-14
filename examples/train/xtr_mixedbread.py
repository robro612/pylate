from __future__ import annotations

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils, scores

import os
# os.environ["WANDB_DISABLED"] = "true"

# Define model parameters for contrastive training
model_name = "answerdotai/ModernBERT-base"  # Choose the pre-trained model you want to use as base
batch_size = 64  # Larger batch size often improves results, but requires more memory

num_train_epochs = 1  # Adjust based on your requirements
# Set the run name for logging and output directory
run_name = "xtr-composite-loss-testing"
output_dir = f"output/{run_name}"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
print("Loading model...")
model = models.ColBERT(model_name_or_path=model_name, query_length=32, document_length=300)
model.compile()
print("Model compiled")

# Load dataset
print("Loading dataset...")
dataset = load_dataset("bclavie/msmarco-10m-triplets", split="train")

# Split the dataset (this dataset does not have a validation set, so we split the training set)
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]

print("Loading loss function...")
# make a composite loss function of contrastive and xtr
train_loss = losses.CompositeLoss(
    model=model,
    losses={
        "contrastive": (losses.ContrastiveFromEmbeddings(score_metric=scores.xtr_contrastive_training_scores), 1.0),
        "xtr": (losses.PairwiseLogisticFromEmbeddings(k_prime=128, log_prefix="xtr"), 0.5),
    },
)


# Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    max_steps=10_000,
    per_device_train_batch_size=batch_size,
    run_name=run_name,
    learning_rate=3e-5,
    bf16=True,
    fp16=False,
    logging_steps=1,
)

print("Initializing trainer...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    data_collator=utils.ColBERTCollator(model.tokenize),
)
print("Starting training...")
trainer.train()
