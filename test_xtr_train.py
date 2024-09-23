from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate.scores import colbert_xtr_train_scores

from pylate import losses, models, utils

from functools import partial

print("Loading Data")
# Load the datasets required for knowledge distillation (train, queries, documents)
train = load_dataset(
    path="lightonai/ms-marco-en-bge",
    name="train",
    keep_in_memory=True,
    num_proc=8,
)

queries = load_dataset(
    path="lightonai/ms-marco-en-bge",
    name="queries",
    keep_in_memory=True,
    num_proc=8,
)

documents = load_dataset(
    path="lightonai/ms-marco-en-bge",
    name="documents",
    keep_in_memory=True,
    num_proc=8,
)
print("Data Loaded")

print("Transforming Data")
# Set the transformation to load the documents/queries texts using the corresponding ids on the fly
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents, n_ways=16).transform,
)
print("Data Transformed")

print("Creating Model")
# Define model parameters for contrastive training
model_name = "nreimers/MiniLM-L6-H384-uncased"  # Choose the pre-trained model you want to use as base
batch_size = 8  # Larger batch size often improves results, but requires more memory

num_train_epochs = 1  # Adjust based on your requirements
# Set the run name for logging and output directory
run_name = "kd-minilm-test"
output_dir = f"output/{run_name}"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
model = models.ColBERT(model_name_or_path=model_name)

# Compiling the model makes the training faster
# model = torch.compile(model)

print("Model Created")

# Use the Distillation loss function for training
train_loss = losses.Distillation(
    model=model,
    score_metric=partial(colbert_xtr_train_scores, k_prime=(16 * 180 - 1)),
)

# Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    learning_rate=3e-6,
    logging_steps=10,
    save_steps=5000,
    save_total_limit=5,
)

# Initialize the trainer for the contrastive training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    # eval_dataset=eval_dataset,
    loss=train_loss,
    # evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(model.tokenize),
)

# Start the training process
trainer.train()
