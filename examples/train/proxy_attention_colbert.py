"""Script to train ProxyAttentionColBERT using Knowledge Distillation on MS MARCO.

ProxyAttentionColBERT uses learnable proxy tokens to select the most salient document
tokens via attention-based scoring, reducing the number of stored embeddings while
maintaining retrieval quality.
"""

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils

# Load the datasets required for knowledge distillation (train, queries, documents)
print("Loading train dataset...")
train = load_dataset(
    path="lightonai/ms-marco-en-bge-gemma",
    name="train",
)

queries = load_dataset(
    path="lightonai/ms-marco-en-bge-gemma",
    name="queries",
)

documents = load_dataset(
    path="lightonai/ms-marco-en-bge-gemma",
    name="documents",
)

# Set the transformation to load the documents/queries texts using the corresponding ids on the fly
print("Setting up dataset transformation...")
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents).transform,
)

# Define the base model, training parameters, and output directory
# model_name = "Alibaba-NLP/gte-modernbert-base"
model_name = "lightonai/GTE-ModernColBERT-v1"
batch_size = 12
lr = 3e-5
num_train_epochs = 3

# ProxyAttentionColBERT specific parameters
num_proxy_tokens = 32  # Number of learnable proxy query tokens
num_select_tokens = 32  # Number of document tokens to select (controls storage)
use_cluster_pooling = True  # Use cluster-based pooling for selected tokens
proxy_tau = 1.0  # Temperature for saliency softmax

# Set the run name for logging and output directory
run_name = f"ProxyAttention-ColBERT-{num_select_tokens}tok-{lr}-lr-{num_train_epochs}-epochs"
output_dir = f"output/{run_name}"

# Initialize the ProxyAttentionColBERT model
print("Initializing model...")
model = models.ProxyAttentionColBERT(
    model_name_or_path=model_name,
    document_length=300,
    num_proxy_tokens=num_proxy_tokens,
    num_select_tokens=num_select_tokens,
    use_cluster_pooling=use_cluster_pooling,
    proxy_tau=proxy_tau,
)

dev_evaluator = evaluation.NanoBEIREvaluator()

# Configure the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=5000,
    logging_steps=20,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,
    learning_rate=lr,
    warmup_ratio=0.00,
)

# Use the ProxyAttentionDistillation loss function for training
train_loss = losses.ProxyAttentionDistillation(model=model)

# Initialize the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)

# Start the training process
print("Training...")
trainer.train()
model.save_pretrained(f"{output_dir}/final")

