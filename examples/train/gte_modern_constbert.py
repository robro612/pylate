"""Script to reproduce the training of GTE-ModernColBERT using Knowledge Distillation on MS MARCO with Gemma reranker."""

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


# Define the base model, training parameters, and output directory
model_name = "Alibaba-NLP/gte-modernbert-base"
batch_size = 24
n_ways = 16
lr = 3e-5
constbert_output_length = 32

# Load the datasets required for knowledge distillation (train, queries, documents)
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
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents, n_ways=n_ways).transform,
)

# Set the run name for logging and output directory
run_name = f"GTE-ModernColBERT-bs{batch_size}nway{n_ways}-ConstBERT-C-{constbert_output_length}"
output_dir = f"output/{run_name}"

# Initialize the ColBERT model from the base model
model = models.ConstBERT(model_name_or_path=model_name, document_length=300, constbert_output_length=constbert_output_length)

dev_evaluator = evaluation.NanoBEIREvaluator()
# Configure the training arguments (e.g., epochs, batch size, learning rate)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    max_steps=25_000,
    per_device_train_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=10_000,
    logging_steps=10,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,
    learning_rate=lr,
    warmup_ratio=0.00,
)

# Use the Distillation loss function for training
train_loss = losses.Distillation(model=model)

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
trainer.train()
model.save_pretrained(f"{output_dir}/final")
