"""Quantization-aware training (QAT) for ColBERT via a straight-through estimator.

This mirrors ``contrastive.py`` but appends a permanent ``StraightThroughEstimator``
module after the Dense projection. Every embedding the loss sees is passed through
a quantizer on the forward pass while gradients flow straight through to the
full-precision weights -- so the *same* loss functions train the network to be
robust to the chosen quantization.

The estimator is asymmetric here: queries are quantized to int8 and documents are
binarized. The module is saved with the model and is active at inference, so
``encode`` emits embeddings on the same quantization grid used during training.
"""

from __future__ import annotations

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils

model_name = "bert-base-uncased"
batch_size = 32
num_train_epochs = 1
run_name = "qat-int8-query-binary-doc"
output_dir = f"output/{run_name}"

# 1. Base ColBERT model (adds a linear projection layer if the base is not a ColBERT).
model = models.ColBERT(model_name_or_path=model_name)

# 2. Append the quantization-aware training module. Use a single `transform=...`
#    for symmetric quantization, or `query_transform` / `document_transform` for
#    asymmetric quantization as below. Any callable `Tensor -> Tensor` works as a
#    transform; Quantizer subclasses additionally serialize with the model.
model.append(
    models.StraightThroughEstimator(
        query_transform=models.ScalarQuantizer(n_bits=8),  # int8 queries
        document_transform=models.BinaryQuantizer(),  # binary documents
    )
)

# Load dataset
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# Define the loss function (unchanged -- the STE module makes it quantization-aware)
train_loss = losses.Contrastive(model=model)

dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,  # Set to False if your GPU can't run FP16
    bf16=False,
    run_name=run_name,
    learning_rate=3e-6,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(model.tokenize),
)
trainer.train()

# The saved model keeps the StraightThroughEstimator (asymmetric int8/binary)
# and applies it at inference time too.
model.save_pretrained(f"{output_dir}/final")
