"""Script to train ProxyAttentionColBERT using Knowledge Distillation on MS MARCO.

ProxyAttentionColBERT uses learnable proxy tokens to select the most salient document
tokens via attention-based scoring, reducing the number of stored embeddings while
maintaining retrieval quality.

Multi-GPU Training:
    By default, this script uses only 1 GPU. For multi-GPU training, use one of:

    # Option 1: torchrun (recommended)
    torchrun --nproc_per_node=4 proxy_attention_colbert.py

    # Option 2: accelerate
    accelerate launch proxy_attention_colbert.py

    # Option 3: Specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 proxy_attention_colbert.py

Training Modes:
    - "full": Train all parameters (default)
    - "proxy_only": Only train proxy embeddings, freeze everything else
    - "freeze_word_embeddings": Freeze word embeddings, train transformer + proxy
    - "lora": Apply LoRA to transformer, train LoRA + proxy (requires peft)
"""

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def setup_training_mode(model, mode="full", lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Configure which parameters are trainable based on training mode.

    Args:
        model: ProxyAttentionColBERT model
        mode: Training mode - "full", "proxy_only", "freeze_word_embeddings", or "lora"
        lora_r: LoRA rank (only for mode="lora")
        lora_alpha: LoRA alpha (only for mode="lora")
        lora_dropout: LoRA dropout (only for mode="lora")

    Returns:
        model (potentially wrapped with LoRA)
    """
    if mode == "full":
        # All parameters trainable (default)
        print("Training mode: full - all parameters trainable")

    elif mode == "proxy_only":
        # Freeze everything except proxy embeddings
        print("Training mode: proxy_only - freezing all except proxy embeddings")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model._proxy_embeddings.weight.requires_grad = True

    elif mode == "freeze_word_embeddings":
        # Freeze word embeddings only, train transformer + proxy
        print("Training mode: freeze_word_embeddings - freezing word embeddings")
        transformer = model[0]
        word_embeddings = transformer.auto_model.get_input_embeddings()
        for param in word_embeddings.parameters():
            param.requires_grad = False

    elif mode == "lora":
        # Apply LoRA to transformer, train LoRA + proxy
        print(f"Training mode: lora - applying LoRA (r={lora_r}, alpha={lora_alpha})")
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("LoRA mode requires peft library. Install with: pip install peft")

        # First freeze everything
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Apply LoRA to the transformer
        transformer = model[0]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"],
        )
        transformer.auto_model = get_peft_model(transformer.auto_model, lora_config)

        # Unfreeze proxy embeddings
        model._proxy_embeddings.weight.requires_grad = True

    else:
        raise ValueError(f"Unknown training mode: {mode}. Choose from: full, proxy_only, freeze_word_embeddings, lora")

    # Print trainable parameters summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model

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
batch_size = 8
lr = 3e-5
num_train_epochs = 3

# ProxyAttentionColBERT specific parameters
num_proxy_tokens = 32  # Number of learnable proxy query tokens
num_select_tokens = 32  # Number of document tokens to select (controls storage)
use_cluster_pooling = True  # Use cluster-based pooling for selected tokens
proxy_tau = 1.0  # Temperature for saliency softmax

# Training mode: "full", "proxy_only", "freeze_word_embeddings", or "lora"
training_mode = "full"
# LoRA parameters (only used when training_mode="lora")
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1

# Set the run name for logging and output directory
run_name = f"ProxyAttention-ColBERT-{num_select_tokens}tok-{lr}-lr-{num_train_epochs}-epochs-{training_mode}"
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

# Configure training mode (freeze/unfreeze parameters)
model = setup_training_mode(
    model,
    mode=training_mode,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
)

dev_evaluator = evaluation.NanoBEIREvaluator()

# Configure the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,  # Increase for larger effective batch size
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

# Run evaluation first to verify setup
print("Running initial evaluation...")
eval_results = trainer.evaluate()
print(f"Initial evaluation results: {eval_results}")

# Start the training process
print("Training...")
trainer.train()
model.save_pretrained(f"{output_dir}/final")

