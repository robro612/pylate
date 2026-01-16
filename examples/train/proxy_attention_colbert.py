"""Script to train ProxyAttentionColBERT using Knowledge Distillation on MS MARCO.

ProxyAttentionColBERT uses learnable proxy tokens to select the most salient document
tokens via attention-based scoring, reducing the number of stored embeddings while
maintaining retrieval quality.

Multi-GPU Training:
    By default, this script uses only 1 GPU. For multi-GPU training, use one of:

    # Option 1: torchrun (recommended)
    torchrun --nproc_per_node=4 proxy_attention_colbert.py --batch_size 24 --lr 3e-5

    # Option 2: accelerate
    accelerate launch proxy_attention_colbert.py --batch_size 24 --lr 3e-5

    # Option 3: Specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 proxy_attention_colbert.py

Training Modes:
    - "full": Train all parameters (default)
    - "proxy_only": Only train proxy embeddings, freeze everything else
    - "freeze_word_embeddings": Freeze word embeddings, train transformer + proxy
    - "lora": Apply LoRA to transformer, train LoRA + proxy (requires peft)

Example Usage:
    python proxy_attention_colbert.py --model_name Alibaba-NLP/gte-modernbert-base --batch_size 24 --lr 3e-5 --epochs 3
    python proxy_attention_colbert.py --training_mode lora --lora_r 16 --lora_alpha 32
    python proxy_attention_colbert.py --num_select_tokens 64 --num_proxy_tokens 64
"""

import argparse
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ProxyAttentionColBERT using Knowledge Distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Alibaba-NLP/gte-modernbert-base",
        # "lightonai/GTE-ModernColBERT-v1"
        help="Base model name or path",
    )
    parser.add_argument(
        "--document_length",
        type=int,
        default=300,
        help="Maximum document length in tokens",
    )

    # ProxyAttentionColBERT specific arguments
    parser.add_argument(
        "--num_proxy_tokens",
        type=int,
        default=32,
        help="Number of learnable proxy query tokens",
    )
    parser.add_argument(
        "--num_select_tokens",
        type=int,
        default=32,
        help="Number of document tokens to select (controls storage)",
    )
    parser.add_argument(
        "--use_cluster_pooling",
        action="store_true",
        default=True,
        help="Use cluster-based pooling for selected tokens",
    )
    parser.add_argument(
        "--no_cluster_pooling",
        action="store_true",
        help="Disable cluster-based pooling",
    )
    parser.add_argument(
        "--proxy_tau",
        type=float,
        default=1.0,
        help="Temperature for saliency softmax",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--n_ways",
        type=int,
        default=16,
        help="Number of documents per query for KD (controls memory usage)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )

    # Training mode arguments
    parser.add_argument(
        "--training_mode",
        type=str,
        default="full",
        choices=["full", "proxy_only", "freeze_word_embeddings", "lora"],
        help="Training mode: full, proxy_only, freeze_word_embeddings, or lora",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (only used when training_mode=lora)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (only used when training_mode=lora)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (only used when training_mode=lora)",
    )

    # Evaluation and logging arguments
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
        help="Log every N steps",
    )
    parser.add_argument(
        "--skip_initial_eval",
        action="store_true",
        help="Skip initial evaluation before training",
    )

    # Precision arguments
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 mixed precision (default)",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable BF16 mixed precision",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated based on config)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for logging (default: auto-generated based on config)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="lightonai/ms-marco-en-bge-gemma",
        help="HuggingFace dataset path for KD training",
    )

    return parser.parse_args()


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


def main():
    """Main training function."""
    args = parse_args()

    # Handle boolean flags
    use_cluster_pooling = args.use_cluster_pooling and not args.no_cluster_pooling
    use_bf16 = args.bf16 and not args.no_bf16
    use_fp16 = args.fp16 and not use_bf16  # fp16 only if bf16 is disabled

    # Generate run name if not provided
    if args.run_name is None:
        run_name = (
            f"ProxyAttention-ColBERT-{args.num_select_tokens}tok-"
            f"{args.lr}-lr-{args.epochs}-epochs-{args.training_mode}-"
            f"bs{args.batch_size}-nway{args.n_ways}"
        )
    else:
        run_name = args.run_name

    # Generate output directory if not provided
    if args.output_dir is None:
        output_dir = f"output/{run_name}"
    else:
        output_dir = args.output_dir

    print(f"=" * 60)
    print(f"ProxyAttentionColBERT Training")
    print(f"=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Training mode: {args.training_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"N-ways: {args.n_ways}")
    print(f"Num proxy tokens: {args.num_proxy_tokens}")
    print(f"Num select tokens: {args.num_select_tokens}")
    print(f"Output dir: {output_dir}")
    print(f"=" * 60)

    # Load the datasets required for knowledge distillation
    print("Loading train dataset...")
    train = load_dataset(path=args.dataset_path, name="train")
    queries = load_dataset(path=args.dataset_path, name="queries")
    documents = load_dataset(path=args.dataset_path, name="documents")

    # Set the transformation to load the documents/queries texts
    print("Setting up dataset transformation...")
    train.set_transform(
        utils.KDProcessing(
            queries=queries,
            documents=documents,
            n_ways=args.n_ways,
        ).transform,
    )

    # Initialize the ProxyAttentionColBERT model
    print("Initializing model...")
    model = models.ProxyAttentionColBERT(
        model_name_or_path=args.model_name,
        document_length=args.document_length,
        num_proxy_tokens=args.num_proxy_tokens,
        num_select_tokens=args.num_select_tokens,
        use_cluster_pooling=use_cluster_pooling,
        proxy_tau=args.proxy_tau,
    )

    # Configure training mode (freeze/unfreeze parameters)
    model = setup_training_mode(
        model,
        mode=args.training_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Initialize evaluator
    dev_evaluator = evaluation.NanoBEIREvaluator()

    # Configure the training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        run_name=run_name,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
    )

    # Use the ProxyAttentionDistillation loss function for training
    train_loss = losses.ProxyAttentionDistillation(model=model)

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    # Run initial evaluation if requested
    if not args.skip_initial_eval:
        print("Running initial evaluation...")
        eval_results = trainer.evaluate()
        print(f"Initial evaluation results: {eval_results}")

    # Start the training process
    print("Training...")
    trainer.train()

    # Save final model
    final_path = f"{output_dir}/final"
    print(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)

    print("Training complete!")


if __name__ == "__main__":
    main()
