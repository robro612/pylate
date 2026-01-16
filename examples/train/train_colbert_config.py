"""Unified training script for ColBERT models using JSON config.

Supports both standard ColBERT and ProxyAttentionColBERT models with
Knowledge Distillation training on MS MARCO.

Multi-GPU Training:
    torchrun --nproc_per_node=4 train_colbert_config.py --config config.json

Example Config (config.json):
{
    "model_type": "colbert",
    "model_name": "Alibaba-NLP/gte-modernbert-base",
    "batch_size": 24,
    "lr": 3e-5,
    "epochs": 3,
    "n_ways": 16
}

For ProxyAttentionColBERT:
{
    "model_type": "proxy_attention",
    "model_name": "Alibaba-NLP/gte-modernbert-base",
    "num_proxy_tokens": 32,
    "num_select_tokens": 32,
    ...
}
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


@dataclass
class TrainingConfig:
    """Configuration for ColBERT training."""
    
    # Model configuration
    model_type: str = "colbert"  # "colbert" or "proxy_attention"
    model_name: str = "Alibaba-NLP/gte-modernbert-base"
    document_length: int = 300
    
    # ProxyAttentionColBERT specific (ignored for standard ColBERT)
    num_proxy_tokens: int = 32
    num_select_tokens: int = 32
    use_cluster_pooling: bool = True
    proxy_tau: float = 1.0
    
    # Training parameters
    batch_size: int = 24
    n_ways: int = 16
    lr: float = 3e-5
    epochs: int = 3
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    
    # Training mode (for ProxyAttentionColBERT)
    training_mode: str = "full"  # "full", "proxy_only", "freeze_word_embeddings", "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Evaluation and logging
    eval_steps: int = 500
    save_steps: int = 5000
    logging_steps: int = 20
    skip_initial_eval: bool = False
    
    # Precision
    fp16: bool = False
    bf16: bool = True
    
    # Output
    output_dir: Optional[str] = None
    run_name: Optional[str] = None
    
    # Dataset
    dataset_path: str = "lightonai/ms-marco-en-bge-gemma"
    
    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, json_path: str) -> None:
        """Save config to JSON file."""
        with open(json_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    def generate_run_name(self) -> str:
        """Generate a run name based on config with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.model_type == "proxy_attention":
            return (
                f"ProxyAttention-ColBERT-{self.num_select_tokens}tok-"
                f"{self.lr}-lr-{self.epochs}ep-{self.training_mode}-"
                f"bs{self.batch_size}-nway{self.n_ways}-{timestamp}"
            )
        else:
            return (
                f"ColBERT-{self.lr}-lr-{self.epochs}ep-"
                f"bs{self.batch_size}-nway{self.n_ways}-{timestamp}"
            )


def setup_training_mode(model, config: TrainingConfig):
    """Configure trainable parameters based on training mode."""
    mode = config.training_mode
    
    if mode == "full":
        print("Training mode: full - all parameters trainable")
    
    elif mode == "proxy_only":
        print("Training mode: proxy_only - freezing all except proxy embeddings")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "_proxy_embeddings"):
            model._proxy_embeddings.weight.requires_grad = True
    
    elif mode == "freeze_word_embeddings":
        print("Training mode: freeze_word_embeddings")
        transformer = model[0]
        word_embeddings = transformer.auto_model.get_input_embeddings()
        for param in word_embeddings.parameters():
            param.requires_grad = False
    
    elif mode == "lora":
        print(f"Training mode: lora (r={config.lora_r}, alpha={config.lora_alpha})")
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("LoRA requires peft: pip install peft")
        
        for param in model.parameters():
            param.requires_grad = False
        
        transformer = model[0]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"],
        )
        transformer.auto_model = get_peft_model(transformer.auto_model, lora_config)
        
        if hasattr(model, "_proxy_embeddings"):
            model._proxy_embeddings.weight.requires_grad = True
    else:
        raise ValueError(f"Unknown training mode: {mode}")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def create_model(config: TrainingConfig):
    """Create the appropriate model based on config."""
    if config.model_type == "proxy_attention":
        print(f"Creating ProxyAttentionColBERT model...")
        model = models.ProxyAttentionColBERT(
            model_name_or_path=config.model_name,
            document_length=config.document_length,
            num_proxy_tokens=config.num_proxy_tokens,
            num_select_tokens=config.num_select_tokens,
            use_cluster_pooling=config.use_cluster_pooling,
            proxy_tau=config.proxy_tau,
        )
    elif config.model_type == "colbert":
        print(f"Creating ColBERT model...")
        model = models.ColBERT(
            model_name_or_path=config.model_name,
            document_length=config.document_length,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}. Use 'colbert' or 'proxy_attention'")

    return model


def create_loss(model, config: TrainingConfig):
    """Create the appropriate loss function based on model type."""
    if config.model_type == "proxy_attention":
        return losses.ProxyAttentionDistillation(model=model)
    else:
        return losses.Distillation(model=model)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train ColBERT models using JSON config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python train_colbert_config.py --config config.json
  python train_colbert_config.py --config config.json --output_dir ./my_output
  python train_colbert_config.py --generate_config default_config.json

Multi-GPU:
  torchrun --nproc_per_node=4 train_colbert_config.py --config config.json
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--generate_config",
        type=str,
        help="Generate a default config file and exit",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Override run name from config",
    )
    parser.add_argument(
        "--skip_initial_eval",
        action="store_true",
        help="Skip initial evaluation before training",
    )

    args = parser.parse_args()

    # Generate default config if requested
    if args.generate_config:
        config = TrainingConfig()
        config.to_json(args.generate_config)
        print(f"Generated default config: {args.generate_config}")
        return

    # Load config
    if not args.config:
        parser.error("--config is required (or use --generate_config to create one)")

    print(f"Loading config from: {args.config}")
    config = TrainingConfig.from_json(args.config)

    # Override with command line args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.run_name:
        config.run_name = args.run_name
    if args.skip_initial_eval:
        config.skip_initial_eval = True

    # Generate run name and output dir if not set
    run_name = config.run_name or config.generate_run_name()
    output_dir = config.output_dir or f"output/{run_name}"

    # Print configuration
    print("=" * 60)
    print(f"Training Configuration")
    print("=" * 60)
    print(f"Model type: {config.model_type}")
    print(f"Model name: {config.model_name}")
    print(f"Training mode: {config.training_mode}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"N-ways: {config.n_ways}")
    if config.model_type == "proxy_attention":
        print(f"Num proxy tokens: {config.num_proxy_tokens}")
        print(f"Num select tokens: {config.num_select_tokens}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Save config to output directory
    os.makedirs(output_dir, exist_ok=True)
    config.to_json(os.path.join(output_dir, "config.json"))

    # Load datasets
    print("Loading datasets...")
    train = load_dataset(path=config.dataset_path, name="train")
    queries = load_dataset(path=config.dataset_path, name="queries")
    documents = load_dataset(path=config.dataset_path, name="documents")

    # Set up dataset transformation
    print("Setting up dataset transformation...")
    train.set_transform(
        utils.KDProcessing(
            queries=queries,
            documents=documents,
            n_ways=config.n_ways,
        ).transform,
    )

    # Create model
    print("Initializing model...")
    model = create_model(config)

    # Configure training mode
    model = setup_training_mode(model, config)

    # Initialize evaluator
    dev_evaluator = evaluation.NanoBEIREvaluator()

    # Configure training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        run_name=run_name,
        learning_rate=config.lr,
        warmup_ratio=config.warmup_ratio,
    )

    # Create loss function
    train_loss = create_loss(model, config)

    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    # Run initial evaluation if requested
    if not config.skip_initial_eval:
        print("Running initial evaluation...")
        eval_results = trainer.evaluate()
        print(f"Initial evaluation results: {eval_results}")

    # Start training
    print("Training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(output_dir, "final")
    print(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)

    print("Training complete!")


if __name__ == "__main__":
    main()

