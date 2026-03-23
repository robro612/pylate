"""
Train a ModernBERT-based ColBERT model with XTR cached contrastive loss
on the RLHN-680K msmarco_passage subset.
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

from pylate import evaluation, losses, models, scores, utils

QUERY_LENGTH = 32
DOCUMENT_LENGTH = 512

DATASET_CONFIGS = {
    "rlhn": {
        "hf_path": "rlhn/rlhn-680K",
        "hf_name": None,
        "hf_split": "train",
        "tevatron": True,
    },
    "bm25": {
        "hf_path": "sentence-transformers/msmarco-bm25",
        "hf_name": "triplet",
        "hf_split": "train",
        "tevatron": False,
    },
    "bclavie": {
        "hf_path": "bclavie/msmarco-10m-triplets",
        "hf_name": None,
        "hf_split": "train",
        "tevatron": False,
    },
}


def load_train_dataset(cfg: dict):
    """Load and prepare a dataset based on its config."""
    kwargs = {"path": cfg["hf_path"], "split": cfg["hf_split"]}
    if cfg["hf_name"] is not None:
        kwargs["name"] = cfg["hf_name"]

    dataset = load_dataset(**kwargs)

    if cfg["tevatron"]:
        dataset = dataset.filter(
            lambda x: x["subset"] == "msmarco_passage",
            desc="Filtering to msmarco_passage",
        )
        processing = utils.RLHNProcessing(n_ways=1)
        dataset = dataset.map(
            processing.map,
            remove_columns=dataset.column_names,
            desc="Converting to triplets",
        )

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="XTR cached contrastive training with ModernBERT on RLHN msmarco."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--learnable_temperature",
        action="store_true",
        help="Make temperature a learnable parameter. If not set, temperature is fixed.",
    )
    parser.add_argument(
        "--score_fn",
        type=str,
        default="xtr",
        choices=["xtr", "colbert"],
        help="Scoring function: 'xtr' for XTR scores, 'colbert' for ColBERT MaxSim.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rlhn",
        choices=list(DATASET_CONFIGS.keys()),
        help="Training dataset: 'rlhn' (RLHN-680K msmarco), 'bm25' (msmarco-bm25), 'bclavie' (msmarco-10m-triplets).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cached",
        choices=["cached", "standard"],
        help="Loss type: 'cached' for CachedContrastive (GradCache), 'standard' for Contrastive.",
    )
    parser.add_argument("--k_train", type=int, default=128, help="Top-k for XTR scoring during training.")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or f"xtr-contrastive-modernbert-{args.dataset}-ktrain{args.k_train}"
    output_dir = f"output/{run_name}"

    # Wandb config — Trainer handles init on main process only
    os.environ["WANDB_PROJECT"] = "xtr"
    os.environ["WANDB_RUN_GROUP"] = "contrastive"

    # Model
    model = models.ColBERT(
        model_name_or_path=args.model_name,
        query_length=QUERY_LENGTH,
        document_length=DOCUMENT_LENGTH,
        query_prefix="",
        document_prefix="",
    )

    # Dataset
    train_dataset = load_train_dataset(DATASET_CONFIGS[args.dataset])

    # Loss
    if args.learnable_temperature:
        temperature = torch.nn.Parameter(torch.tensor(args.temperature))
    else:
        temperature = args.temperature
    score_metric = scores.XTRScores(k=args.k_train) if args.score_fn == "xtr" else scores.colbert_scores
    if args.loss == "cached":
        train_loss = losses.CachedContrastive(
            model=model,
            score_metric=score_metric,
            mini_batch_size=args.mini_batch_size,
            temperature=temperature,
            gather_across_devices=True,
        )
    else:
        train_loss = losses.Contrastive(
            model=model,
            score_metric=score_metric,
            temperature=temperature,
            gather_across_devices=True,
        )

    # Evaluator
    dev_evaluator = evaluation.NanoBEIREvaluator()

    # Training args
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=10,
        fp16=True,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        run_name=run_name,
        dataloader_num_workers=8,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    # Log temperature callback — log directly to wandb since the wandb
    # callback's on_log fires before ours and misses keys we add to logs.
    class TemperatureCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if isinstance(train_loss.temperature, torch.nn.Parameter):
                temp_val = train_loss.temperature.item()
                if logs is not None:
                    logs["temperature"] = temp_val
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"train/temperature": temp_val}, step=state.global_step)
                except ImportError:
                    pass

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=[TemperatureCallback()],
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
