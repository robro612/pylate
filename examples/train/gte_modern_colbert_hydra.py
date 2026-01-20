"""Train ColBERT-family models with Hydra/OmegaConf on MS MARCO KD data.

Supports:
  - ColBERT
  - ConstBERT
  - MemoryTokenColBERT
  - ProxyAttentionColBERT

Example usage:
  python gte_modern_colbert_hydra.py
  python gte_modern_colbert_hydra.py model=constbert model.variant_args.constbert_seq_length=32
  python gte_modern_colbert_hydra.py model=memory_token model.variant_args.num_memory_tokens=64 model.variant_args.attend_to_memory_tokens=true
  python gte_modern_colbert_hydra.py model=proxy_attention model.variant_args.num_proxy_tokens=64 model.variant_args.num_select_tokens=64
"""

from __future__ import annotations

import os
from typing import Optional

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def build_run_name(cfg: DictConfig) -> str:
    variant_args = cfg.model.get("variant_args") or {}
    base = (
        f"{cfg.model.type}-lr-{cfg.lr}-bs{cfg.batch_size}nway{cfg.n_ways}"
    )
    if cfg.model.type == "constbert":
        constbert_variant = variant_args["constbert_variant"]
        constbert_seq_length = variant_args["constbert_seq_length"]
        base = f"{base}-{constbert_variant}-C{constbert_seq_length}"
    if cfg.model.type == "memory_token":
        num_memory_tokens = variant_args["num_memory_tokens"]
        base = f"{base}-M{num_memory_tokens}"
    if cfg.model.type == "proxy_attention":
        num_proxy_tokens = variant_args["num_proxy_tokens"]
        num_select_tokens = variant_args["num_select_tokens"]
        base = f"{base}-P{num_proxy_tokens}-S{num_select_tokens}"
    return base


def create_model(cfg: DictConfig):
    variant_args = cfg.model.get("variant_args") or {}
    if cfg.model.type == "colbert":
        return models.ColBERT(
            model_name_or_path=cfg.model.name,
            document_length=cfg.model.document_length,
            query_length=cfg.model.query_length,
        )
    if cfg.model.type == "constbert":
        return models.ConstBERT(
            model_name_or_path=cfg.model.name,
            document_length=cfg.model.document_length,
            query_length=cfg.model.query_length,
            constbert_variant=variant_args["constbert_variant"],
            constbert_seq_length=variant_args["constbert_seq_length"],
        )
    if cfg.model.type == "memory_token":
        return models.MemoryTokenColBERT(
            model_name_or_path=cfg.model.name,
            document_length=cfg.model.document_length,
            query_length=cfg.model.query_length,
            num_memory_tokens=variant_args["num_memory_tokens"],
            attend_to_memory_tokens=variant_args["attend_to_memory_tokens"],
        )
    if cfg.model.type == "proxy_attention":
        return models.ProxyAttentionColBERT(
            model_name_or_path=cfg.model.name,
            document_length=cfg.model.document_length,
            num_proxy_tokens=variant_args["num_proxy_tokens"],
            num_select_tokens=variant_args["num_select_tokens"],
            use_cluster_pooling=variant_args["use_cluster_pooling"],
            proxy_tau=variant_args["proxy_tau"],
        )
    raise ValueError(
        "Unknown model.type. Use one of: colbert, constbert, memory_token, proxy_attention."
    )


def create_loss(cfg: DictConfig, model):
    if cfg.model.type == "proxy_attention":
        return losses.ProxyAttentionDistillation(model=model)
    return losses.Distillation(model=model, normalize_scores=cfg.normalize_scores)


@hydra.main(version_base=None, config_path="../../conf", config_name="gte_modern_colbert")
def main(cfg: DictConfig) -> None:
    original_cwd = hydra.utils.get_original_cwd()

    run_name = cfg.run_name or build_run_name(cfg)
    output_dir = os.path.join("output", run_name)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(original_cwd, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            wandb.save(config_path)
    except Exception:
        pass

    train = load_dataset(path=cfg.dataset_path, name="train")
    queries = load_dataset(path=cfg.dataset_path, name="queries")
    documents = load_dataset(path=cfg.dataset_path, name="documents")

    train.set_transform(
        utils.KDProcessing(
            queries=queries,
            documents=documents,
            n_ways=cfg.n_ways,
        ).transform,
    )

    model = create_model(cfg)
    if cfg.compile:
        model.compile()

    dev_evaluator = evaluation.NanoBEIREvaluator(batch_size=cfg.batch_size)

    # resolve dtype to fp16 or bf16 (or fp32 by default)
    if cfg.dtype == "fp16":
        fp16, bf16 = True, False
    elif cfg.dtype == "bf16":
        fp16, bf16 = False, True
    else:
        fp16, bf16 = False, False

    print(f"dtype: {cfg.dtype}, fp16: {fp16}, bf16: {bf16}")

    args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.batch_size,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        fp16=fp16,
        bf16=bf16,
        run_name=run_name,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
    )
    if cfg.max_steps is not None:
        args_kwargs["max_steps"] = cfg.max_steps
    else:
        args_kwargs["num_train_epochs"] = cfg.num_train_epochs

    args = SentenceTransformerTrainingArguments(**args_kwargs)
    train_loss = create_loss(cfg, model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()
