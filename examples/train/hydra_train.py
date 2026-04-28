"""
Hydra-configured training script for ColBERT/XTR models.
Supports contrastive, cached_contrastive, and KD training.

Usage examples:
  # Contrastive XTR (default)
  python examples/train/hydra_train.py

  # Contrastive ColBERT
  python examples/train/hydra_train.py loss=contrastive_colbert run_name=modernbert_colbert_contrastive

  # Contrastive XTR with specific k
  python examples/train/hydra_train.py loss=contrastive_xtr 'loss.k_train=[256]' run_name=modernbert_xtr_contrastive_k256

  # Contrastive XTR multi-k
  python examples/train/hydra_train.py loss=contrastive_xtr 'loss.k_train=[128,256,512]' run_name=modernbert_xtr_contrastive_multik128-256-512

  # Distillation (KD) from a contrastive checkpoint
  python examples/train/hydra_train.py --config-name distillation loss=kd_xtr 'loss.k_train=[128]' model_name=output/modernbert_xtr_contrastive_k128/final run_name=modernbert_xtr_kd_k128

  # Override run name
  python examples/train/hydra_train.py run_name=my-experiment
"""

from __future__ import annotations

import os

import hydra
import torch
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import TrainerCallback

from pylate import evaluation, losses, models, scores, utils


def load_contrastive_dataset(dataset_cfg: DictConfig):
    kwargs = {"path": dataset_cfg.hf_path, "split": dataset_cfg.hf_split}
    if dataset_cfg.hf_name is not None:
        kwargs["name"] = dataset_cfg.hf_name

    dataset = load_dataset(**kwargs)

    if dataset_cfg.tevatron:
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


def load_kd_dataset(dataset_cfg: DictConfig):
    train = load_dataset(path=dataset_cfg.hf_path, name="train")
    queries = load_dataset(path=dataset_cfg.hf_path, name="queries")
    documents = load_dataset(path=dataset_cfg.hf_path, name="documents")

    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents, n_ways=dataset_cfg.n_ways).transform,
    )

    return train


def build_score_metric(loss_cfg: DictConfig):
    score_fn = loss_cfg.score_fn
    is_kd = loss_cfg.type == "kd"

    if score_fn == "colbert":
        return scores.colbert_kd_scores if is_kd else scores.colbert_scores

    # XTR
    k_train = list(loss_cfg.k_train)
    k_weights = list(loss_cfg.k_weights) if loss_cfg.k_weights is not None else None

    def _k_arg():
        if len(k_train) == 1:
            return k_train[0]
        elif k_weights is not None:
            return list(zip(k_train, k_weights))
        else:
            return k_train

    if is_kd:
        return scores.XTRKDScores(k=_k_arg())
    return scores.XTRScores(k=_k_arg())


def make_run_name(cfg: DictConfig) -> str:
    choices = HydraConfig.get().runtime.choices
    dataset_name = choices.get("dataset", "dataset")
    loss_name = choices.get("loss", "loss")
    model_short = cfg.model_name.split("/")[-1]
    lr_str = f"{cfg.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")

    parts = [model_short, loss_name, dataset_name]

    loss_cfg = cfg.loss
    if loss_cfg.type == "kd":
        if loss_cfg.score_fn == "xtr":
            k_label = "_".join(str(k) for k in loss_cfg.k_train)
            parts.append(f"k{k_label}")
        parts.append(f"temp{loss_cfg.temperature}")
    else:
        if loss_cfg.score_fn == "xtr" and loss_cfg.k_train:
            k_label = "_".join(str(k) for k in loss_cfg.k_train)
            parts.append(f"k{k_label}")
        temp_prefix = "learnabletemp" if loss_cfg.learnable_temperature else "temp"
        parts.append(f"{temp_prefix}{loss_cfg.temperature}")

    parts.append(f"bs{cfg.batch_size}")
    parts.append(f"lr{lr_str}")

    return "-".join(parts)


def _build_query_token_weight_head(model: models.ColBERT, cfg: DictConfig):
    if not cfg.enabled:
        return None

    activation_map = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "identity": torch.nn.Identity,
    }
    if cfg.hidden_activation not in activation_map:
        raise ValueError(
            f"Unsupported hidden_activation '{cfg.hidden_activation}'. "
            f"Use one of {sorted(activation_map.keys())}."
        )

    output_dim = model[-1].out_features
    hidden_dims = [int(dim) for dim in cfg.hidden_dims]
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("All query_token_weight_head.hidden_dims must be > 0.")

    layer_dims = [output_dim, *hidden_dims, 1]
    layers = []
    for i in range(len(layer_dims) - 1):
        is_last = i == len(layer_dims) - 2
        activation = (
            torch.nn.Identity()
            if is_last
            else activation_map[cfg.hidden_activation]()
        )
        layers.append(
            models.Dense(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                bias=cfg.bias,
                activation_function=activation,
                use_residual=bool(cfg.use_residual) and not is_last,
            )
        )

    return models.QueryTokenWeightHead(
        layers=layers,
        normalization_mode=cfg.normalization_mode,
        positive_activation=cfg.positive_activation,
    )


@hydra.main(config_path="../../conf/train", config_name="contrastive", version_base=None)
def main(cfg: DictConfig):
    loss_cfg = cfg.loss
    dataset_cfg = cfg.dataset
    is_kd = loss_cfg.type == "kd"

    if cfg.max_steps is not None and cfg.num_epochs is not None:
        raise ValueError("Set at most one of max_steps and num_epochs.")
    if cfg.max_steps is None and cfg.num_epochs is None:
        raise ValueError("Set at least one of max_steps and num_epochs.")

    if is_kd and dataset_cfg.type != "kd":
        raise ValueError(f"KD loss requires a KD dataset (e.g. dataset=kd_msmarco), got '{dataset_cfg.type}'.")
    if not is_kd and dataset_cfg.type == "kd":
        raise ValueError(f"Contrastive loss requires a contrastive dataset, got kd dataset.")

    run_name = cfg.run_name or make_run_name(cfg)
    output_dir = cfg.output_dir or f"output/{run_name}"

    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    os.environ["WANDB_RUN_GROUP"] = cfg.wandb.group

    model = models.ColBERT(
        model_name_or_path=cfg.model_name,
        query_length=cfg.query_length,
        document_length=cfg.document_length,
        query_prefix=cfg.query_prefix,
        document_prefix=cfg.document_prefix,
    )
    query_weight_head = _build_query_token_weight_head(
        model=model, cfg=cfg.query_token_weight_head
    )
    if query_weight_head is not None:
        model.append(query_weight_head)

    train_dataset = load_kd_dataset(dataset_cfg) if is_kd else load_contrastive_dataset(dataset_cfg)

    score_metric = build_score_metric(loss_cfg)

    if loss_cfg.learnable_temperature:
        # Register on model so trainer optimizers that only inspect model params still update it.
        if hasattr(model, "learnable_temperature"):
            temperature = model.learnable_temperature
        else:
            temperature = torch.nn.Parameter(torch.tensor(float(loss_cfg.temperature)))
            model.register_parameter("learnable_temperature", temperature)
    else:
        temperature = float(loss_cfg.temperature)

    if loss_cfg.type == "contrastive":
        train_loss = losses.Contrastive(
            model=model,
            score_metric=score_metric,
            temperature=temperature,
            gather_across_devices=True,
        )
    elif loss_cfg.type == "cached_contrastive":
        train_loss = losses.CachedContrastive(
            model=model,
            score_metric=score_metric,
            mini_batch_size=loss_cfg.mini_batch_size,
            temperature=temperature,
            gather_across_devices=True,
        )
    elif loss_cfg.type == "kd":
        train_loss = losses.Distillation(
            model=model,
            score_metric=score_metric,
            temperature=temperature,
            normalize_scores=loss_cfg.minmax_normalize_scores,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_cfg.type}")
    evaluators = []
    if cfg.eval_steps > 0:
        dev_evaluator = evaluation.NanoBEIREvaluator(write_csv=False)
        evaluators.append(dev_evaluator)

    dtype_kwargs = {
        "fp16": cfg.dtype == "fp16",
        "bf16": cfg.dtype == "bf16",
    }

    training_args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_on_start=cfg.eval_on_start,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        **dtype_kwargs,
        torch_compile=cfg.compile,
        run_name=run_name,
        dataloader_num_workers=8,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )
    if cfg.max_steps is not None:
        training_args_kwargs["max_steps"] = cfg.max_steps
    else:
        training_args_kwargs["num_train_epochs"] = cfg.num_epochs

    training_args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    class TemperatureCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            t = train_loss.temperature
            temp_val = t.item() if isinstance(t, torch.Tensor) else float(t)
            if logs is not None:
                logs["temperature"] = temp_val
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"train/temperature": temp_val}, step=state.global_step)
            except ImportError:
                pass

    callbacks = [TemperatureCallback()]

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluators,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
        callbacks=callbacks or None,
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
