"""Config-driven (Q)AT runner for ColBERT.

Trains a ColBERT model from a small YAML config, optionally appending a permanent
``StraightThroughEstimator`` for quantization-aware training. Disabling the
``quantization`` block yields a full-precision, otherwise-identical "iso"
baseline for apples-to-apples comparison.

Usage
-----
    uv run python examples/train/qat_run.py --config examples/train/configs/qat_int8_binary.yaml
    uv run python examples/train/qat_run.py --config examples/train/configs/baseline_fp32.yaml

Config schema
-------------
    run_name: str
    base_model: str                 # HF id or local path
    output_dir: str
    dataset: str                    # HF dataset id (expects query/positive/negative)
    dataset_config: str             # e.g. "triplet"
    max_train_samples: int | null   # cap for a fast run
    eval_fraction: float
    quantization:                   # omit or enabled: false for the iso baseline
      enabled: bool
      pre_normalize: bool
      query: {type: ScalarQuantizer, n_bits: 8}     # any entry in pylate.models.QUANTIZERS
      document: {type: BinaryQuantizer}
    training:
      num_train_epochs / per_device_train_batch_size / learning_rate / fp16 / bf16 / ...
"""

from __future__ import annotations

import argparse
import os

import yaml
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from pylate import evaluation, losses, models, utils
from pylate.models import QUANTIZERS, StraightThroughEstimator


def build_quantizer(spec: dict | None):
    """Instantiate a quantizer from a ``{type: ..., **kwargs}`` spec."""
    if spec is None:
        return None
    spec = dict(spec)
    quantizer_type = spec.pop("type")
    if quantizer_type not in QUANTIZERS:
        raise ValueError(
            f"Unknown quantizer {quantizer_type!r}. Available: {sorted(QUANTIZERS)}."
        )
    return QUANTIZERS[quantizer_type](**spec)


def build_ste(quant_cfg: dict | None) -> StraightThroughEstimator | None:
    """Build the StraightThroughEstimator module, or None for a baseline run."""
    if not quant_cfg or not quant_cfg.get("enabled", False):
        return None
    query = build_quantizer(quant_cfg.get("query"))
    document = build_quantizer(quant_cfg.get("document"))
    pre_normalize = quant_cfg.get("pre_normalize", True)
    # If only one of query/document is given, the module reuses it for both.
    if query is not None or document is not None:
        return StraightThroughEstimator(
            query_transform=query,
            document_transform=document,
            pre_normalize=pre_normalize,
        )
    # No per-route spec but enabled: fall back to a single symmetric transform.
    return StraightThroughEstimator(
        transform=build_quantizer(quant_cfg.get("transform")),
        pre_normalize=pre_normalize,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        cfg = yaml.safe_load(f)

    model = models.ColBERT(model_name_or_path=cfg["base_model"])

    ste = build_ste(cfg.get("quantization"))
    if ste is not None:
        # Permanent module: saved with the model and active at inference. It is
        # the single source of truth for quantization, so do not also pass
        # `precision=` to encode().
        model.append(ste)
        print(f"Quantization-aware training enabled: {ste.get_config_dict()}")
    else:
        print("Full-precision (iso) baseline: no quantization module appended.")

    dataset = load_dataset(
        cfg["dataset"], cfg.get("dataset_config", "triplet"), split="train"
    )
    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples:
        dataset = dataset.select(range(min(max_train_samples, len(dataset))))
    # An absolute `max_eval_samples` keeps evaluation fast on large corpora;
    # otherwise fall back to a fractional split.
    test_size = cfg.get("max_eval_samples") or cfg.get("eval_fraction", 0.1)
    splits = dataset.train_test_split(test_size=test_size, seed=cfg.get("seed", 42))
    train_dataset, eval_dataset = splits["train"], splits["test"]

    train_loss = losses.Contrastive(model=model)

    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    # Weights & Biases logging (uses ~/.netrc credentials). Disabled => "none".
    wandb_cfg = cfg.get("wandb") or {}
    report_to = "none"
    if wandb_cfg.get("enabled", False):
        report_to = "wandb"
        if wandb_cfg.get("project"):
            os.environ.setdefault("WANDB_PROJECT", wandb_cfg["project"])

    t = cfg.get("training", {})
    batch_size = t.get("per_device_train_batch_size", 16)
    save_steps = t.get("save_steps")
    args = SentenceTransformerTrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=t.get("num_train_epochs", 1),
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", batch_size),
        learning_rate=float(t.get("learning_rate", 3e-6)),
        warmup_ratio=t.get("warmup_ratio", 0.0),
        fp16=t.get("fp16", False),
        bf16=t.get("bf16", False),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 42),
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 10),
        logging_steps=t.get("logging_steps", 5),
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_total_limit=t.get("save_total_limit", 1),
        report_to=report_to,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )
    trainer.train()

    final_dir = f"{cfg['output_dir']}/final"
    model.save_pretrained(final_dir)
    print(f"Saved model to {final_dir}")


if __name__ == "__main__":
    main()
