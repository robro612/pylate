"""Config-driven ColBERT training runner.

Trains a ColBERT model from a small YAML config. Compression is optional and
layered on top of an ordinary contrastive run:

* no ``compression``/``quantization`` block -> a plain full-precision baseline;
* ``compression:`` -> compression-aware training (``CompressionAwareLoss``: a
  lam-weighted mix of the loss on raw and on quantized/pooled embeddings);
* ``quantization:`` -> the legacy in-place ``StraightThroughEstimator`` module.

The dataset can be a flat ``query/positive/negative`` triplet set or an
RLHN-style hard-negative set (auto-detected and flattened via
:func:`pylate.data.rlhn_to_contrastive`).

Usage
-----
    uv run python examples/train/train.py --config examples/train/configs/baseline_fp32.yaml
    uv run python examples/train/train.py --config examples/train/configs/cat_rlhn.yaml

Config schema
-------------
    run_name: str
    base_model: str                 # HF id or local path
    output_dir: str
    base_model: str                 # HF id or local path
    document_length: int | null     # max doc tokens (ModernBERT defaults huge; set ~300)
    query_length: int | null        # max query tokens
    dataset: str                    # HF dataset id
    dataset_config: str | null      # HF config name, e.g. "triplet" (omit for RLHN)
    num_negatives: int              # hard negatives/query when the dataset is RLHN-style
    max_train_samples: int | null   # cap for a fast run
    eval_fraction: float
    quantization:                   # legacy in-place STE module (single loss)
      enabled: bool                 #   omit or enabled: false for the iso baseline
      pre_normalize: bool
      query: {type: ScalarQuantizer, n_bits: 8}     # any entry in pylate.models.QUANTIZERS
      document: {type: BinaryQuantizer}
    compression:                    # CompressionAwareLoss: lam-mixed raw vs compressed
      enabled: bool                 #   (mutually exclusive with `quantization`)
      lambda: 0.5                   # weight on the full-precision loss (1.0 == baseline)
      query:                        # per-side pipeline: optional pooler then quantizer
        quantizer: {type: CastQuantizer, dtype: int8}     # int8 | float16 | bfloat16 | float32
      document:
        quantizer:                  # SHBQ = Hadamard rotation around an inner quantizer
          type: SHBQQuantizer
          inner: {type: BinaryQuantizer}                  # SHBQ binary documents
        pooler: {name: ward, similarity_threshold: 0.7}   # kmeans | ward
    training:
      num_train_epochs / per_device_train_batch_size / learning_rate / fp16 / bf16 / ...
"""

from __future__ import annotations

import argparse
import json
import os

import yaml
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from pylate import data, evaluation, losses, models, utils
from pylate.models import StraightThroughEstimator, build_quantizer


def build_ste(quant_cfg: dict | None) -> StraightThroughEstimator | None:
    """Build the legacy StraightThroughEstimator module, or None for a baseline run."""
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

    # Model kwargs (e.g. document_length / query_length) are forwarded from the
    # config; ModernBERT-based encoders default to a very long context, so a
    # sensible document_length keeps memory in check.
    model_kwargs = {
        key: cfg[key]
        for key in ("document_length", "query_length")
        if cfg.get(key) is not None
    }
    model = models.ColBERT(model_name_or_path=cfg["base_model"], **model_kwargs)

    # Two mutually exclusive QAT paths:
    #   * `compression:` -> CompressionAwareLoss mixes the base loss over raw and
    #     compressed (quantized and/or pooled) embeddings, lam-weighted. The model
    #     stays vanilla; appending an STE module here would double-compress.
    #   * `quantization:` -> the legacy in-place StraightThroughEstimator module
    #     (single loss on quantized embeddings), saved and active at inference.
    compression_cfg = cfg.get("compression")
    use_compression = bool(compression_cfg and compression_cfg.get("enabled", False))

    # `dataset_config` is the HF config name (e.g. "triplet"); omit it for datasets
    # like RLHN that have no named config.
    dataset_config = cfg.get("dataset_config")
    if dataset_config is not None:
        dataset = load_dataset(cfg["dataset"], dataset_config, split="train")
    else:
        dataset = load_dataset(cfg["dataset"], split="train")

    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples:
        dataset = dataset.select(range(min(max_train_samples, len(dataset))))

    # Auto-detect RLHN-style hard-negative schema and flatten it to the
    # query/positive/negative_1..k columns Contrastive expects.
    if "negative_passages" in dataset.column_names:
        num_negatives = int(cfg.get("num_negatives", 7))
        dataset = data.rlhn_to_contrastive(
            dataset, num_negatives=num_negatives, seed=cfg.get("seed", 42)
        )
        print(f"Detected RLHN schema: using {num_negatives} hard negatives per query.")

    # An absolute `max_eval_samples` keeps evaluation fast on large corpora;
    # otherwise fall back to a fractional split.
    test_size = cfg.get("max_eval_samples") or cfg.get("eval_fraction", 0.1)
    splits = dataset.train_test_split(test_size=test_size, seed=cfg.get("seed", 42))
    train_dataset, eval_dataset = splits["train"], splits["test"]

    if use_compression:
        train_loss = losses.build_compression_aware_loss(
            base_loss=losses.Contrastive(model=model),
            spec=compression_cfg,
        )
        print(
            "Compression-aware training enabled (lam="
            f"{train_loss.lam}): "
            f"query={train_loss.query_compressor.get_config_dict()}, "
            f"document={train_loss.document_compressor.get_config_dict()}"
        )
    else:
        ste = build_ste(cfg.get("quantization"))
        if ste is not None:
            # Permanent module: saved with the model and active at inference. It is
            # the single source of truth for quantization, so do not also pass
            # `precision=` to encode().
            model.append(ste)
            print(f"Quantization-aware training enabled: {ste.get_config_dict()}")
        else:
            print("Full-precision (iso) baseline: no quantization module appended.")
        train_loss = losses.Contrastive(model=model)

    # Triplet datasets expose "negative"; flattened RLHN exposes "negative_1..k".
    negative_column = "negative" if "negative" in eval_dataset.column_names else "negative_1"
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset[negative_column],
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

    # Persist the compression spec next to the (vanilla) model so index-time can
    # rebuild the *same* query/document Compressors via
    # pylate.models.build_compressor -- one config atom, two phases.
    if use_compression:
        compression_spec = {
            "lambda": train_loss.lam,
            "query": train_loss.query_compressor.get_config_dict(),
            "document": train_loss.document_compressor.get_config_dict(),
        }
        with open(os.path.join(final_dir, "compression.json"), "w") as f:
            json.dump(compression_spec, f, indent=2)
        print(f"Saved compression spec to {final_dir}/compression.json")


if __name__ == "__main__":
    main()
