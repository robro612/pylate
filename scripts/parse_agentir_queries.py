"""Load and parse AgentIR-style queries into final query strings.

Expected raw format (single string field):

    Reasoning: ...
    ...
    Query: "keyword phrase" "another phrase"

Expansion modes:
  - plain: use only parsed Query text
  - query_plus_reasoning: prepend parsed Reasoning to Query
  - rm3_positive: RM3-style lexical expansion from positive passages
  - hyde: generate hypothetical doc via DSPy (replace or append)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm


QUERY_MARKER_RE = re.compile(r"(?im)^\s*Query\s*:\s*")
REASONING_MARKER_RE = re.compile(r"(?im)^\s*Reasoning\s*:\s*")
FILETYPE_OP_RE = re.compile(r"(?i)(?:^|\s)filetype:[^\s]+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def infer_id_column(columns: list[str], requested: str | None) -> str | None:
    if requested:
        if requested not in columns:
            raise ValueError(
                f"--id-column '{requested}' not found in dataset columns: {columns}"
            )
        return requested
    for candidate in ("query_id", "qid", "id"):
        if candidate in columns:
            return candidate
    return None


def strip_marker_prefix(text: str, marker_re: re.Pattern[str]) -> str:
    return marker_re.sub("", text, count=1).strip()


def normalize_query_text(query: str) -> str:
    cleaned = query.replace('"', "").replace("“", "").replace("”", "")
    cleaned = FILETYPE_OP_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def lexical_tokenize(
    text: str,
    min_token_len: int,
    remove_stopwords: bool,
) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= min_token_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]
    return tokens


def parse_agentir_text(raw_text: str) -> dict[str, str | None]:
    text = raw_text.replace("\r\n", "\n").strip()
    if not text:
        return {
            "reasoning": None,
            "query": "",
            "parse_status": "empty",
        }

    query_matches = list(QUERY_MARKER_RE.finditer(text))
    if not query_matches:
        # Fallback: no explicit Query marker, treat whole text as query.
        return {
            "reasoning": None,
            "query": normalize_query_text(text),
            "parse_status": "no_query_marker",
        }

    last_query = query_matches[-1]
    before = text[: last_query.start()].strip()
    query = text[last_query.end() :].strip()

    reasoning = None
    if before:
        reasoning = before
        if REASONING_MARKER_RE.match(reasoning):
            reasoning = strip_marker_prefix(reasoning, REASONING_MARKER_RE)

    return {
        "reasoning": reasoning if reasoning else None,
        "query": normalize_query_text(query),
        "parse_status": "ok",
    }


def build_final_query(
    parsed: dict[str, str | None],
    row: dict,
    raw_text: str,
    expansion_mode: str,
    expansion_cfg: DictConfig,
    hyde_generator=None,
) -> str:
    query = str(parsed["query"] or "").strip()
    if expansion_mode == "plain":
        return query
    if expansion_mode == "query_plus_reasoning":
        # Keep original AgentIR format, including Reasoning:/Query: markers.
        return raw_text.strip()
    if expansion_mode == "rm3_positive":
        return rm3_expand_query(
            query=query,
            row=row,
            expansion_cfg=expansion_cfg,
        )
    if expansion_mode == "hyde":
        if hyde_generator is None:
            raise RuntimeError("HyDE mode requested but DSPy generator is not configured.")
        hyde_doc = hyde_generate_document(query=query, hyde_generator=hyde_generator)
        if not hyde_doc:
            return query
        behavior = str(expansion_cfg.hyde_behavior)
        separator = str(expansion_cfg.hyde_separator)
        if behavior == "replace":
            return hyde_doc
        if behavior == "append":
            return f"{query}{separator}{hyde_doc}".strip()
        raise ValueError(
            f"Unknown expansion.hyde_behavior='{behavior}'. Expected: append or replace."
        )
    raise ValueError(
        f"Unknown expansion.mode='{expansion_mode}'. "
        "Expected one of: plain, query_plus_reasoning, rm3_positive, hyde."
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def resolve_output_path(cfg: DictConfig) -> Path:
    if cfg.output_file is not None:
        return Path(str(cfg.output_file))

    dataset_slug = sanitize_name(str(cfg.dataset))
    split_slug = sanitize_name(str(cfg.split))
    mode_slug = sanitize_name(str(cfg.expansion.mode))
    extra_suffix = ""
    if str(cfg.expansion.mode) == "hyde":
        behavior_slug = sanitize_name(str(cfg.expansion.hyde_behavior))
        model_slug = sanitize_name(str(cfg.lm.model))
        extra_suffix = f"_{behavior_slug}_{model_slug}"
    elif str(cfg.expansion.mode) == "rm3_positive":
        if "feedback_fields" in cfg.expansion and cfg.expansion.feedback_fields is not None:
            field_names = [str(x) for x in cfg.expansion.feedback_fields]
        else:
            field_names = [str(cfg.expansion.feedback_field)]
        if field_names == ["positive_passages"]:
            field_slug = "fpos"
        elif sorted(field_names) == ["negative_passages", "positive_passages"]:
            field_slug = "fposneg"
        else:
            field_slug = "f" + "_".join(sanitize_name(f) for f in field_names)
        extra_suffix = (
            f"_t{int(cfg.expansion.feedback_terms)}"
            f"_d{int(cfg.expansion.feedback_docs)}"
            f"_w{int(float(cfg.expansion.orig_query_weight) * 100)}"
            f"_{field_slug}"
        )
    filename = f"agentir_queries_{dataset_slug}_{split_slug}_{mode_slug}{extra_suffix}.jsonl"
    return Path(str(cfg.output_dir)) / filename


def build_dspy_hyde_generator(cfg: DictConfig):
    if not bool(cfg.lm.enabled):
        raise ValueError("HyDE requires lm.enabled=true in query_expansion config.")
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "HyDE mode requires dspy. Install with `pip install dspy`."
        ) from exc

    lm_kwargs = {
        "model": str(cfg.lm.model),
        "temperature": float(cfg.lm.temperature),
        "max_tokens": int(cfg.lm.max_tokens),
    }
    if cfg.lm.api_key_env:
        import os

        api_key = os.environ.get(str(cfg.lm.api_key_env))
        if api_key:
            lm_kwargs["api_key"] = api_key
    if cfg.lm.api_base:
        lm_kwargs["api_base"] = str(cfg.lm.api_base)

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)

    class HyDEGenerateDoc(dspy.Signature):
        """Write a concise hypothetical document likely to answer the query."""

        query = dspy.InputField()
        hypothetical_document = dspy.OutputField(
            desc="A concise passage containing facts likely relevant to the query."
        )

    return dspy.Predict(HyDEGenerateDoc)


def hyde_generate_document(query: str, hyde_generator) -> str:
    prediction = hyde_generator(query=query)
    # DSPy returns a Prediction object with output field attributes.
    return str(getattr(prediction, "hypothetical_document", "") or "").strip()


def rm3_expand_query(
    query: str,
    row: dict,
    expansion_cfg: DictConfig,
) -> str:
    """RM3-style lexical expansion using row-level feedback passages."""
    if "feedback_fields" in expansion_cfg and expansion_cfg.feedback_fields is not None:
        feedback_fields = [str(x) for x in expansion_cfg.feedback_fields]
    else:
        feedback_fields = [str(expansion_cfg.feedback_field)]
    feedback_text_field = str(expansion_cfg.feedback_text_field)
    fb_docs = int(expansion_cfg.feedback_docs)
    fb_terms = int(expansion_cfg.feedback_terms)
    orig_query_weight = float(expansion_cfg.orig_query_weight)
    min_token_len = int(expansion_cfg.min_token_len)
    remove_stopwords = bool(expansion_cfg.remove_stopwords)

    query_tokens = lexical_tokenize(
        query,
        min_token_len=min_token_len,
        remove_stopwords=remove_stopwords,
    )
    if not query_tokens:
        return query

    feedback_passages: list[dict] = []
    for field in feedback_fields:
        passages = row.get(field, [])
        if isinstance(passages, list):
            feedback_passages.extend(
                [p for p in passages if isinstance(p, dict)]
            )

    if not feedback_passages:
        return query

    doc_tokens: list[list[str]] = []
    for passage in feedback_passages[:fb_docs]:
        text = str(passage.get(feedback_text_field, "") or "")
        toks = lexical_tokenize(
            text,
            min_token_len=min_token_len,
            remove_stopwords=remove_stopwords,
        )
        if toks:
            doc_tokens.append(toks)

    if not doc_tokens:
        return query

    # p(q|d) under unigram LM with tiny additive smoothing, then softmax over docs
    log_pqd = []
    for toks in doc_tokens:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        length = len(toks)
        lp = 0.0
        for qt in query_tokens:
            p = tf.get(qt, 0) / length
            lp += math.log(p + 1e-12)
        log_pqd.append(lp)

    mx = max(log_pqd)
    exp_scores = [math.exp(v - mx) for v in log_pqd]
    z = sum(exp_scores)
    doc_weights = [v / z for v in exp_scores]

    # Relevance model term weights: sum_d p(t|d) * p(d|q)
    rm = {}
    for d_weight, toks in zip(doc_weights, doc_tokens):
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        length = len(toks)
        for term, count in tf.items():
            rm[term] = rm.get(term, 0.0) + d_weight * (count / length)

    if not rm:
        return query

    # Mix in original query model
    q_model = {}
    q_len = len(query_tokens)
    for t in query_tokens:
        q_model[t] = q_model.get(t, 0.0) + 1.0 / q_len

    mixed = {}
    for t, w in rm.items():
        mixed[t] = (1.0 - orig_query_weight) * w
    for t, w in q_model.items():
        mixed[t] = mixed.get(t, 0.0) + orig_query_weight * w

    ranked_terms = sorted(mixed.items(), key=lambda x: x[1], reverse=True)
    selected_terms = [t for t, _ in ranked_terms[:fb_terms]]

    # Keep original query prefix + append selected terms not already present
    query_lower_terms = set(query.lower().split())
    additions = [t for t in selected_terms if t not in query_lower_terms]
    if not additions:
        return query
    return (query + " " + " ".join(additions)).strip()


@hydra.main(
    config_path="../conf/query_expansion",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    output_path = resolve_output_path(cfg)
    hyde_generator = None
    if str(cfg.expansion.mode) == "hyde":
        hyde_generator = build_dspy_hyde_generator(cfg)

    ds = load_dataset(cfg.dataset, split=cfg.split)
    if cfg.max_rows is not None:
        ds = ds.select(range(min(int(cfg.max_rows), len(ds))))

    columns = list(ds.column_names)
    if cfg.text_column not in columns:
        raise ValueError(
            f"text_column '{cfg.text_column}' not found in dataset columns: {columns}"
        )
    id_column = infer_id_column(columns=columns, requested=cfg.id_column)

    # Quick visibility into feedback corpus available for RM3.
    for field in ("positive_passages", "negative_passages"):
        if field in columns:
            counts = [len(row[field]) if row[field] is not None else 0 for row in ds]
            arr = sorted(counts)
            n = len(arr)
            mean = sum(arr) / max(1, n)
            median = arr[n // 2] if n else 0
            mn = arr[0] if n else 0
            mx = arr[-1] if n else 0
            zeros = sum(1 for c in arr if c == 0)
            print(
                f"{field}: n={n:,} mean={mean:.2f} median={median} min={mn} max={mx} "
                f"zero_count={zeros:,}"
            )

    parsed_rows: list[dict] = []
    count_ok = 0
    count_no_query_marker = 0
    count_empty = 0
    sample_rows: list[dict] = []

    for i, row in enumerate(tqdm(ds, desc="Parsing queries", unit="query")):
        raw_text = str(row[cfg.text_column] or "")
        parsed = parse_agentir_text(raw_text)
        final_query = build_final_query(
            parsed=parsed,
            row=row,
            raw_text=raw_text,
            expansion_mode=cfg.expansion.mode,
            expansion_cfg=cfg.expansion,
            hyde_generator=hyde_generator,
        )

        out = {
            "id": str(row[id_column]) if id_column else str(i),
            "query": final_query,
        }
        parsed_rows.append(out)

        status = parsed["parse_status"]
        if status == "ok":
            count_ok += 1
        elif status == "no_query_marker":
            count_no_query_marker += 1
        elif status == "empty":
            count_empty += 1

        if len(sample_rows) < int(cfg.print_samples):
            sample_rows.append(
                {
                    "id": out["id"],
                    "status": parsed["parse_status"],
                    "expansion_mode": cfg.expansion.mode,
                    "query_preview": (
                        (out["query"][:160] + "...")
                        if len(out["query"]) > 160
                        else out["query"]
                    ),
                }
            )

    write_jsonl(output_path, parsed_rows)

    total = len(parsed_rows)
    print("\nDone")
    print(f"dataset={cfg.dataset} split={cfg.split}")
    print(f"expansion.mode={cfg.expansion.mode}")
    print(f"rows={total:,} output={output_path}")
    print(
        "status counts: "
        f"ok={count_ok:,} "
        f"no_query_marker={count_no_query_marker:,} "
        f"empty={count_empty:,}"
    )
    print("\nSamples:")
    for s in sample_rows:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
