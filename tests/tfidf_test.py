from pylate.models.utils import TokenTFIDFStats
from pylate.models import ColBERT
from pylate import evaluation
from tqdm.rich import tqdm

# Assuming you have a ColBERT model
model_name = "lightonai/GTE-ModernColBERT-v1"
model = ColBERT(
    model_name_or_path=model_name,
    document_length=300,
    query_length=32,
)
docs, queries, qrels = evaluation.load_beir(dataset_name="nfcorpus", split="test")

# Tokenize your corpus
tokenized_docs = [model.tokenizer.encode(doc["text"]) for doc in docs]

# Collect TF-IDF statistics
stats = TokenTFIDFStats()
special_token_ids = [
    model.tokenizer.convert_tokens_to_ids(token)
    for token in model.tokenizer.all_special_tokens
]
print(f"Excluding {len(special_token_ids)} special tokens")
tokenized_docs_excluded = [
    [token for token in doc if token not in special_token_ids]
    for doc in tqdm(tokenized_docs, desc="Excluding special tokens")
]
stats.fit(tokenized_docs_excluded)

print(stats)

# Get corpus statistics
corpus_stats = stats.get_corpus_statistics()

# Global IDF-based pruning: find common tokens to prune
common_tokens = stats.get_low_idf_tokens(k=20)
common_token_names = [model.tokenizer.decode(token) for token in common_tokens]
common_token_dict = {tok: name for tok, name in zip(common_tokens, common_token_names)}
print(f"Tokens to prune globally: {common_token_dict}")

# Document-wise IDF pruning: get token scores for a specific document
doc_scores = stats.get_document_token_scores(doc_idx=0, score_type="tfidf")
# Returns: {token_id: tfidf_score, ...}

# Sort tokens by TF-IDF and keep top-k
sorted_tokens = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
top_k_tokens = [token_id for token_id, score in sorted_tokens[:10]]
