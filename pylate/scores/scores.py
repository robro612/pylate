from __future__ import annotations

import inspect
import numpy as np
import torch
from transformers import TrainerCallback

from ..utils.tensor import convert_to_tensor

def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities
    between the query and the document.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)
    queries_mask
        The mask for the queries embeddings. Shape: (batch_size, num tokens queries)
    documents_mask
        The mask for the documents embeddings. Shape: (batch_size, num tokens documents)

    Returns
    -------
    scores
        The scores between the queries and documents. Shape: (batch_size, batch_size)
    
    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [10.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> documents_mask = torch.tensor([
    ...     [1., 1., 1.],
    ...     [1., 0., 1.],
    ...     [1., 1., 1.],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])

    >>> scores = colbert_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )

    >>> scores
    tensor([[  10.,  10., 1000.],
            [  20.,  20., 2000.],
            [  0.,  0., 0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)
    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    # (batch_size, batch_size, q_seq_len, d_seq_len) -> (batch_size, batch_size)
    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


def colbert_scores_pairwise(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities
    between the query and the document for corresponding pairs.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [1.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> scores = colbert_scores_pairwise(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings
    ... )

    >>> scores
    tensor([  10.,  200., 3000.])

    """
    scores = []

    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings
    ):
        query_embedding = convert_to_tensor(query_embedding)
        document_embedding = convert_to_tensor(document_embedding)

        query_document_score = torch.einsum(
            "sh,th->st",
            query_embedding,
            document_embedding,
        )

        scores.append(query_document_score.max(axis=-1).values.sum())

    return torch.stack(scores, dim=0)


def colbert_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. This scoring function is dedicated to the knowledge distillation pipeline.

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[[10.], [0.], [1.]], [[20.], [0.], [1.]], [[30.], [0.], [1.]]],
    ...     [[[0.], [100.], [1.]], [[0.], [200.], [1.]], [[0.], [300.], [1.]]],
    ...     [[[1.], [0.], [1000.]], [[1.], [0.], [2000.]], [[10.], [0.], [3000.]]],
    ... ])
    >>> documents_mask = torch.tensor([
    ...     [[0., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])
    >>> colbert_kd_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )
    tensor([[ 1.,  20.,  30.],
            [200., 400., 600.],
            [  0.,   0.,   0.]])

    """
    # (batch_size, q_seq_len, embedding_size)
    queries_embeddings = convert_to_tensor(queries_embeddings)
    # (batch_size, n_ways, d_seq_len, embedding_size)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    # (batch_size, n_ways, batch_size, q_seq_len, d_seq_len)
    scores = torch.einsum(
        "ash,abth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        mask = convert_to_tensor(documents_mask)
        scores = scores * mask.unsqueeze(2)

    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores



class ScheduledXTRScore:
    """Callable wrapper for XTR score functions with scheduled k_prime.
    
    This class allows k_prime to be annealed during training based on the current
    training step. The scheduler function should take the current step as input
    and return the k_prime value to use. Works with both contrastive and KD score functions.
    
    Parameters
    ----------
    score_fn
        The XTR score function to wrap. Must accept (queries_embeddings, documents_embeddings,
        queries_mask, documents_mask, k_prime, use_normalizer_Z, Z_clamp_value).
        Examples: xtr_contrastive_training_scores, xtr_kd_training_scores
    k_prime_scheduler
        Callable that takes the current training step (int) and returns the k_prime (int)
        to use at that step. Example: lambda step: min(100, 10 + step // 100)
    use_normalizer_Z
        Whether to use the normalizer Z in the score computation.
    Z_clamp_value
        Minimum value to clamp Z to prevent division by zero.
    start_normalizer_Z_at_step
        Step at which to start using the normalizer Z.
    
    Examples
    --------
    >>> from pylate.scores import ScheduledXTRScore, xtr_contrastive_training_scores
    >>> 
    >>> # Linear annealing from 10 to 100 over 10000 steps
    >>> def k_prime_scheduler(step: int) -> int:
    ...     k_prime_start = 10
    ...     k_prime_end = 100
    ...     total_steps = 10000
    ...     if step >= total_steps:
    ...         return k_prime_end
    ...     progress = step / total_steps
    ...     return int(k_prime_start + (k_prime_end - k_prime_start) * progress)
    >>> 
    >>> scheduled_score = ScheduledXTRScore(
    ...     score_fn=xtr_contrastive_training_scores,
    ...     k_prime_scheduler=k_prime_scheduler,
    ...     use_normalizer_Z=False,
    ... )
    >>> 
    >>> # Update step before each forward pass (done via callback)
    >>> scheduled_score.update_step(5000)
    >>> 
    >>> # Use as score_metric in loss functions
    >>> # train_loss = losses.Contrastive(model=model, score_metric=scheduled_score)
    >>> # train_loss = losses.Distillation(model=model, score_metric=scheduled_score)
    """
    
    def __init__(
        self,
        score_fn,  # Callable that accepts XTR score function signature
        k_prime_scheduler,  # Callable[[int], int]
        use_normalizer_Z: bool = False,
        Z_clamp_value: float = 1.0,
        start_normalizer_Z_at_step: int = 0,
        impute_scores_instead_of_zero: bool = False,
        log_gradients: bool = False,
        log_prefix: str = "xtr",
        log_frequency: int = 1,
        log_z_stats: bool = False,
        positive_document_index: int = 0,
    ):
        self.score_fn = score_fn
        self.k_prime_scheduler = k_prime_scheduler
        self.use_normalizer_Z = use_normalizer_Z
        self.Z_clamp_value = Z_clamp_value
        self.start_normalizer_Z_at_step = start_normalizer_Z_at_step
        self.impute_scores_instead_of_zero = impute_scores_instead_of_zero
        self.log_gradients = log_gradients
        self.log_prefix = log_prefix
        self.log_frequency = log_frequency
        self.log_z_stats = log_z_stats
        self.positive_document_index = positive_document_index
        self.current_step = 0
        self.current_k_prime = None
    
    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute XTR scores with scheduled k_prime.
        
        Parameters
        ----------
        queries_embeddings
            Query embeddings. Shape: (batch_size, num_tokens_queries, embedding_size)
        documents_embeddings
            Document embeddings. Shape: (batch_size, num_tokens_documents, embedding_size) for contrastive
            or (batch_size, n_ways, num_tokens_documents, embedding_size) for KD
        queries_mask
            Mask for query embeddings. Shape: (batch_size, num_tokens_queries)
        documents_mask
            Mask for document embeddings. Shape: (batch_size, num_tokens_documents) for contrastive
            or (batch_size, n_ways, num_tokens_documents) for KD
        
        Returns
        -------
        scores
            XTR scores. Shape: (batch_size, batch_size) for contrastive or (batch_size, n_ways) for KD
        """
        # Compute current k_prime based on step
        self.current_k_prime = self.k_prime_scheduler(self.current_step)
        
        # Determine whether to use normalizer Z based on current step
        should_use_normalizer_Z = (
            self.use_normalizer_Z 
            and self.current_step >= self.start_normalizer_Z_at_step
        )
        
        # Call the wrapped score function with scheduled k_prime

        args = dict(
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
            k_prime=int(self.current_k_prime),  # Ensure it's an integer
            use_normalizer_Z=should_use_normalizer_Z,
            Z_clamp_value=self.Z_clamp_value,
            positive_document_index=self.positive_document_index,
        )
        if self.impute_scores_instead_of_zero:
            args["impute_scores_instead_of_zero"] = self.impute_scores_instead_of_zero
        if self.log_gradients:
            args["log_gradients"] = self.log_gradients
            args["log_prefix"] = self.log_prefix
            args["log_frequency"] = self.log_frequency
        if self.log_z_stats:
            args["log_z_stats"] = self.log_z_stats
            args["log_frequency"] = self.log_frequency

        score_params = inspect.signature(self.score_fn).parameters
        args = {key: value for key, value in args.items() if key in score_params}
        
        return self.score_fn(**args)
    
    def update_step(self, step: int):
        """Update the current training step.
        
        Parameters
        ----------
        step
            Current global training step.
        """
        self.current_step = step

class KPrimeSchedulerCallback(TrainerCallback):
    """Callback to update k_prime scheduler with current training step.
    
    This callback should be added to the trainer when using ScheduledXTRScore
    to ensure the k_prime value is updated based on the current training step.
    
    Examples
    --------
    >>> from pylate.scores import ScheduledXTRScore, KPrimeSchedulerCallback
    >>> from pylate.scores import xtr_contrastive_training_scores
    >>> 
    >>> def k_prime_scheduler(step: int) -> int:
    ...     return min(100, 10 + step // 100)
    >>> 
    >>> scheduled_score = ScheduledXTRScore(
    ...     score_fn=xtr_contrastive_training_scores,
    ...     k_prime_scheduler=k_prime_scheduler,
    ... )
    >>> 
    >>> # Add to trainer
    >>> trainer.add_callback(KPrimeSchedulerCallback(scheduled_score))
    """
    
    def __init__(self, scheduled_score_fn):
        """Initialize the callback.
        
        Parameters
        ----------
        scheduled_score_fn
            The ScheduledXTRScore instance to update.
        """
        self.scheduled_score_fn = scheduled_score_fn
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update the step in the scheduled score function.
        
        Parameters
        ----------
        args
            Training arguments.
        state
            Training state containing global_step.
        control
            Training control object.
        
        Returns
        -------
        control
            The training control object.
        """
        if hasattr(self.scheduled_score_fn, 'update_step'):
            self.scheduled_score_fn.update_step(state.global_step)
        return control
