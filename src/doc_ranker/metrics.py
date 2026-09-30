import torch
from torch import Tensor


def mrr_at_k(scores: Tensor, doc_ids: list[int], relevant_ids: dict[int, int], k: int = 10) -> float:
    # ... (docstring e corpo igual ao do notebook)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)