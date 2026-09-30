import torch
from torch import Tensor


def top_k(scores: Tensor, k: int = 10):
    """
    Retorna os k maiores scores de cada consulta e os índices dos documentos correspondentes.

    scores -- matriz (n_queries, n_docs) de similaridade
    """
    k = min(k, scores.shape[1])
    return torch.topk(scores, k=k, dim=1)