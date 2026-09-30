import torch
from torch import Tensor


def mrr_at_k(scores: Tensor, doc_ids: list[int], relevant_ids: dict[int, int], k: int = 10) -> float:
    """
    Calcula o MRR@k (Mean Reciprocal Rank) de um conjunto de consultas.

    scores       -- matriz (n_queries, n_docs): scores[i][j] é a similaridade
                    entre a consulta i e o documento da coluna j
    doc_ids      -- id oficial de cada coluna de `scores`, na mesma ordem
    relevant_ids -- {índice da consulta em `scores`: id do documento relevante}
    k            -- profundidade de corte do ranking (k=10 -> MRR@10)
    """
    doc_id_to_col = {doc_id: col for col, doc_id in enumerate(doc_ids)}
    reciprocal_ranks = []

    for query_idx, relevant_id in relevant_ids.items():
        ranking = torch.argsort(scores[query_idx], descending=True)[:k]
        relevant_col = doc_id_to_col.get(relevant_id)

        if relevant_col is None:
            reciprocal_ranks.append(0.0)
            continue

        hits = (ranking == relevant_col).nonzero()

        if hits.numel() == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / (hits.item() + 1))

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)