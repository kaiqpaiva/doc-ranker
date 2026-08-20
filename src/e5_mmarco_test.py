import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Só as 5 primeiras linhas do dataset mmarco
print("Baixando os dados...\n")
dataset = load_dataset('unicamp-dl/mmarco', 'portuguese', split='train[:5]', trust_remote_code=True)
print(dataset)

queries = dataset['query']
passages = dataset['positive']
""" Como vem o dataset: 
    {
        "query": "qual a temperatura normal do corpo humano",
        "positive": "A temperatura normal do corpo humano geralmente varia de 36,5°C a 37,2°C. No entanto, ela pode variar de acordo com a idade, atividade física e o horário do dia."
    }, ...
"""
# Tem negative também, mas nao to usando

# Formatar do jeito que o modelo entende
formatted_queries = [f"query: {q}" for q in queries]
formatted_passages = [f"passage: {p}" for p in passages]

print("\nPrimeira Pergunta:", formatted_queries[0])
print("Primeiro Texto:", formatted_passages[0][:100], "...")