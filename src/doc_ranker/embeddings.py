import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/multilingual-e5-base"


def load_model(model_name: str = MODEL_NAME):
    """Carrega o tokenizer e o modelo do Hugging Face Hub (fica em cache depois da primeira vez)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Média dos vetores dos tokens de cada sequência, ignorando o padding."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode(texts: list[str], tokenizer, model, prefix: str,
           batch_size: int = 32, max_length: int = 512) -> Tensor:
    """
    Gera embeddings normalizados (L2) para uma lista de textos.

    prefix -- "query: " para consultas ou "passage: " para documentos, como o E5 exige
    """
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = [prefix + text for text in texts[start:start + batch_size]]
        batch = tokenizer(batch_texts, max_length=max_length, padding=True,
                          truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**batch)

        embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
        all_embeddings.append(F.normalize(embeddings, p=2, dim=1))

    return torch.cat(all_embeddings, dim=0)