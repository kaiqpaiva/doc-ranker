"""
Smoke test do E5 multilingual no mMARCO (português).

Lê os TSVs direto do Hub por streaming HTTP (sem load_dataset, sem baixar
a coleção inteira). Pega N queries do dev.small e M passagens da coleção,
e ranqueia as passagens para cada query.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------- config
REPO = "unicamp-dl/mmarco"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# 'google' = tradução usada no paper. 'helsinki' é a alternativa.
QUERIES_URL = f"{BASE}/data/google/queries/dev/portuguese_queries.dev.small.tsv"
COLLECTION_URL = f"{BASE}/data/google/collections/portuguese_collection.tsv"

MODEL_NAME = "intfloat/multilingual-e5-base"  # E5 monolíngue inglês não serve p/ pt
N_QUERIES = 5
N_PASSAGES = 2_000
TOP_K = 3
BATCH_SIZE = 32
MAX_LEN = 512


# ------------------------------------------------------------ carregamento
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Lendo queries (dev.small)...")
    queries = pd.read_csv(
        QUERIES_URL,
        sep="\t",
        names=["qid", "query"],
        nrows=N_QUERIES,
        quoting=3,  # QUOTE_NONE: TSVs do MS MARCO não escapam aspas
    )

    print(f"Lendo as primeiras {N_PASSAGES} passagens da coleção...")
    passages = pd.read_csv(
        COLLECTION_URL,
        sep="\t",
        names=["pid", "text"],
        nrows=N_PASSAGES,
        quoting=3,
    )
    return queries, passages


# --------------------------------------------------------------- modelo
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Média dos tokens ignorando padding. E5 exige isso — não use o CLS."""
    masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.inference_mode()
def encode(texts: list[str], tokenizer, model, device) -> Tensor:
    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        tokens = tokenizer(
            batch,
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        out = model(**tokens)
        emb = average_pool(out.last_hidden_state, tokens["attention_mask"])
        embeddings.append(F.normalize(emb, p=2, dim=1).cpu())

        print(f"  {min(start + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")

    print()
    return torch.cat(embeddings)


# ----------------------------------------------------------------- main
def main() -> None:
    queries, passages = load_data()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nCarregando {MODEL_NAME} em {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # Os prefixos são obrigatórios no E5 e são assimétricos.
    print("\nCodificando queries...")
    q_emb = encode([f"query: {q}" for q in queries["query"]], tokenizer, model, device)

    print("Codificando passagens...")
    p_emb = encode([f"passage: {p}" for p in passages["text"]], tokenizer, model, device)

    # Vetores normalizados -> produto interno == similaridade de cosseno.
    scores = q_emb @ p_emb.T

    for i, row in queries.reset_index(drop=True).iterrows():
        print(f"\n{'=' * 70}\nQuery [{row.qid}]: {row['query']}\n{'-' * 70}")
        top = scores[i].topk(TOP_K)
        for rank, (score, idx) in enumerate(zip(top.values, top.indices), start=1):
            passage = passages.iloc[idx.item()]
            print(f"{rank}. ({score:.4f}) [pid {passage.pid}] {passage.text[:180]}...")


if __name__ == "__main__":
    main()