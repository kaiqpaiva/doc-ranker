import torch.nn.functional as F
import pandas as pd

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

df = pd.read_csv('data/raw/mini-dataset.csv')
queries_unicas = df.groupby('query_id')['query_text'].first().reset_index()
print(queries_unicas)

docs_unicos = df.drop_duplicates(subset=['doc_id'])[['doc_id', 'document_text']].reset_index(drop=True)
print(docs_unicos)

docs_texts = docs_unicos['document_text'].tolist()

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

batch_dict = tokenizer(docs_texts, max_length=512, padding=True,
                       truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
docs_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)

print(docs_embeddings.shape)
# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.