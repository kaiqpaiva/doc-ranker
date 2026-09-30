# doc-ranker

Ranqueamento semântico de documentos com embeddings multilíngues. Dada uma consulta e um conjunto de passagens, o projeto gera um vetor para cada texto com o modelo [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) e ordena as passagens pela similaridade de cosseno com a consulta.

A avaliação usa o [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco), a versão em português do MS MARCO, com o gabarito oficial e a métrica MRR@10.

O notebook principal também foi escrito para servir de material de estudo: cada etapa tem uma explicação do que o código faz, do formato dos tensores e do porquê de cada escolha.

## Como funciona

1. **Prefixos.** O E5 espera `query: ` na frente das consultas e `passage: ` na frente dos documentos. É assim que ele diferencia os dois papéis.
2. **Tokenização e modelo.** O texto vira tokens e o modelo devolve um vetor de 768 dimensões para cada token.
3. **Mean pooling.** A média dos vetores dos tokens gera um único vetor por texto. A `attention_mask` garante que os tokens de padding fiquem de fora da média.
4. **Normalização L2.** Todos os vetores passam a ter comprimento 1, então o produto interno entre eles é igual à similaridade de cosseno.
5. **Ranqueamento.** A multiplicação da matriz de consultas pela de documentos dá todos os scores de uma vez, e as passagens são ordenadas do maior score para o menor.

## Resultado

| Configuração | MRR@10 |
|---|---|
| Sem prefixos `query:` / `passage:` | 0,966 |
| Com prefixos | 0,970 |

Avaliado em 100 consultas do conjunto *dev small* do mMARCO em português, com um pool de 200 documentos: as 100 passagens relevantes dessas consultas e mais 100 passagens da coleção como distratoras.

Esse número serve para validar que o pipeline funciona e para comparar variações entre si. Ele não é comparável com os resultados publicados para o MS MARCO, que são medidos contra a coleção inteira de ~8,8 milhões de passagens e ficam na faixa de 0,30 a 0,40. Com 200 candidatos a tarefa é bem mais fácil.

## Estrutura

```
doc-ranker/
├── src/doc_ranker/
│   ├── embeddings.py      carregamento do modelo, mean pooling e geração de embeddings
│   ├── ranking.py         top-k por consulta
│   └── metrics.py         MRR@k
├── notebooks/
│   └── pipeline_e_avaliacao.ipynb   pipeline completo explicado passo a passo + avaliação
├── examples/
│   └── demo_multilingue.py          demo rápida com consultas em português, inglês e chinês
├── data/
│   └── passagens_relevantes_cache.json
└── requirements.txt
```

O arquivo em `data/` guarda as passagens relevantes já encontradas na coleção. Sem ele, o notebook precisa percorrer a coleção inteira em streaming, o que leva uns 11 minutos.

## Como rodar

Requer Python 3.10 ou mais recente.

```bash
git clone https://github.com/kaiqpaiva/doc-ranker.git
cd doc-ranker
python -m venv venv
```

Ativando o ambiente:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

O `datasets` fica travado abaixo da versão 4.0 porque o mMARCO é carregado por um script próprio, e esse tipo de script deixou de ser suportado nas versões novas.

Para a demo rápida:

```bash
python examples/demo_multilingue.py
```

Para o pipeline completo com avaliação, abra `notebooks/pipeline_e_avaliacao.ipynb` no Jupyter ou no VS Code e rode as células em ordem. Na primeira execução o modelo (~1 GB) é baixado do Hugging Face e fica em cache.

## Usando o pacote

```python
import sys
sys.path.append("src")

from doc_ranker import load_model, encode, top_k

tokenizer, model = load_model()

documentos = [
    "A taxa Selic é a taxa básica de juros da economia brasileira.",
    "Para remover manchas de café, aplique água fria imediatamente.",
    "Sagittarius A* é o buraco negro no centro da Via Láctea.",
]

consultas = encode(["como tirar mancha de café da roupa"], tokenizer, model, prefix="query: ")
docs = encode(documentos, tokenizer, model, prefix="passage: ")

scores = consultas @ docs.T
valores, indices = top_k(scores, k=2)

for score, idx in zip(valores[0], indices[0]):
    print(f"{score:.4f}  {documentos[idx]}")
```

## Próximos passos

- Aumentar o pool de distratoras para deixar a avaliação mais próxima do cenário real
- Indexar os vetores com FAISS para buscar em coleções maiores
- Comparar outros modelos de embedding com o mesmo protocolo de avaliação

## Referências

- Wang et al. *Text Embeddings by Weakly-Supervised Contrastive Pre-training*, 2022. (modelo E5)
- Bonifacio et al. *mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset*, 2021.

## Autores

- Kaique Barros Paiva ([github.com/kaiqpaiva](https://github.com/kaiqpaiva))
- Isabela Hissa Pinto ([github.com/hissapinto](https://github.com/hissapinto))
