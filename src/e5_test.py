import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = [
    # --- QUERIES (Índices 0, 1 e 2) ---
    'query: how much protein should a female eat',
    'query: 南瓜的家常做法',
    'query: como tirar mancha de café da roupa branca',
    
    # --- PASSAGES (Índices 3, 4, 5, 6 e 7) ---
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅",
    'passage: Para remover manchas de café recentes em tecidos brancos, aplique água fria imediatamente. Em seguida, esfregue suavemente o local com uma mistura de água morna, detergente neutro e algumas gotas de vinagre branco. Deixe agir por 15 minutos e lave a peça normalmente.',
    'passage: O buraco negro supermassivo no centro da nossa galáxia, a Via Láctea, é conhecido como Sagittarius A*. Ele possui uma massa equivalente a cerca de 4 milhões de vezes a do nosso Sol, exercendo uma força gravitacional extrema.',
    'passage: A taxa Selic é a taxa básica de juros da economia brasileira. Ela é definida a cada 45 dias pelo Comitê de Política Monetária (Copom) e serve de referência para todas as outras taxas de juros do país, incluindo financiamentos.'
]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
# average_pool: Modelos de linguagem geram um vetor para cada palavra/token da frase.
# Para comparar frases inteiras, precisamos de um único vetor que represente o texto todo.
# Essa função faz exatamente isso: ela calcula a média matemática dos vetores de todas as
# palavras da frase (técnica conhecida como Mean Pooling). Ela usa a attention_mask para
# garantir que palavras de preenchimento (padding) adicionadas artificialmente não entrem nessa média.

# Normaliza os vetores (Normalização L2) para que todos tenham um "comprimento" igual a 1
embeddings = F.normalize(embeddings, p=2, dim=1)
# embeddings[:3] pega os três primeiros itens (as 3 queries)
# embeddings[3:] pega do quarto item em diante (as 5 passages)
scores = (embeddings[:3] @ embeddings[3:].T) * 100

# Converte a matriz de tensores para uma lista do Python
scores_list = scores.tolist()

# Limpamos os prefixos "query: " e "passage: " apenas para a exibição ficar mais limpa
queries_limpas = [q.replace('query: ', '') for q in input_texts[:3]]
passages_limpas = [p.replace('passage: ', '') for p in input_texts[3:]]

# Loop para imprimir os resultados organizados
for i, query in enumerate(queries_limpas):
    print(f"\nPERGUNTA: '{query}'")
    print("-" * 70)
    
    # 1. Juntamos as notas (scores) com seus respectivos textos
    resultados = list(zip(scores_list[i], passages_limpas))
    
    # 2. Ordenamos a lista baseada na nota (x[0]), do maior para o menor (reverse=True)
    resultados_ordenados = sorted(resultados, key=lambda x: x[0], reverse=True)
    
    # 3. Imprimimos os resultados já ordenados
    for score, passage in resultados_ordenados:
        # Cortamos o texto em 60 caracteres
        texto_curto = passage[:60].replace('\n', ' ') + "..." 
        
        # Destaca visualmente se o score for maior que 85
        destaque = "✅" if score > 85 else "  "
        
        print(f"{destaque} Score: {score:.1f} | Texto: {texto_curto}")
