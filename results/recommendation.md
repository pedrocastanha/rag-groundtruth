# Recomendação de configuração para produção

## Decisão

Recomendamos **Hybrid chunk200** como configuração padrão para produção, com Dense chunk200 como alternativa conservadora e o reranker como opção quando a qualidade do ranking justificar o custo e a latência adicionais.

Hybrid chunk200 mantém Recall@10 em 0.99, exatamente 1 ponto percentual abaixo do baseline Dense chunk200. Como o gate permite uma queda de até 1 ponto percentual, essa configuração passa no critério atual. Ao mesmo tempo, melhora MRR e nDCG em relação ao baseline, sem adicionar uma chamada de LLM ao fluxo de retrieval.

## Resultados de retrieval

| Configuração | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| Dense chunk100 | 0.9742 | 0.9045 | 0.8831 |
| Dense chunk200 | **1.0000** | 0.8580 | 0.8817 |
| Hybrid chunk200 | 0.9900 | 0.9117 | 0.9208 |
| Dense + GPT reranker, chunk200 | 0.9900 | **0.9950** | **0.9817** |

## Tradeoffs

- **Dense chunk100:** tem resultados razoáveis de ranking, mas o menor Recall@10 entre as configurações avaliadas. Não é a recomendação padrão diante das alternativas medidas.
- **Dense chunk200:** alcança o maior Recall@10 e usa o fluxo mais simples. É a escolha conservadora quando evitar perda de cobertura é mais importante que posicionar os trechos relevantes no topo.
- **Hybrid chunk200:** perde 1 ponto percentual de Recall em relação ao baseline, dentro do limite do gate, e melhora MRR e nDCG. É o melhor equilíbrio medido entre cobertura, ordem dos resultados e complexidade, sem reranking por LLM.
- **Dense + GPT reranker:** obtém o melhor MRR e nDCG, mantendo Recall@10 em 0.99. A chamada adicional de `gpt-4.1-mini` traz custo, latência e dependência operacional que ainda não foram quantificados neste projeto.

Custo e latência não foram medidos nos experimentos. Portanto, a comparação desses fatores é qualitativa: o reranker adiciona uma etapa de LLM; os resultados disponíveis não permitem estimar o impacto em dinheiro ou tempo de resposta.

## Uso recomendado por prioridade

- Escolha **Hybrid chunk200** como padrão quando o objetivo for equilibrar recall e qualidade de ranking.
- Escolha **Dense chunk200** quando Recall máximo e simplicidade forem os requisitos prioritários.
- Considere **Dense + GPT reranker** quando MRR e nDCG forem prioritários e medições de custo e latência confirmarem que o ganho atende aos limites do produto.

A configuração reranked tem Recall@10 de 0.99 contra 1.00 do baseline. A queda é exatamente 1 ponto percentual e, pela regra implementada, passa no gate. Isso não significa que não houve queda; significa que ela está no limite permitido.

## Generation evaluation

A avaliação de geração foi executada em uma amostra determinística de 12 queries, com uma query por categoria. Houve 10 avaliações válidas e 2 falhas do avaliador RAGAS. As falhas foram excluídas das médias, sem atribuir score zero.

Nas avaliações válidas, Faithfulness média foi 1.0 e Answer Relevancy média foi aproximadamente 0.8982. Esses valores descrevem a amostra avaliada; não demonstram que todas as 100 queries foram avaliadas quanto à geração nem permitem comparar as quatro configurações de retrieval.

## Análise de cinco queries problemáticas

1. **Cobertura na fronteira de chunks:** uma resposta relevante atravessa `chunk200_001` e `chunk200_002`. O segundo trecho foi recuperado, mas o primeiro ficou fora do top 10.
2. **Relevância desigual no ground truth:** para a pergunta sobre respostas ruins apesar de recall adequado, `chunk200_034` responde diretamente; `chunk200_011` oferece contexto mais genérico. A rotulagem binária conta ambos igualmente.
3. **Ordem dos resultados e overlap:** `chunk200_021` apareceu antes do trecho relevante `chunk200_022`. Os chunks têm conteúdo sobreposto na seção de reranking.
4. **Relevância desigual no ground truth:** `chunk200_038` responde diretamente à pergunta sobre feature flags; `chunk200_034` é contexto relacionado. A avaliação binária trata os dois como igualmente relevantes.
5. **Relevância desigual no ground truth:** `chunk200_023` contém a explicação direta sobre MRR e Recall; `chunk200_021` é contexto geral de reranking. Ambos contam igualmente no ground truth atual.

Os casos sugerem três fontes distintas de erro: cobertura nas fronteiras dos chunks, ordenação entre trechos sobrepostos e falta de níveis de relevância no ground truth. Uma evolução possível é usar graded relevance, por exemplo: 3 para resposta direta, 2 para trecho altamente relevante, 1 para contexto útil e 0 para irrelevante.

## Limitações e próximos passos

- Conforme confirmação do responsável pelo projeto, as 100 queries foram aprovadas por revisão humana; o campo `review_status` está marcado como `approved` em todas as entradas.
- Este projeto usa o handbook Cast como corpus, uma adaptação do cenário de pesquisa jurídica apresentado no guia.
- O ground truth atual usa relevância binária e não expressa diferenças entre resposta direta e contexto útil.
- Os resultados de retrieval não incluem medições comparáveis de custo ou latência.
- A avaliação de geração cobre uma amostra de 12 queries, e duas avaliações falharam no avaliador.
- Antes de ampliar o uso do reranker, medir custo e latência no fluxo representativo de produção.
