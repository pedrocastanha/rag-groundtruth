# Comparação dos benchmarks Groundtruth

Este relatório compara o benchmark V1, de um documento, com a rodada multidocumento e os agentes. Os números ajudam a entender cada configuração dentro do seu próprio experimento; eles **não formam um ranking direto entre V1 e multidocumento**, pois corpus, perguntas e tamanho da amostra são diferentes.

## O que cada estratégia faz

Pense nos chunks como páginas de uma biblioteca:

- **Flat (dense):** procura em todas as páginas das três fontes usando similaridade semântica. É a busca de referência sem uma escolha prévia de documento.
- **Filtered:** limita a busca às páginas de um documento escolhido. No experimento retrieval-only atual, usamos `expected_document_ids` do gabarito para escolher esse documento; por isso, essa avaliação recebe uma pista que flat não recebe. Nos agentes, a ferramenta `search_by_document` precisa escolher um ID válido.
- **Graph:** faz uma busca densa inicial para encontrar chunks-semente e usa as conexões do grafo para trazer outros chunks relacionados. No ranking, os seeds usam reciprocal rank e os chunks expandidos recebem um bônus conforme a distância no grafo.

Nos agentes, **controlled** mantém o mesmo modelo, instrução, limite e interface de ferramenta entre retrievers; muda o backend de retrieval. **Tuned** permite instruções e ferramentas específicas de cada backend. Controlled ajuda a comparar o retrieval com menos variáveis; tuned compara sistemas completos, mas não permite atribuir diferenças apenas ao retriever.

As métricas de retrieval têm focos diferentes: **Recall@10** mede quanto do conteúdo marcado como relevante aparece nos dez primeiros chunks; **MRR** recompensa encontrar o primeiro chunk relevante mais perto do topo; **nDCG@10** considera a ordem de todos os relevantes no top 10 em relação a uma ordenação ideal. Neste benchmark, relevância é binária: cada chunk de referência conta como relevante ou não.

## V1 — um documento

O V1 usa o handbook *Cast — AI Engineering Handbook*, um golden set de 100 queries e compara chunk sizes, dense, hybrid e reranking.

| Configuração | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| Dense, chunk 100 | 0.9742 | 0.9045 | 0.8831 |
| Dense, chunk 200 | **1.0000** | 0.8580 | 0.8817 |
| Hybrid, chunk 200 | 0.9900 | 0.9117 | 0.9208 |
| Dense + GPT reranker, chunk 200 | 0.9900 | **0.9950** | **0.9817** |

Dense chunk 200 encontrou todas as evidências relevantes no top 10 e é a referência de Recall. O reranker teve o melhor ranking, com Recall 0.99 — exatamente 1 ponto percentual abaixo, dentro do limite do gate. A recomendação V1 é quality-first: usar o reranker, aceitando a chamada adicional de LLM porque MRR e nDCG são superiores. Hybrid mantém Recall 0.99 e melhora MRR/nDCG sem essa chamada; é a alternativa quando simplicidade operacional, custo ou latência forem prioritários. Esses custos e latências não foram medidos no V1.

Na avaliação de geração do V1, 10 de 12 avaliações RAGAS foram válidas: Faithfulness média 1.0 e Answer Relevancy média ≈ 0.8982. Duas falhas do avaliador ficaram sem score, não receberam zero.

## Novo benchmark — três documentos

O novo corpus tem o handbook, o perfil do Pedro e os projetos pessoais. Foram aprovadas 18 queries iniciais, divididas entre uma fonte, cross-document e multi-hop. Os chunks usam tamanho 200 e overlap 40. A amostra ainda não chega às 100 queries do desafio original.

### Retrieval

| Estratégia | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| Flat / dense | 0.9167 | 0.7324 | 0.7557 |
| Filtered | **0.9722** | **0.7602** | **0.7908** |
| Graph | 0.8889 | 0.7269 | 0.7510 |

Flat foi registrado como baseline `multidoc_v1`. Filtered tem os melhores resultados da tabela, mas seu retrieval-only recebe do gabarito o documento correto. Esses números medem busca condicionada a saber a fonte, não a capacidade de escolher a fonte. Graph mudou os rankings, mas ficou abaixo de flat nas três métricas; sua queda de Recall frente ao baseline é 2.78 pp, acima do limite de 1 pp do gate.

### Agentes: custo, latência e uso de tools

Cada combinação abaixo tem 18 respostas. Custos são estimativas por tokens e preços configurados, não valores de fatura.

| Modo | Retriever | Chamadas de tool por resposta | Custo médio/resposta | Latência média | Falhas de tool |
|---|---|---:|---:|---:|---:|
| Controlled | flat | 1.06 (`retrieve`: 19) | US$ 0.000990 | 3.77 s | 0 |
| Controlled | filtered | 1.06 (`retrieve`: 19) | US$ 0.001138 | 4.00 s | 0 |
| Controlled | graph | 1.06 (`retrieve`: 19) | US$ 0.000916 | 3.72 s | 0 |
| Tuned | flat | 1.06 (`search_all`: 19) | US$ 0.001689 | 4.10 s | 0 |
| Tuned | filtered | 1.06 (`search_by_document`: 19) | US$ 0.001145 | 4.08 s | 0 |
| Tuned | graph | 1.72 (`graph_search`: 18; `fetch_source_chunks`: 13) | US$ 0.001831 | 4.87 s | 0 |

O agente tuned com GraphRAG fez mais chamadas e teve o maior custo e latência médios desta amostra. O schema de `search_by_document` passou a listar os IDs e títulos permitidos; isso eliminou as falhas de seleção de documento observadas na primeira execução.

O benchmark novo ainda não tem agregado comparável de Faithfulness e Answer Relevancy. Há um diagnóstico de `q001` no agente em condição controlada/busca geral (1.0 e 0.9515) e um caso cacheado GPT-4.1-mini/filtro em `q009` (1.0 e 0.6285); nenhum é média representativa. A avaliação matricial falhou por `Temporary failure in name resolution` ao acessar a API OpenAI. Scores indisponíveis ficam ausentes, nunca são convertidos em zero. **Faithfulness** verifica se a resposta é sustentada pelos trechos recuperados; **Answer Relevancy** mede se ela responde à pergunta. Consulte a matriz e o status mais recentes no [`README.md`](../README.md).

## O que os números permitem concluir

1. **O V1 tem avaliação de retrieval mais ampla:** usa 100 queries e inclui experimentos de chunking, hybrid e reranking. O reranker é a recomendação quality-first; Dense chunk 200 maximiza Recall; Hybrid evita uma chamada extra de LLM.
2. **O novo benchmark testa outro problema:** selecionar e combinar evidências de três fontes, inclusive em perguntas cross-document e multi-hop, além de observar comportamento do agente.
3. **Não interprete 0.9167 do flat novo como regressão contra o 1.0 do Dense 200 V1.** O V1 e o novo benchmark usam documentos, perguntas e gabaritos diferentes; a amostra nova também é muito menor.
4. **Filtered ainda não demonstra ganho autônomo de fonte.** Sua avaliação retrieval-only recebe a fonte correta; os resultados do agente são a parte que testa a escolha da ferramenta/documento.
5. **Graph ainda não justifica a complexidade observada.** No retrieval teve scores inferiores ao flat e o agente tuned usou mais tools, custo e latência. Isso é uma conclusão provisória para as 18 queries atuais.
6. **A comparação de geração está incompleta.** Não podemos comparar Faithfulness/Answer Relevancy entre V1 e o benchmark novo até RAGAS concluir a rodada nova.

## Estado do projeto

- V1: 100 queries, retrieval comparado e recomendação registrada em [`recommendation.md`](recommendation.md).
- Multidocumento: 18 queries aprovadas; retrieval e seis configurações de agente medidos. Baseline em [`../baselines/multidoc_v1.json`](../baselines/multidoc_v1.json).
- Pendente: completar RAGAS no multidocumento, ampliar o golden set novo se o alvo também for 100 e integrar a comparação do baseline multi-documento ao CI.

## Artefatos detalhados

- V1: [`retrieval_comparison.csv`](retrieval_comparison.csv), [`recommendation.md`](recommendation.md)
- Multidocumento: [`multidoc/comparison.md`](multidoc/comparison.md)
- Experimentos e métricas atualizadas: [`../README.md`](../README.md)
- Guia para entender os módulos: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
