# Groundtruth multidocument — comparação inicial

## Escopo

Esta rodada usa 18 queries aprovadas, 48 chunks e três fontes: handbook de AI Engineering, perfil do Pedro e projetos pessoais. Retrieval foi avaliado com Recall@10, MRR e nDCG@10. Os agentes foram avaliados nos modos `controlled` e `tuned`, com 18 respostas por retriever.

Os valores são descritivos desta amostra. Ainda não representam evidência estatística suficiente para generalizar para outro corpus ou para produção.

## Retrieval

| Estratégia | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| Flat / dense | 0.9167 | 0.7324 | 0.7557 |
| Filtered | **0.9722** | **0.7602** | **0.7908** |
| Graph | 0.8889 | 0.7269 | 0.7510 |

`filtered` recebe `expected_document_ids` como escopo durante a avaliação de retrieval. Portanto, seus números usam a resposta correta sobre a fonte como pista e não são uma comparação equivalente ao flat para uso em produção. O modo de agente `tuned/filtered`, por outro lado, escolhe o documento por meio da ferramenta. Registramos flat/dense como baseline `multidoc_v1` por ser a referência sem essa pista externa.

O GraphRAG mudou os rankings das 18 queries após corrigirmos a fusão de scores: seeds densos usam reciprocal rank e chunks adicionados pela expansão recebem score baseado na distância do grafo. Mesmo assim, nesta amostra, graph ficou 2.78 pontos percentuais abaixo do flat em Recall@10, além de ligeiramente abaixo em MRR e nDCG. Comparado ao baseline flat, graph ultrapassa a queda máxima de 1 pp e falharia no gate. O workflow ainda não executa essa comparação automaticamente.

## Agentes

| Modo | Retriever | Chamadas por resposta | Chamadas de tool | Falhas de tool | Custo médio/resposta | Latência média |
|---|---|---:|---|---:|---:|---:|
| Controlled | flat | 1.06 | `retrieve`: 19 | 0 | US$ 0.000990 | 3.77 s |
| Controlled | filtered | 1.06 | `retrieve`: 19 | 0 | US$ 0.001138 | 4.00 s |
| Controlled | graph | 1.06 | `retrieve`: 19 | 0 | US$ 0.000916 | 3.72 s |
| Tuned | flat | 1.06 | `search_all`: 19 | 0 | US$ 0.001689 | 4.10 s |
| Tuned | filtered | 1.06 | `search_by_document`: 19 | 0 | US$ 0.001145 | 4.08 s |
| Tuned | graph | 1.72 | `graph_search`: 18; `fetch_source_chunks`: 13 | 0 | US$ 0.001831 | 4.87 s |

Todas as seis configurações produziram 18 respostas sem falhas finais de agente. O grafo tuned usou mais chamadas e teve a maior latência e o maior custo médio nesta rodada. O `filtered` tuned teve zero falhas depois que o schema passou a informar ao modelo os IDs de documento válidos e seus títulos.

## Faithfulness e Answer Relevancy

**A matriz comparável ainda não foi concluída.** Além do diagnóstico de `q001` no agente controlled/busca geral (Faithfulness 1.0; Answer Relevancy 0.9515), há um caso cacheado de GPT-4.1-mini com tool de filtro (`q009`: 1.0; 0.6285). Esse caso isolado não representa a média da estratégia. A tentativa da matriz completa encontrou falha de DNS ao acessar a API OpenAI; scores ausentes seguem sem valor, não recebem zero. A tabela e o status atualizados estão no [`README.md`](../../README.md).

## Leitura e recomendação provisória

- **Busca geral em todos os documentos** é o baseline mais simples e teve resultado superior à expansão por relações nas três métricas de retrieval.
- **Tool com filtro de documento** alcançou os melhores números de retrieval com uma pista que não está disponível na busca geral; não usar esse ganho para estimar desempenho sem classificação prévia de fonte.
- **Expansão por relações em grafo (GraphRAG)** alterou o ranking, mas não melhorou as métricas observadas e elevou chamadas/latência na condição ajustada. Não há evidência nesta rodada para escolhê-la como configuração de produção.
- A condição ajustada não reduziu chamadas de forma geral. Ela especializou a interface, mas a expansão em grafo fez mais operações e custou mais que as demais opções ajustadas.
- A recomendação de qualidade do agente fica pendente da matriz completa de Faithfulness e Answer Relevancy e de uma avaliação maior. O dataset atual contém 18 queries, não as 100 do desafio original.

## Artefatos e reprodução

- Retrieval: `results/multidoc/retrieval_controlled.json`
- Agentes controlled: `results/multidoc/agent_controlled.json`
- Agentes tuned: `results/multidoc/agent_tuned.json`
- Baseline: `baselines/multidoc_v1.json`
- Dataset aprovado: `datasets/golden_set_multidoc.jsonl`
- Relatório detalhado e métricas atualizadas: [`README.md`](../../README.md)
- Guia para entender os módulos: [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
