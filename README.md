# Groundtruth — The Ship Gate

Evaluation harness para medir mudanças em retrieval de RAG antes de merge ou deploy. O projeto adapta o exercício Groundtruth do BASWE para o domínio **Cast — AI Engineering Handbook**: em vez de arquivos jurídicos, o corpus é o manual em [`docs/sources/ai_engineering.md`](docs/sources/ai_engineering.md).

A regra do projeto é simples: uma mudança de retrieval precisa mostrar seus números e passar pelo gate de regressão antes de seguir.

## O que é avaliado

O harness usa um golden set de 100 perguntas e passagens relevantes, cobrindo 12 categorias. Conforme confirmação do responsável pelo projeto, os 100 itens foram revisados e aprovados; cada registro em [`golden_set.jsonl`](golden_set.jsonl) tem `review_status: "approved"`.

A avaliação calcula:

- **Recall@10:** fração dos trechos relevantes encontrados entre os dez primeiros resultados.
- **MRR:** posição do primeiro trecho relevante, em média entre as queries.
- **nDCG@10:** qualidade da ordem dos resultados em relação à ordem ideal.
- **Faithfulness** e **Answer Relevancy:** métricas de geração avaliadas separadamente com RAGAS.

As métricas de retrieval são agregadas no total e por categoria. Os detalhes por query ficam nos arquivos JSON em [`results/`](results/).

## Arquitetura

1. O corpus é dividido em chunks de 100 ou 200 palavras, com overlap de 20 ou 40 palavras, respectivamente.
2. O golden set é remapeado aos chunks por spans do documento, para que cada tamanho de chunk seja avaliado contra seus próprios IDs.
3. O retrieval compara busca densa e busca híbrida. A variante reranked recupera até 30 candidatos densos e usa `gpt-4.1-mini` para ordená-los, retornando os dez primeiros.
4. Embeddings e resultados do reranker são cacheados em `cache/` para reaproveitar chamadas já feitas.
5. Os resultados são gravados em JSON e a comparação consolidada em CSV.

O pacote `groundtruth/` contém os arquivos vazios que organizam as próximas responsabilidades: `corpus/` para fontes e chunking; `retrieval/` para dense, filtros e grafo; `knowledge_graph/` para relações; `agents/` para tools e traces; `evaluation/` para métricas; `experiments/` para execução; e `cache.py` para a infraestrutura comum de cache. Eles são apenas a estrutura por enquanto. O código ativo continua em `main.py` e `generation_eval.py`; vamos migrá-lo módulo por módulo.

As três comparações estão descritas em [`configs/experiments/`](configs/experiments/): retrieval controlado, agente controlado e agente ajustado por backend. O perfil e os projetos em [`docs/sources/pedro_castanheira.md`](docs/sources/pedro_castanheira.md) e [`docs/sources/projects.md`](docs/sources/projects.md) foram organizados a partir do currículo fornecido.

O cache ativo continua no `main.py`: embeddings usam SHA-256 de texto e modelo; reranking usa pergunta, candidatos, `k`, modelo e versão do prompt. A arquitetura proposta mantém uma infraestrutura comum em `groundtruth/cache.py`, mas deixa cada etapa definir os dados de sua própria chave. Os namespaces planejados são `cache/text_embeddings/`, `cache/reranker/`, `cache/knowledge_graph/` e `cache/agent_runs/`. Assim, um resultado só é reutilizado quando suas entradas e versões correspondem.

Para o grafo, a chave deve incluir fingerprint do corpus, versão do chunking, modelo extrator e versão do prompt. Para uma execução de agente, deve incluir ID da pergunta, backend/índice, modelo, versão do prompt, versão das tools e limites de execução. A organização deve preservar as chaves e caminhos atuais de embedding e reranking durante a migração, evitando chamadas pagas repetidas.

Principais arquivos legados: [`main.py`](main.py) orquestra o pipeline atual; [`test_retrieval.py`](test_retrieval.py) cobre o gate e a regressão deliberada; [`baselines/retrieval_v1.json`](baselines/retrieval_v1.json) versiona a referência do gate.

## Experimentos e resultados

Resultados medidos em Recall@10, MRR e nDCG@10:

| Configuração | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| Dense, chunk 100 | 0.9742 | 0.9045 | 0.8831 |
| Dense, chunk 200 | **1.0000** | 0.8580 | 0.8817 |
| Hybrid, chunk 200 | 0.9900 | 0.9117 | 0.9208 |
| Dense + GPT reranker, chunk 200 | 0.9900 | **0.9950** | **0.9817** |

Os resultados completos estão em [`results/retrieval_comparison.csv`](results/retrieval_comparison.csv). O baseline versionado é Dense chunk200: Recall@10 1.0, MRR 0.8580 e nDCG@10 0.8817.

## Recomendação

A recomendação padrão é **Hybrid chunk200**: mantém 0.99 de Recall@10, melhora MRR e nDCG sobre o baseline e não adiciona uma chamada de LLM no retrieval. A queda de Recall é exatamente 1 ponto percentual, portanto está no limite permitido pelo gate.

Use **Dense chunk200** quando máxima cobertura e simplicidade forem a prioridade. Considere **Dense + GPT reranker** quando a ordem dos resultados for mais importante: ele alcançou os melhores MRR e nDCG, mas adiciona custo, latência e dependência de serviço. Custo e latência não foram medidos, então essa decisão precisa de medições próprias antes de uma adoção com limites de produção.

A análise detalhada dos tradeoffs e das cinco queries problemáticas está em [`results/recommendation.md`](results/recommendation.md).

## Avaliação de geração

A amostra determinística contém 12 queries, uma por categoria. Foram 10 avaliações válidas e 2 falhas do avaliador RAGAS, excluídas das médias sem receber score zero. Nas avaliações válidas:

- Faithfulness média: **1.0000**
- Answer Relevancy média: **0.8982**

Os dados estão em [`results/generation_eval_sample12.json`](results/generation_eval_sample12.json) e o resumo em [`results/generation_eval_summary.json`](results/generation_eval_summary.json). Essa amostra não é uma comparação de geração entre as quatro configurações de retrieval.

## Análise de falhas

As cinco queries inspecionadas apontam para três causas principais:

1. Trechos de evidência podem atravessar a fronteira entre chunks, deixando parte da resposta fora do top 10.
2. Chunks sobrepostos podem competir pela posição mais alta, causando falha de ranking.
3. O ground truth binário não diferencia uma resposta direta de um trecho que só oferece contexto útil; isso afeta a interpretação de algumas penalidades em nDCG.

Uma evolução possível é usar níveis graduados de relevância para distinguir resposta direta, trecho altamente relevante, contexto útil e irrelevante.

## Gate de CI

O baseline é [`baselines/retrieval_v1.json`](baselines/retrieval_v1.json). O gate compara Recall@10 e permite queda máxima de 0.01: queda menor ou igual a 1 ponto percentual passa; queda maior falha.

A suíte inclui testes para os limites do gate, para o retrieval avaliado e para um retriever deliberadamente quebrado que retorna uma lista vazia. O workflow do GitHub Actions roda `test_retrieval.py` em pull requests e pushes para `master`.

Para executar localmente:

```bash
python -m pip install -r requirements.txt
pytest test_retrieval.py -v
```

A avaliação de retrieval que gera embeddings e executa o reranker usa a API da OpenAI; configure `OPENAI_API_KEY` no ambiente. Os artefatos em `results/` e `cache/` permitem consultar os resultados já produzidos sem repetir os experimentos.

## Limitações

- O corpus é um handbook de Engenharia de IA, não um conjunto de arquivos jurídicos; é uma adaptação do caso de uso do exercício original.
- O ground truth usa relevância binária, embora os trechos tenham graus diferentes de utilidade.
- Custo e latência dos experimentos não foram medidos.
- A avaliação de geração cobre 12 queries e teve 2 falhas do avaliador.
- A recomendação sobre produção se baseia em qualidade de retrieval e complexidade esperada; deve ser complementada por medições de custo e latência no ambiente-alvo.
