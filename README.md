# Groundtruth — The Ship Gate

Harness de avaliação para medir mudanças em sistemas RAG antes de merge ou deploy. Este README reúne o objetivo, as métricas, os resultados de cada experimento, as conclusões e os passos para reproduzir as avaliações. O benchmark V1 e o experimento multi-documento usam corpora e perguntas diferentes; seus scores não são diretamente comparáveis.

## 1. Objetivo do experimento

### V1: retrieval em um documento

1. Como tamanho de chunk, busca híbrida e reranking afetam cobertura e posição dos trechos relevantes?
2. É possível impedir merge de uma mudança que derruba Recall@10 acima de 1 ponto percentual?
3. Que configuração vale recomendar considerando Recall, qualidade de ranking e complexidade?

### Multi-documento e agente

1. Como se comparam a busca geral em todos os documentos, a busca com filtro de documento por tool e a expansão de contexto por relações em grafo (GraphRAG) num corpus com três fontes?
2. O agente consegue escolher uma fonte por meio de ferramentas e instruções próprias?
3. Como o comportamento operacional muda entre `controlled` e `tuned`?
4. Mantendo perguntas, fonte, prompt e ferramentas iguais, como GPT-4.1-mini, GPT-4o-mini e GPT-6 Luna (esforço baixo) variam em chamadas, custo e latência?
5. As respostas têm suporte nos chunks usados e respondem à pergunta? Essas duas perguntas seriam medidas por Faithfulness e Answer Relevancy do RAGAS.

## 2. Métricas usadas

### Retrieval

- **Recall@10:** fração dos chunks de referência recuperados nos dez primeiros resultados. Mede cobertura.
- **MRR:** média do inverso da posição do primeiro chunk relevante. Premia encontrar uma evidência logo no topo.
- **nDCG@10:** qualidade da ordem dos relevantes no top 10 comparada com a ordem ideal.
- **Faithfulness:** proporção de afirmações da resposta apoiadas nos contextos recuperados, estimada pelo juiz RAGAS.
- **Answer Relevancy:** quanto a resposta atende ao que foi perguntado, estimada pelo juiz RAGAS.
- **Chamadas de tools:** quantas vezes o agente pediu execução de ferramentas e quais ferramentas usou; mede interação e complexidade operacional.
- **Custo por resposta:** estimativa baseada em tokens e tabela de preços versionada. É aproximação, não valor faturado.
- **Latência:** tempo observado por execução do agente, incluindo as etapas e as tools registradas.

Os dois benchmarks usam relevância binária nos resultados registrados: cada referência conta como relevante ou não. Isso não distingue um trecho que responde diretamente de outro que apenas oferece contexto.

### Geração e operação do agente

- **Faithfulness:** avalia se as afirmações da resposta são sustentadas pelos contextos recuperados.
- **Answer Relevancy:** avalia se a resposta aborda a pergunta feita.
- **Tool calls:** número total e distribuição das chamadas de ferramentas; chamadas com falha e repetidas também são contadas.
- **Custo por resposta:** estimativa por tokens e preços em `configs/model_pricing.yaml`, incluindo a execução observada. Não é valor faturado.
- **Latência:** tempo registrado para execução do agente, incluindo seus passos e ferramentas.

Métricas de retrieval e de geração medem partes diferentes do sistema. Retrieval pode encontrar a evidência sem o agente usá-la bem; geração pode parecer pertinente sem estar fundamentada. Por isso, os resultados devem ser lidos em conjunto.

### Gate de Recall

Se `drop = baseline_recall - candidate_recall`, a mudança passa quando `drop < 0.01` ou é numericamente igual a `0.01` dentro da tolerância. Uma queda maior que 1 ponto percentual falha. Assim, Recall 0.99 contra baseline 1.00 passa exatamente no limite; Recall 0.9899 falha.

## 3. Resultados de cada experimento

### 3.1 Benchmark V1 — um documento

### Dados e configurações

- Documento: [`docs/sources/ai_engineering.md`](docs/sources/ai_engineering.md) — Cast, AI Engineering Handbook.
- Golden set: 100 queries human-reviewed e aprovadas.
- Chunking: tamanho 100 com overlap 20; tamanho 200 com overlap 40.
- Métricas: Recall@10, MRR e nDCG@10.
- Baseline versionado: `baselines/retrieval_v1.json`, Dense chunk200.

### Retrieval

| Configuração | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| Dense, chunk100 | 0.9741667 | 0.9045000 | 0.8830826 |
| Dense, chunk200 | **1.0000000** | 0.8580397 | 0.8817093 |
| Hybrid, chunk200 | 0.9900000 | 0.9116667 | 0.9208364 |
| Dense + GPT-4.1-mini reranker v2, chunk200 | 0.9900000 | **0.9950000** | **0.9817145** |

Dense chunk200 encontrou todos os chunks relevantes no top 10 e é o baseline de cobertura. O reranker desloca Recall em 0.01, dentro do gate, e melhora substancialmente MRR e nDCG; foi escolhido como recomendação quality-first. Hybrid mantém a queda de Recall em 0.01 e melhora as duas métricas de ranking sem uma chamada de reranker no retrieval. O custo e a latência do reranker não foram comparados neste experimento.

### Gate e regressão deliberada

`test_retrieval.py` carrega o baseline versionado, calcula Recall da configuração candidata e verifica a regra do gate. O teste `retrieve_broken()` retorna lista vazia; a queda de Recall deve fazer o gate falhar. O workflow `.github/workflows/retrieval-eval.yml` executa essa suite no GitHub Actions e usa o segredo OpenAI para recalcular o retrieval.

O gate aceita exatamente 1 ponto percentual de queda. A implementação usa tolerância numérica para evitar que representação em ponto flutuante transforme uma queda de 0.01 em falha por arredondamento.

### Avaliação de geração V1

Uma amostra determinística de 12 perguntas, uma por categoria, foi avaliada com RAGAS 0.3.9 e GPT-4.1-mini como juiz.

| Medida | Resultado |
| --- | ---: |
| Queries selecionadas | 12 |
| Avaliações válidas | 10 |
| Avaliações com erro no avaliador | 2 |
| Faithfulness média das válidas | 1.0000 |
| Answer Relevancy média das válidas | 0.8982 |

Os erros de avaliação não foram transformados em score zero nem incluídos nas médias. Os registros individuais estão em `results/generation_eval_sample12.json`; agregados em `results/generation_eval_summary.json`.

### Análise das cinco queries difíceis do V1

1. **Cobertura / fronteira de chunk.** Query: “Se meu sistema de IA passa nos testes de unidade normais, isso garante que ele funciona bem em produção?”. Ground truth: `chunk200_001` e `chunk200_002`. O resultado recuperou `chunk200_002`, mas deixou `chunk200_001` fora do top 10. Recall 0.5, RR 1.0, nDCG@10 0.6131. A informação útil atravessa a fronteira dos dois chunks; o 002 basta para responder, mas a cobertura formal fica incompleta.
2. **Rótulo binário / relevância desigual.** Query: “Se recall está ok mas as respostas continuam ruins, onde devo investigar primeiro?”. Ground truth: `chunk200_011` e `chunk200_034`; o primeiro resultado foi `chunk200_034`. Recall 0.5, RR 1.0, nDCG@10 0.6131. O chunk034 responde diretamente (prompt e montagem do contexto), enquanto o chunk011 é contexto mais genérico. Ambos contam igualmente como obrigatórios.
3. **Ranking / overlap.** Query: “Como funciona o reranking usando um LLM de propósito geral?”. O único relevante era `chunk200_022`, mas `chunk200_021` veio primeiro e o 022 ficou em segundo. Recall 1.0, RR 0.5, nDCG@10 0.6309. Os chunks têm grande overlap no início da seção de reranking por LLM, causando ambiguidade de ordenação.
4. **Rótulo binário / relevância desigual.** Query: “Qual o papel do golden set em relação a uma feature flag antes de um rollout completo?”. Ground truth: `chunk200_034` e `chunk200_038`; o 038 apareceu primeiro e o 034 ficou em segundo. Recall 1.0, RR 1.0, nDCG@10 0.8175. O chunk038 responde diretamente; o 034 é apenas contexto relacionado. A relevância binária penaliza a ordem como se ambos tivessem igual importância.
5. **Rótulo binário / relevância desigual.** Query: “Por que reranking pode melhorar MRR sem melhorar Recall@K?”. Ground truth: `chunk200_021` e `chunk200_023`; o chunk023 apareceu primeiro e o 021 depois. Recall 1.0, RR 1.0, nDCG@10 0.8503. O chunk023 contém a explicação pedida, enquanto o 021 é contexto geral. O gabarito atual não codifica essa diferença.

Síntese: uma falha foi principalmente de cobertura; uma de ranking sob overlap; três expõem a falta de níveis de relevância. Uma evolução possível é usar graus 0–3 (irrelevante, contexto útil, altamente relevante, resposta direta).

### 3.2 Benchmark multi-documento e estratégias de retrieval

### Corpus, golden set e chunking

Fontes:

1. `docs/sources/ai_engineering.md` — handbook técnico.
2. `docs/sources/pedro_castanheira.md` — perfil profissional adaptado do currículo.
3. `docs/sources/projects.md` — projetos pessoais e seus usos, tecnologias e aprendizados.

O arquivo `datasets/golden_set_multidoc.jsonl` possui 18 perguntas aprovadas: 15 `single_document`, 2 `cross_document` e 1 `multi_hop`. Cada registro contém documentos esperados e trechos de referência. Chunking: 200 palavras, overlap 40; o corpus gerou 48 chunks. A amostra aprovada é útil para protótipo, mas não satisfaz o requisito de 100 queries do desafio original.

### Estratégias

- **Busca geral (Dense):** busca semântica em todos os chunks do corpus; não recebe indicação prévia da fonte.
- **Tool com filtro de documento:** limita a busca a uma das fontes. No benchmark de retrieval isolado, `expected_document_ids` do gabarito escolhe a fonte, portanto essa condição recebe uma pista-oráculo. No agente ajustado à estratégia, o modelo recebe uma tool com IDs/títulos válidos e deve escolher o documento.
- **Expansão por relações (GraphRAG):** usa a busca vetorial semântica para encontrar chunks-semente e expande para trechos ligados por entidades/relações. Seeds e trechos expandidos competem no ranking por posição e distância no grafo.

### Retrieval

| Estratégia | Recall@10 | MRR | nDCG@10 | Δ Recall vs baseline de busca geral |
| --- | ---: | ---: | ---: | ---: |
| Busca geral em todos os documentos | 0.9166667 | 0.7324074 | 0.7557163 | baseline |
| Tool com filtro, fonte indicada pelo gabarito | **0.9722222** | **0.7601852** | **0.7907679** | +5.56 pp |
| Expansão por relações (GraphRAG) | 0.8888889 | 0.7268519 | 0.7510117 | −2.78 pp |

A busca com filtro ficou à frente, mas a comparação somente de retrieval não é simétrica: recebeu a fonte correta. A melhoria de 5.56 pp não deve ser atribuída à filtragem sem ressalva nem projetada no agente. A busca geral foi escolhida como baseline multi-documento neutro em `baselines/multidoc_v1.json`. A expansão por relações alterou o ranking das 18 perguntas após a fusão de scores, mas teve resultado inferior à busca geral nas três métricas. Se aplicada ao baseline de busca geral com gate de 1 pp, falharia nesta amostra.

### Agentes: condição controlada e condição ajustada

Cada linha representa 18 respostas. Na **condição controlada** (`controlled`), o agente mantém o mesmo contrato `retrieve(query, scope, k)` e os mesmos limites; muda o modelo e/ou backend conforme a comparação. Na **condição ajustada** (`tuned`), instruções e tools são especializadas para cada estratégia: `search_all`, `search_by_document` ou `graph_search` e `fetch_source_chunks`. A condição controlada reduz variáveis; a ajustada mede sistemas completos, não o efeito isolado do backend. Cruzar 2 modos × 3 modelos × 3 estratégias forma 18 células experimentais: 324 execuções operacionais no golden set de 18 perguntas e até 72 respostas na amostra RAGAS de quatro perguntas (144 scores, duas métricas por resposta).

#### Condição controlada (`controlled`)

| Estratégia de recuperação | Sucesso | Chamadas médias / resposta | Chamadas por ferramenta | Custo médio / resposta | Latência média | Falhas de tool |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Busca geral | 18/18 | 1.0556 | `retrieve`: 19 | US$ 0.0009902 | 3.769 s | 0 |
| Tool com filtro de documento | 18/18 | 1.0556 | `retrieve`: 19 | US$ 0.0011380 | 3.998 s | 0 |
| Expansão por relações | 18/18 | 1.0556 | `retrieve`: 19 | US$ 0.0009160 | 3.724 s | 0 |

Com interface idêntica e apenas uma tool genérica, o número de chamadas de ferramenta foi igual nas três estratégias. O custo observado da expansão por relações controlada foi menor que os demais nesta amostra, mas isso não supera suas métricas de retrieval inferiores.

#### Condição ajustada (`tuned`)

| Estratégia de recuperação | Sucesso | Chamadas médias / resposta | Chamadas por ferramenta | Custo médio / resposta | Latência média | Falhas de tool |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Busca geral | 18/18 | 1.0556 | `search_all`: 19 | US$ 0.0016893 | 4.101 s | 0 |
| Tool com filtro de documento | 18/18 | 1.0556 | `search_by_document`: 19 | US$ 0.0011454 | 4.080 s | 0 |
| Expansão por relações | 18/18 | 1.7222 | `graph_search`: 18; `fetch_source_chunks`: 13 | US$ 0.0018309 | 4.867 s | 0 |

O schema da tool filtrada explicita IDs e títulos válidos. Após essa correção, não houve falhas de seleção de documento na execução salva. A ferramenta com filtro teve custo estimado e latência inferiores à busca geral e à expansão por relações na condição ajustada. A expansão em grafo usou mais ferramentas, teve maior custo e maior latência nesta amostra.

### 3.3 Modelos do agente: operação nas três estratégias ajustadas

### Comparação operacional dos três modelos

Cada modelo respondeu às mesmas 18 perguntas, com os mesmos três caminhos de retrieval e ferramentas ajustadas. GPT-6 Luna foi executado com esforço de raciocínio baixo. A coluna de chamadas mostra contagem total no conjunto e média por resposta.

| Modelo | Estratégia de recuperação | Sucesso | Chamadas de tools (total; média) | Custo estimado / resposta | Latência média |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-4.1-mini | Busca geral | 18/18 | 19; 1.056 | US$ 0.0016893 | 4.101 s |
| GPT-4.1-mini | Tool com filtro de documento | 18/18 | 19; 1.056 | US$ 0.0011454 | 4.080 s |
| GPT-4.1-mini | Expansão por relações | 18/18 | 31; 1.722 | US$ 0.0018309 | 4.867 s |
| GPT-4o-mini | Busca geral | 18/18 | 18; 1.000 | US$ 0.0004333 | 4.045 s |
| GPT-4o-mini | Tool com filtro de documento | 18/18 | 23; 1.278 | US$ 0.0004869 | 5.314 s |
| GPT-4o-mini | Expansão por relações | 18/18 | 29; 1.611 | US$ 0.0007850 | 5.049 s |
| GPT-6 Luna (baixo) | Busca geral | 18/18 | 20; 1.111 | US$ 0.0004199 | 4.608 s |
| GPT-6 Luna (baixo) | Tool com filtro de documento | 18/18 | 25; 1.389 | **US$ 0.0003476** | 5.905 s |
| GPT-6 Luna (baixo) | Expansão por relações | 18/18 | 40; 2.222 | US$ 0.0005605 | 7.440 s |

As 18/18 respostas sem erro medem conclusão do fluxo, não correção semântica. Em geral, a expansão por relações adicionou chamadas, custo e latência. O modelo mais barato depende da estratégia: Luna na busca filtrada e busca geral, enquanto GPT-4o-mini foi mais barato na condição de expansão por relações. Custos são estimativas do ledger; os artefatos JSON guardam a decomposição por ferramenta.

#### Estado do desenho 2 × 3 × 3

| Condição | Modelos | Estratégias | Estado operacional |
| --- | --- | --- | --- |
| Controlada | GPT-4.1-mini | busca geral, filtro por documento, expansão em grafo | 3 células executadas |
| Controlada | GPT-4o-mini | busca geral, filtro por documento, expansão em grafo | 3 células configuradas, aguardam execução |
| Controlada | GPT-6 Luna (baixo) | busca geral, filtro por documento, expansão em grafo | 3 células configuradas, aguardam execução |
| Ajustada | GPT-4.1-mini, GPT-4o-mini, GPT-6 Luna (baixo) | busca geral, filtro por documento, expansão em grafo | 9 células executadas |

Assim, há métricas operacionais de 12 das 18 células. Os YAMLs [`agent_model_gpt4o_mini_controlled.yaml`](configs/experiments/agent_model_gpt4o_mini_controlled.yaml) e [`agent_model_gpt6_luna_controlled.yaml`](configs/experiments/agent_model_gpt6_luna_controlled.yaml) completam a configuração das seis restantes. Elas ainda precisam ser executadas quando a API estiver acessível.

### 3.4 Qualidade da geração: Faithfulness e Answer Relevancy

A avaliação usa RAGAS 0.3.9 e o mesmo juiz GPT-4.1-mini em todos os casos. Para comparar os modos, modelos e estratégias, a amostra comum contém quatro perguntas (`q001`, `q009`, `q016`, `q018`): inclui perguntas sobre AI Engineering e Pedro, uma pergunta cross-document que envolve projetos e uma pergunta multi-hop sobre projetos. Cada score válido entra na média; falhas do avaliador ficam sem score e são reportadas separadamente, nunca convertidas em zero.

O juiz recebe a pergunta, a resposta e os contextos recuperados. Portanto, estes scores são estimativas automáticas condicionadas à amostra e ao juiz; não substituem revisão humana. O custo do juiz RAGAS é separado do custo operacional por resposta apresentado nas tabelas anteriores. **A matriz de qualidade 2 × 3 × 3 não pôde ser concluída neste ambiente:** a API falhou com `Temporary failure in name resolution`. Existem dois scores individuais cacheados que correspondem a células atuais — GPT-4.1-mini/controlado/busca geral (`q001`: 1.0; 0.9515) e GPT-4.1-mini/ajustado/filtro (`q009`: 1.0; 0.6285). Cada um tem n=1/4 e não representa uma média comparável.

Também corrigi a avaliação: antes, as duas métricas eram aguardadas juntas e um erro em Answer Relevancy podia descartar o score de Faithfulness já calculado. Agora cada métrica é avaliada, armazenada em cache e agregada independentemente. A limitação de DNS ainda impede preencher a matriz localmente; o job de CI faz a rodada em um runner com acesso à API.

| Modo | Modelo respondente | Estratégia de recuperação | Faithfulness (n válido; média) | Answer Relevancy (n válido; média) |
| --- | --- | --- | ---: | ---: |
| Controlado | GPT-4.1-mini | Busca geral | 1.0000 (n=1/4) | 0.9515 (n=1/4) |
| Controlado | GPT-4.1-mini | Tool com filtro de documento | — (0 scores válidos) | — (0 scores válidos) |
| Controlado | GPT-4.1-mini | Expansão por relações | — (0 scores válidos) | — (0 scores válidos) |
| Controlado | GPT-4o-mini | Busca geral | — (0 scores válidos) | — (0 scores válidos) |
| Controlado | GPT-4o-mini | Tool com filtro de documento | — (0 scores válidos) | — (0 scores válidos) |
| Controlado | GPT-4o-mini | Expansão por relações | — (0 scores válidos) | — (0 scores válidos) |
| Controlado | GPT-6 Luna (baixo) | Busca geral | — (0 scores válidos) | — (0 scores válidos) |
| Controlado | GPT-6 Luna (baixo) | Tool com filtro de documento | — (0 scores válidos) | — (0 scores válidos) |
| Controlado | GPT-6 Luna (baixo) | Expansão por relações | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-4.1-mini | Busca geral | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-4.1-mini | Tool com filtro de documento | 1.0000 (n=1/4) | 0.6285 (n=1/4) |
| Ajustado | GPT-4.1-mini | Expansão por relações | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-4o-mini | Busca geral | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-4o-mini | Tool com filtro de documento | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-4o-mini | Expansão por relações | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-6 Luna (baixo) | Busca geral | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-6 Luna (baixo) | Tool com filtro de documento | — (0 scores válidos) | — (0 scores válidos) |
| Ajustado | GPT-6 Luna (baixo) | Expansão por relações | — (0 scores válidos) | — (0 scores válidos) |

O travessão indica ausência de score válido, não score zero. Os dois scores apresentados são diagnósticos de uma pergunta cada (n=1/4); não devem ser tratados como resultados médios das células.

As métricas de operação acima não substituem essa avaliação de qualidade. A avaliação V1, em um documento, foi uma amostra determinística de 12 perguntas: 10 scores válidos, 2 erros do avaliador, Faithfulness média 1.0000 e Answer Relevancy média 0.8982.

## 4. Conclusões gerais

### Um documento

Recomendação quality-first: Dense chunk200 + GPT reranker v2, pois oferece os melhores MRR/nDCG e sua queda de Recall é exatamente o limite aceito. Hybrid chunk200 é a opção sem chamada extra de LLM no retrieval. Dense chunk200 é a referência se Recall máximo e simplicidade forem prioritários.

### Vários documentos

Manter a tool com filtro de documento como candidata para a próxima rodada: o agente recebe IDs/títulos válidos, escolhe a fonte e pode consultar outra. Isso deve ser comparado com retrieval normal como baseline. O resultado retrieval-only filtrado usa a fonte esperada do gabarito como pista-oráculo; não prova que o agente escolhe sempre a fonte correta. A recomendação final de qualidade depende da matriz RAGAS comum.

Não adotar expansão por relações em grafo como padrão no estado atual: ficou abaixo da busca geral no retrieval desta amostra e aumentou chamadas, custo e latência nas três execuções de agente ajustado. Manter busca geral como baseline sem informação prévia da fonte.

### Modelo

Não eleger vencedor de qualidade apenas por custo ou latência. A matriz RAGAS das 18 combinações ainda precisa ser concluída quando o DNS/API estiver disponível; por isso, ainda não há base para recomendar um modelo por qualidade. O melhor modelo pode mudar conforme o modo e a ferramenta: contagem de chamadas, custo e latência não determinam se a resposta está fundamentada ou responde corretamente.

O experimento mostra por que se deve medir o sistema em partes. Recall/MRR/nDCG avaliam recuperação e ordenação de evidências; Faithfulness e Answer Relevancy avaliam a resposta; traces, custo e latência mostram o comportamento operacional. Uma mudança pode melhorar uma dimensão e piorar outra. O gate torna esse tradeoff visível e impede regressões de cobertura acima do limite definido, sem declarar automaticamente que uma métrica isolada representa qualidade total.

### Estado do gate e limitações

### Concluído

- V1 com 100 queries aprovadas por revisão humana.
- V1 mediu chunk size, Dense, Hybrid e reranker; recomendações e análise das cinco queries difíceis estão em `results/recommendation.md`.
- Gate V1 falha acima de 1 pp, passa exatamente em 1 pp e inclui um retriever deliberadamente quebrado para verificar a detecção.
- Geração V1: amostra de 12, 10 scores válidos e 2 erros tratados sem zeros artificiais.
- Novo corpus multi-documento, 18 queries aprovadas, busca geral, tool com filtro e expansão por relações, além das condições controlada e ajustada.
- 12 células operacionais concluídas: GPT-4.1-mini controlado e os três modelos ajustados, cada qual com as três estratégias.
- Duas configurações YAML adicionadas para executar as seis células controladas ainda ausentes.
- Baseline multi-documento de busca geral versionado.

### Pendente para considerar o novo benchmark completo

1. Executar as seis células controladas de GPT-4o-mini e GPT-6 Luna; hoje há resultados operacionais em 12/18 células.
2. Completar a matriz RAGAS das 18 combinações quando o ambiente voltar a resolver/conectar à API OpenAI; os dois scores cacheados têm n=1/4 e são insuficientes para comparar qualidade dos modelos.
3. A amostra de geração tem quatro perguntas; mesmo concluída, fornece comparação inicial, não estimativa estável de produção.
4. O golden set V1 atende às 100 perguntas revisadas. O novo conjunto multi-documento tem somente 18; ampliá-lo se as 100 também forem exigidas para essa fase.
5. Relevância de retrieval é binária e o novo corpus contém só duas perguntas cross-document e uma multi-hop. Uma amostra maior e rótulos graduados permitiriam avaliar melhor contexto parcial, escolha de fonte e raciocínio entre documentos.
6. O workflow multi-documento agora inclui validação offline e um job de comparação com API: recalcula o retrieval, roda as 18 células, confere a completude de Faithfulness e Answer Relevancy e aplica o gate contra `baselines/multidoc_v1.json`. O job requer o segredo `OPENAI_API_KEY`; pull requests de forks executam apenas a validação offline.
7. Custos vêm de tokens e tabela versionada, não da fatura. Latência é desta execução e ambiente; não representa SLO de produção nem análise de significância estatística.

## 5. Como reproduzir

Instale as dependências a partir da raiz do repositório. Comandos de validação não colhem métricas nem chamam modelos; os comandos de avaliação exigem `OPENAI_API_KEY` e podem gerar custos de API.

```bash
python -m pip install -e ".[dev,evaluation]"
python -m pytest tests -q
python -m groundtruth.cli validate-dataset --config configs/experiments/retrieval_controlled.yaml
python -m groundtruth.cli evaluate-retrieval --config configs/experiments/retrieval_controlled.yaml
python -m groundtruth.cli evaluate-agents --config configs/experiments/agent_controlled.yaml
python -m groundtruth.cli evaluate-agents --config configs/experiments/agent_tuned.yaml
python -m groundtruth.cli evaluate-agents --config configs/experiments/agent_model_gpt4o_mini_controlled.yaml
python -m groundtruth.cli evaluate-agents --config configs/experiments/agent_model_gpt6_luna_controlled.yaml
python -m groundtruth.cli evaluate-agents --config configs/experiments/agent_model_gpt4o_mini.yaml
python -m groundtruth.cli evaluate-agents --config configs/experiments/agent_model_gpt6_luna.yaml
```

Por padrão, `evaluate-agents` também avalia geração RAGAS nas quatro perguntas configuradas. Para repetir somente a comparação operacional, acrescente `--without-generation-eval`. Depois de gerar as seis configurações, `python -m groundtruth.cli compare-agents` valida a matriz completa, compara Recall@10 com o baseline e grava `results/multidoc/agent_matrix.json` e `.md`. O workflow do GitHub Actions executa esse processo no push e em PRs do mesmo repositório, exige `OPENAI_API_KEY` e publica os resultados como artefato. Configure esse segredo no repositório. O cache é restaurado/salvo pelo workflow e reduz chamadas em execuções com código, corpus e configurações iguais; mudanças nesses insumos geram uma nova rodada de API. O V1 tem seu workflow de retrieval separado, incluindo a demonstração com retriever deliberadamente quebrado. O guia dos módulos está em [`ARCHITECTURE.md`](ARCHITECTURE.md).
