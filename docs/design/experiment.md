# Plano do experimento: retrieval e agentes sobre três fontes

## Objetivo

Comparar como três estratégias de retrieval respondem perguntas sobre fontes de naturezas diferentes:

1. AI engineering;
2. Pedro Castanheira;
3. Projetos, tecnologias e aprendizados descritos no currículo de Pedro.

A hipótese principal é que relações em grafo ajudam mais em perguntas difíceis que exigem evidência de mais de uma fonte. Filtros podem ajudar quando o escopo da pergunta é claro. A busca sem filtro funciona como baseline simples.

## Estratégias de retrieval

- **Busca geral:** todos os chunks no mesmo índice, sem filtro de documento.
- **Tool com filtro de documento:** aplica o filtro escolhido pelo agente antes de ranquear os chunks daquele documento.
- **Expansão por relações em grafo (GraphRAG):** encontra entidades e relações entre fontes, expande para chunks de evidência e retorna esses chunks para a resposta.

## Tipos e dificuldade das perguntas

O perfil e os projetos são baseados no currículo fornecido. O golden set marca cada pergunta com categoria, dificuldade, fontes esperadas, chunks relevantes e se exige cruzamento entre fontes. Os grupos são perguntas de uma fonte, cross-document e multi-hop. As mesmas perguntas são executadas nos três braços e nos dois modos de agente. No retrieval filtrado sem agente, os documentos esperados pelo gabarito funcionam como pista-oráculo; isso mede a busca depois da escolha de fonte, não a capacidade de descobrir a fonte.

## Rodadas com agente

### Agente controlado

Modelo, prompt, orçamento de chamadas e interface de tools permanecem iguais. Só muda o backend de retrieval. Esta rodada permite atribuir diferenças ao retrieval com mais confiança.

### Agente ajustado

Cada estratégia pode ter tools e instruções próprias. Esta rodada compara sistemas completos; os resultados não isolam o efeito do retrieval.

### Cruzamento de fatores

O desenho completo cruza 2 modos (controlado/ajustado), 3 modelos (GPT-4.1-mini, GPT-4o-mini, GPT-6 Luna com esforço baixo) e 3 estratégias de retrieval, totalizando 18 células. Isso corresponde a 324 respostas operacionais (18 perguntas × 18 células) e até 72 respostas na amostra RAGAS (4 perguntas × 18 células), com duas métricas por resposta. Todas as comparações usam as mesmas perguntas dentro de cada tipo de avaliação.

## Métricas

Retrieval: Recall@10, MRR e nDCG@10, no total e separados por tipo e dificuldade da pergunta.

Agente: tools chamadas e frequência por tool, chamadas repetidas ou com erro, custo por resposta, latência, Faithfulness e Answer Relevancy. Custo do avaliador fica separado do custo de responder.

## Organização do cache

`../../src/groundtruth` implementa a camada compartilhada de armazenamento, hashing e leitura/escrita atômica. Cada módulo monta sua chave com os parâmetros que alteram a saída; a camada comum não precisa conhecer detalhes de embedding, grafo ou agente.

Namespaces planejados:

- `cache/text_embeddings/`: texto normalizado e modelo de embedding.
- `cache/reranker/`: query, IDs e ordem dos candidatos, `k`, modelo e versão do prompt.
- `cache/knowledge_graph/`: fingerprint das fontes, versão do chunking, modelo extrator e versão do prompt.
- `cache/agent_runs/`: ID da pergunta, backend e fingerprint do índice, modelo, versão do prompt, versão das tools e limites de execução.

Os caminhos e o formato da chave de embeddings foram preservados para compatibilidade com o cache anterior. Resultados finais ficam em `results/`, separados de artefatos intermediários em `cache/`. O custo do avaliador RAGAS não entra no custo operacional da resposta do agente.

## Estado atual e limites

O fluxo está implementado em `../../src/groundtruth`. O golden set multi-documento tem 18 perguntas aprovadas. Os resultados operacionais existem em 12 das 18 células: GPT-4.1-mini no modo controlado e os três modelos no modo ajustado, todos nas três estratégias. Há YAMLs configurados para as seis células controladas restantes. A avaliação RAGAS da matriz não pôde ser concluída neste ambiente devido a `Temporary failure in name resolution` ao acessar a API OpenAI; os poucos scores individuais em cache não permitem comparar as células. O baseline de busca geral está versionado em `../../baselines/multidoc_v1.json`.

As métricas de retrieval/agente são resultados do conjunto aprovado atual, não generalizam por si só para produção. O baseline antigo é exclusivo do handbook e não deve ser reaproveitado no corpus multi-documento.
