# Plano do experimento: retrieval e agentes sobre três fontes

## Objetivo

Comparar como três estratégias de retrieval respondem perguntas sobre fontes de naturezas diferentes:

1. AI engineering;
2. Pedro Castanheira;
3. Projetos, tecnologias e aprendizados descritos no currículo de Pedro.

A hipótese principal é que relações em grafo ajudam mais em perguntas difíceis que exigem evidência de mais de uma fonte. Filtros podem ajudar quando o escopo da pergunta é claro. A busca sem filtro funciona como baseline simples.

## Braços de retrieval

- **Flat:** todos os chunks no mesmo índice, sem filtro de documento.
- **Filtered:** aplica filtro de documento ou tema antes de ranquear os chunks.
- **Graph:** encontra entidades e relações entre fontes, expande para chunks de evidência e retorna esses chunks para a resposta.

## Tipos e dificuldade das perguntas

O perfil e os projetos são baseados no currículo fornecido. O golden set deve marcar cada pergunta com categoria, dificuldade, fontes esperadas, chunks relevantes e se exige cruzamento entre fontes. Os grupos iniciais são perguntas de uma fonte, cross-document e multi-hop. As mesmas perguntas são executadas nos três braços.

## Rodadas com agente

### Agente controlado

Modelo, prompt, orçamento de chamadas e interface de tools permanecem iguais. Só muda o backend de retrieval. Esta rodada permite atribuir diferenças ao retrieval com mais confiança.

### Agente ajustado

Cada estratégia pode ter tools e instruções próprias. Esta rodada compara sistemas completos; os resultados não isolam o efeito do retrieval.

## Métricas

Retrieval: Recall@10, MRR e nDCG@10, no total e separados por tipo e dificuldade da pergunta.

Agente: tools chamadas e frequência por tool, chamadas repetidas ou com erro, custo por resposta, latência, Faithfulness e Answer Relevancy. Custo do avaliador fica separado do custo de responder.

## Organização do cache

`../../src/groundtruth` será a camada compartilhada de armazenamento, hashing e leitura/escrita atômica. Cada módulo será responsável por montar sua chave com todos os parâmetros que alteram a saída; assim, a camada comum não precisa conhecer detalhes de embedding, grafo ou agente.

Namespaces planejados:

- `cache/text_embeddings/`: texto normalizado e modelo de embedding.
- `cache/reranker/`: query, IDs e ordem dos candidatos, `k`, modelo e versão do prompt.
- `cache/knowledge_graph/`: fingerprint das fontes, versão do chunking, modelo extrator e versão do prompt.
- `cache/agent_runs/`: ID da pergunta, backend e fingerprint do índice, modelo, versão do prompt, versão das tools e limites de execução.

Os caches atuais de embeddings e reranking devem continuar válidos após a migração, preservando seus caminhos e formato de chave. Resultados finais dos experimentos continuam em `results/`, separados de artefatos intermediários em `cache/`. O custo do avaliador não entra no custo cacheado da resposta do agente.

## Estado

Este arquivo registra o desenho do experimento. `docs/sources/pedro_castanheira.md` e `docs/sources/projects.md` foram redigidos com informações do currículo fornecido. Os módulos em `../../src/groundtruth` continuam placeholders vazios; filtros, grafo e agente ainda serão implementados passo a passo.
