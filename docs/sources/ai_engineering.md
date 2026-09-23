# Manual de Engenharia de IA

Este manual documenta como projetar, desenvolver, avaliar, colocar em produção e operar aplicações baseadas em LLMs. Ele é a referência interna usada por engenheiros que constroem qualquer sistema de IA na empresa, incluindo os pipelines de RAG (Retrieval-Augmented Generation) que sustentam produtos internos e externos.

## Fundamentos

### O que é Engenharia de IA

Engenharia de IA é a disciplina de construir sistemas de produção que usam modelos de linguagem como um componente entre vários outros: retrieval, ferramentas, validação de saída, cache e observabilidade. Diferente de pesquisa em IA, que otimiza a qualidade do modelo em si, Engenharia de IA otimiza o sistema completo para confiabilidade, custo e latência em condições reais de uso.

### Engenharia de IA vs Engenharia de Software tradicional

Na engenharia de software tradicional, o mesmo input produz o mesmo output e os testes verificam igualdade exata. Em Engenharia de IA, o mesmo input pode produzir respostas diferentes, então os testes precisam avaliar propriedades (a resposta cita a fonte correta? é fiel ao contexto?) em vez de comparar strings exatas. Por isso, avaliação deve ser tratada como parte do desenvolvimento, e não como uma etapa final opcional.

### Natureza probabilística de sistemas com LLMs

Um LLM não tem uma função determinística entrada-saída: ele amostra tokens de uma distribuição de probabilidade condicionada no contexto. Isso significa que pequenas variações no prompt, na ordem dos documentos recuperados ou na temperatura podem mudar o resultado final. Qualquer métrica de qualidade deve ser tratada como uma média sobre várias execuções, nunca como o resultado de uma única chamada.

### Pipeline de IA (visão geral)

Um pipeline de IA típico tem estágios independentes e testáveis: ingestão de dados, chunking, indexação, retrieval, reranking opcional, montagem de contexto, geração e validação de saída. Cada estágio pode ser avaliado isoladamente, o que permite identificar exatamente onde uma regressão de qualidade aconteceu em vez de culpar "o modelo" de forma genérica.

### Modelo vs Sistema

O modelo é apenas um dos componentes do sistema, e trocar de modelo raramente resolve um problema causado por outro estágio do pipeline. Por exemplo, se o retrieval não traz o passage correto, nenhum modelo de geração — por melhor que seja — vai produzir uma resposta correta. Qualquer investigação de qualidade deve começar identificando se o problema está no sistema (retrieval, contexto, ferramentas) antes de trocar o modelo de geração.

### Inferência e o que acontece em uma chamada de modelo

Inferência é o processo de gerar uma saída a partir de um modelo já treinado, token a token, cada novo token condicionado nos anteriores. Latência de inferência cresce com o número de tokens gerados, não apenas com o tamanho do prompt de entrada, o que é uma causa comum de latência alta em respostas longas.

### Contexto, janela de contexto e tokens

Tokens são as unidades de texto que o modelo processa, e a janela de contexto é o número máximo de tokens que cabem em uma chamada, somando prompt de entrada e saída gerada. Quando o contexto excede esse limite, é necessário reduzir o número de passages recuperados ou resumir o histórico da conversa antes de enviar a chamada; estouro de contexto deve ser tratado como erro de sistema, não como algo para o modelo "resolver sozinho".

### Temperatura e structured outputs

Temperatura controla o quanto a distribuição de probabilidade dos próximos tokens é achatada ou concentrada: valores baixos (perto de 0) tornam a saída mais determinística, valores altos aumentam a diversidade. Para tarefas que alimentam outros sistemas (como extração de campos ou reranking automatizado), recomenda-se temperatura baixa combinada com structured outputs (JSON Schema ou formato equivalente), porque isso reduz variação de formatação e evita falhas de parsing no código que consome a resposta.

## Prompting

### System prompt vs user prompt

O system prompt define o papel, as regras e as restrições permanentes do assistente, enquanto o user prompt carrega a pergunta ou tarefa específica daquela chamada. Como regra geral, regras de negócio e formato de saída sempre vão no system prompt; misturar essas regras dentro do user prompt dificulta reaproveitar o mesmo prompt de sistema entre diferentes perguntas.

### Few-shot prompting

Few-shot prompting é a técnica de incluir exemplos de entrada e saída dentro do prompt para guiar o formato ou o estilo da resposta, sem precisar treinar o modelo. É particularmente útil quando o formato de saída é incomum ou quando instruções em texto puro não são suficientes para eliminar ambiguidade; usam-se poucos exemplos (normalmente de dois a quatro) para não consumir contexto desnecessário.

### Instruções claras e delimitação de contexto

Instruções vagas ("responda bem", "seja preciso") geram comportamento inconsistente porque não restringem a distribuição de respostas possíveis. Recomenda-se instruções específicas sobre formato, extensão e o que fazer quando a informação não está disponível, além de delimitar claramente onde o contexto recuperado começa e termina (por exemplo, com marcadores como `<contexto>` e `</contexto>`) para que o modelo não confunda instruções do usuário com conteúdo recuperado.

### Prompt injection na perspectiva de prompting

Prompt injection ocorre quando texto dentro do conteúdo processado (um documento recuperado, um e-mail, uma página web) contém instruções que tentam sobrescrever o comportamento definido no system prompt. Do ponto de vista de design de prompt, a mitigação básica é nunca tratar conteúdo recuperado como instrução: o prompt deve deixar explícito que qualquer texto dentro dos delimitadores de contexto é dado a ser analisado, não comando a ser seguido.

### Versionamento de prompts

Todo prompt em produção possui uma versão explícita (v1, v2, v3...) e um changelog associado. Um prompt deve receber uma nova versão sempre que a mudança altera o comportamento esperado: mudança de instruções, de exemplos few-shot ou do schema de saída conta como nova versão; correções de digitação que não mudam o significado não contam. Essa regra existe para que resultados de avaliação sejam sempre comparáveis a uma versão específica do prompt, e não a "o prompt atual", que muda com o tempo.

### Quando prompting não resolve o problema

Se o modelo erra porque não tem a informação necessária no contexto, nenhum ajuste de instrução vai corrigir isso — o problema é de retrieval ou de dados, não de prompt. Da mesma forma, se a tarefa exige uma ação determinística (como um cálculo exato ou uma validação de regra de negócio), recomenda-se implementar essa lógica em código e não depender do modelo para executá-la corretamente todas as vezes.

### Delimitadores e formatação para evitar ambiguidade

Usar delimitadores consistentes (tags XML, marcadores de bloco, ou JSON) para separar instrução, exemplos e contexto reduz a chance de o modelo interpretar mal os limites entre essas partes. Esses delimitadores devem ser padronizados por prompt versionado, de forma que uma mudança de formatação de delimitador também é tratada como uma nova versão do prompt.

## Embeddings

### O que são embeddings

Embeddings são representações numéricas (vetores) de um texto, geradas de forma que textos com significado semelhante fiquem próximos no espaço vetorial. Eles permitem comparar o significado de duas frases sem que compartilhem as mesmas palavras, o que é a base técnica que torna a busca semântica possível.

### Similaridade vetorial e cosine similarity

Cosine similarity mede o ângulo entre dois vetores de embedding, ignorando sua magnitude, e retorna um valor entre -1 e 1 (na prática, entre 0 e 1 para a maioria dos embeddings de texto). Quanto mais próximo de 1, mais semanticamente próximos os textos são considerados; cosine similarity é usada como métrica padrão de comparação entre embeddings de query e de chunk no retrieval denso.

### Escolha de embedding model

A escolha do embedding model afeta diretamente a qualidade do retrieval denso, então mudanças nesse componente são tratadas como uma decisão de arquitetura, não como um parâmetro qualquer. Um novo embedding model deve ser avaliado rodando o golden set completo contra o índice gerado com o modelo novo antes de substituir o modelo em produção.

### Dimensionalidade dos vetores

A dimensionalidade é o número de valores em cada vetor de embedding, e não existe uma relação direta de "mais dimensões, melhor qualidade": dimensões maiores aumentam custo de armazenamento e tempo de busca, mas nem sempre melhoram recall. A dimensionalidade é escolhida com base em testes de recall no golden set, não apenas no tamanho do vetor oferecido pelo provedor do modelo.

### Por que trocar de embedding model exige reindexação

Vetores gerados por modelos de embedding diferentes não são comparáveis entre si, porque cada modelo aprende seu próprio espaço vetorial. Por isso, sempre que o embedding model muda, é necessária reindexação completa do corpus: todo chunk precisa ser reprocessado com o novo modelo antes de qualquer busca ser feita, já que misturar vetores antigos e novos no mesmo índice produz resultados de similaridade sem sentido.

### Atualização incremental vs reindexação completa

Adicionar novos documentos a um índice existente (mesmo embedding model) é uma atualização incremental e não exige reprocessar os chunks já indexados. Já qualquer mudança que afete como o vetor é calculado — troca de modelo, mudança de estratégia de chunking, ou mudança de normalização do texto antes da geração do embedding — exige reindexação completa, porque parte do índice ficaria inconsistente com o resto.

### Busca lexical vs busca semântica

Busca lexical (baseada em palavras-chave, como BM25) encontra documentos que compartilham termos exatos ou variações próximas com a query, e falha quando a pergunta usa sinônimos ou paráfrases. Busca semântica, baseada em embeddings, encontra documentos com significado parecido mesmo sem sobreposição de palavras, mas pode falhar em buscas que dependem de termos exatos, como identificadores, códigos de erro ou nomes próprios raros — por isso normalmente combinam-se as duas abordagens em retrieval híbrido.

## RAG

### Arquitetura RAG (visão geral do pipeline)

O pipeline de RAG segue os estágios: ingestão de documentos, chunking, geração de embeddings e indexação, retrieval no momento da pergunta, reranking opcional dos candidatos, montagem do contexto final e geração da resposta com o modelo de linguagem. Cada estágio expõe métricas próprias, permitindo isolar regressões sem precisar reavaliar o pipeline inteiro a cada mudança.

### Ingestão de documentos

Ingestão é o processo de trazer documentos de fontes originais (wikis internas, PDFs, tickets, código) para um formato normalizado antes do chunking. Recomenda-se preservar metadados da fonte durante a ingestão (título, seção, data de atualização), porque esses metadados são usados depois para filtros de retrieval e para exibir citações confiáveis ao usuário final.

### Retrieval dentro do pipeline RAG

Retrieval é o estágio responsável por selecionar, de todo o corpus indexado, o subconjunto de passages mais relevantes para a pergunta do usuário. A qualidade do retrieval é o teto de qualidade de todo o pipeline RAG: se o passage correto nunca é recuperado, a etapa de geração não tem como produzir uma resposta correta, independentemente de quão bom seja o modelo usado.

### Context assembly (montagem do contexto)

Context assembly é a etapa que decide como os passages recuperados (e, se houver, rerankeados) são organizados dentro do prompt enviado ao modelo de geração: ordem, formatação, marcação de fonte de cada trecho e corte pelo limite de tokens disponível. Recomenda-se manter a ordem de relevância (mais relevante primeiro) e sempre incluir o identificador de cada passage junto ao seu texto, para viabilizar citações rastreáveis na resposta final.

### Geração da resposta

Na etapa de geração, o modelo de linguagem recebe a pergunta do usuário junto com o contexto montado e produz a resposta final. A qualidade dessa etapa depende do prompt de geração ser explícito sobre o que fazer quando o contexto não contém a resposta: o prompt deve instruir o modelo a admitir a ausência de informação em vez de inferir uma resposta plausível não sustentada pelo contexto.

### Citações e rastreabilidade

Citações são referências explícitas, na resposta gerada, aos passages do corpus que sustentam cada afirmação. Citação deve ser tratada como requisito de produto, não apenas de auditoria: toda resposta gerada por um sistema RAG voltado a usuários finais deve indicar de qual passage (ou passages) cada afirmação relevante veio, permitindo verificação humana rápida.

### Grounding

Grounding é a propriedade de uma resposta estar apoiada apenas em informações presentes no contexto fornecido, em vez de vir do conhecimento paramétrico geral do modelo. Uma resposta pode estar gramaticalmente correta e até factualmente certa e ainda assim não estar "grounded", se a informação não veio do contexto recuperado — essa distinção é exatamente o que a métrica de faithfulness tenta capturar na avaliação.

### Vantagens e limitações do RAG

RAG permite que um sistema responda com base em informação atualizada ou proprietária sem precisar retreinar o modelo, e reduz (mas não elimina) alucinação ao ancorar a resposta em documentos reais. As limitações aparecem quando o retrieval falha (documento certo não é encontrado), quando o corpus está desatualizado, ou quando a pergunta exige raciocínio sobre múltiplos documentos que nunca aparecem juntos no candidate set — RAG não substitui a necessidade de um bom retrieval.

## Chunking

### Chunk size recomendado

Recomenda-se chunks entre 200 e 500 tokens como ponto de partida para a maioria dos documentos técnicos, ajustando esse intervalo conforme os resultados de Recall@K no golden set. Chunks menores que esse intervalo tendem a fragmentar ideias que precisam de mais contexto para fazer sentido; chunks maiores tendem a diluir a relevância de um chunk que contém a resposta em meio a conteúdo não relacionado.

### Overlap entre chunks

Overlap é a sobreposição de texto entre chunks consecutivos, usada para evitar que uma informação relevante fique cortada exatamente na fronteira entre dois chunks. Usa-se overlap entre 10% e 15% do chunk size como padrão; overlap muito maior que isso aumenta redundância no índice sem ganho proporcional de recall.

### Chunking semântico

Chunking semântico divide o texto em pontos onde o significado muda (por exemplo, mudança de tópico detectada por similaridade entre sentenças consecutivas), em vez de cortar por número fixo de caracteres ou tokens. Essa abordagem tende a gerar chunks mais coerentes, mas exige processamento adicional na ingestão e pode gerar chunks de tamanho bastante variável.

### Chunking baseado em estrutura (headings)

Chunking baseado em estrutura usa a hierarquia do próprio documento (títulos, subtítulos, listas) para definir os limites dos chunks, mantendo cada chunk alinhado a uma seção lógica do documento original. Esse método é priorizado sempre que a fonte tem estrutura clara (como este próprio manual), porque produz chunks que já correspondem a unidades de conhecimento pensadas por quem escreveu o documento.

### Chunks grandes vs pequenos — trade-off

Chunks pequenos aumentam precisão (o passage recuperado tende a ser mais focado), mas podem perder contexto necessário para responder perguntas mais amplas e aumentam o número total de chunks no índice. Chunks grandes preservam mais contexto por chunk, mas reduzem precisão porque um único chunk pode misturar informação relevante e irrelevante para uma dada query, dificultando tanto o retrieval quanto o reranking.

### Impacto do chunking no retrieval

A estratégia de chunking afeta diretamente que perguntas o retrieval consegue responder bem: se uma regra e sua exceção ficam em chunks diferentes, uma query sobre a exceção pode recuperar apenas a regra geral e produzir uma resposta incompleta ou enganosa. Por isso, a estratégia de chunking deve ser revisada sempre que identifica, na avaliação, uma categoria inteira de perguntas com recall sistematicamente baixo.

### Preservação de contexto dentro de um chunk

Um bom chunk deve fazer sentido quando lido isoladamente, sem depender de frases anteriores ou posteriores que não fazem parte dele. Isso é avaliado pedindo que revisores humanos leiam um chunk sem ver o restante do documento e verifiquem se conseguem entender a ideia principal sem ambiguidade — chunks que falham nesse teste são candidatos a serem reescritos ou remarcados nos limites de chunking.

## Retrieval

### Dense retrieval

Dense retrieval usa embeddings para comparar a query e os chunks no espaço vetorial, tipicamente via cosine similarity, retornando os k chunks mais próximos da query. Ele captura relação semântica mesmo sem sobreposição de palavras, mas depende inteiramente da qualidade do embedding model usado para gerar os vetores.

### Keyword retrieval e BM25

Keyword retrieval encontra documentos com base em correspondência de termos, e BM25 é o algoritmo de referência dessa abordagem, ponderando termos raros mais fortemente que termos comuns. Esse tipo de busca continua relevante mesmo em sistemas modernos porque resolve bem casos que embeddings tratam mal, como busca por códigos, siglas ou identificadores exatos.

### Hybrid retrieval (peso alpha)

Retrieval híbrido combina o score denso (semântico) e o score léxico (baseado em palavras-chave) em uma única pontuação, geralmente por meio de uma média ponderada controlada por um parâmetro alpha. Usa-se alpha = 0,5 como padrão, dando peso igual às duas abordagens, e ajusta esse valor experimentalmente quando um domínio específico se beneficia mais de busca exata (alpha menor) ou mais de busca semântica (alpha maior).

### Top-k: o que é e como começar

Top-k é o número de passages retornados pelo estágio de retrieval para uma dada query. Recomenda-se começar qualquer experimento novo de retrieval com top_k = 10, porque esse valor é grande o suficiente para capturar a maioria dos passages relevantes em corpora de tamanho médio e pequeno o suficiente para manter o contexto final enxuto.

### Aumentar top-k nem sempre melhora recall

Aumentar top_k só melhora Recall@K se o passage relevante realmente existir no corpus e estiver perto o bastante do topo do ranking para caber no novo k; se o retriever simplesmente não atribui um score alto ao passage correto, aumentar k pode não trazê-lo de forma alguma, e ainda assim aumenta o custo de contexto e a chance de introduzir ruído na geração. Por isso, "aumentar top-k" deve ser tratado como mitigação temporária, não como correção de um retriever com problema estrutural.

### Filtros de metadata no retrieval

Filtros de metadata restringem o retrieval a um subconjunto de chunks com base em atributos como categoria, data ou fonte, aplicados antes ou depois do cálculo de similaridade. Recomenda-se aplicar filtros de metadata antes do cálculo de similaridade sempre que possível, porque isso reduz o espaço de busca e evita que um chunk irrelevante, mas com alta similaridade textual, ocupe uma vaga no top-k.

### Recall vs precisão

Recall mede a proporção de passages relevantes que foram de fato recuperados; precisão mede a proporção de passages recuperados que são de fato relevantes. Um sistema pode ter recall alto e precisão baixa (recupera tudo que importa, mas também muito lixo) ou o contrário, e otimizar apenas uma dessas métricas sem olhar a outra costuma degradar a experiência final do usuário.

### Candidatos (candidate set) antes do reranking

O candidate set é o conjunto de passages retornado pelo retrieval inicial (denso, léxico ou híbrido) antes de qualquer reranking ser aplicado. Ele geralmente é maior que o número final de passages usados na geração, justamente para dar ao reranker margem para reordenar e escolher os melhores entre um grupo mais amplo de candidatos plausíveis.

### Quando dense retrieval falha

Dense retrieval tende a falhar em queries muito curtas ou ambíguas, em buscas por termos exatos (nomes de erro, códigos, identificadores) e em corpora pequenos onde a distribuição de embeddings fica pouco discriminativa entre chunks diferentes. Nesses casos, recomenda-se testar retrieval híbrido antes de assumir que o problema está no embedding model.

## Reranking

### O que é reranking e diferença para retrieval

Retrieval é o estágio que reduz todo o corpus a um conjunto pequeno de candidatos plausíveis; reranking é o estágio seguinte, que reordena esses candidatos com um modelo mais caro e mais preciso, otimizado para comparar poucos itens entre si em vez de buscar entre milhares. Retrieval prioriza escala e velocidade; reranking prioriza precisão na ordenação final.

### Quando vale usar reranker

Um reranker compensa quando o candidate set inicial contém o passage correto, mas não no topo do ranking — nesse caso, reordenar melhora a posição do passage certo sem precisar mudar o retrieval em si. Quando o passage correto nem aparece no candidate set, nenhum reranker resolve o problema, porque reranking só reordena o que já foi recuperado.

### Cross-encoders

Cross-encoders são modelos que recebem a query e um candidato juntos, como uma única entrada, e produzem diretamente um score de relevância entre os dois. Eles costumam ser mais precisos que a comparação de embeddings separados (bi-encoders), porque permitem atenção cruzada entre query e passage, mas são mais caros computacionalmente porque precisam rodar uma inferência por par query-candidato.

### Reranking por LLM (implementação comum)

Também é comum usar um LLM de propósito geral como reranker, enviando a query e a lista de candidatos e pedindo que o modelo retorne os IDs ordenados por relevância. Essa abordagem é mais flexível que um cross-encoder dedicado (não exige treinar um modelo específico), mas tem latência e custo maiores por chamada, então é usada tipicamente só na etapa final, sobre um candidate set já reduzido.

### Custo e latência do reranker

Reranking adiciona uma chamada de modelo extra ao caminho crítico da resposta, então seu custo e latência devem ser avaliados junto com o ganho de qualidade que ele proporciona. Um reranker só deve ser mantido em produção se ele produzir uma melhora mensurável de MRR ou nDCG no golden set que justifique a latência adicional; caso contrário, a etapa é removida do pipeline.

### Tamanho do candidate set enviado ao reranker

O reranker normalmente recebe entre 20 e 50 candidatos vindos do retrieval inicial. Um candidate set menor que isso corre o risco de já ter descartado o passage correto antes mesmo do reranking; um candidate set muito maior que 50 aumenta custo e latência sem ganho proporcional, porque a maior parte dos candidatos adicionais tem relevância muito baixa.

### Reranker pode melhorar MRR sem melhorar Recall

Recall@K mede apenas se o passage relevante está entre os k primeiros, não a posição exata dele dentro desse grupo; reranking pode mover o passage correto de, por exemplo, a quinta posição para a primeira sem mudar se ele está ou não dentro do top-k. Por isso é normal observar MRR subir depois de adicionar um reranker enquanto Recall@K permanece igual — isso não significa que o reranker "não fez nada", significa que ele melhorou a ordenação dentro de um conjunto que já continha o passage certo.

## Avaliação

### Golden datasets — o que são e como são construídos

Um golden dataset é um conjunto de perguntas com respostas de referência (aqui, passages relevantes conhecidos) usado para medir objetivamente a qualidade de um sistema de retrieval ou geração. O golden set é construído gerando candidatos automaticamente a partir do corpus e depois submetendo cada candidato a revisão humana antes de aceitá-lo como referência confiável.

### Human review no golden set

Human review é a etapa em que uma pessoa confirma se uma query e seus passages relevantes propostos realmente formam um par de referência válido, antes de o exemplo entrar no golden set final. Um candidato de avaliação nunca é promovido diretamente para "aprovado" sem essa revisão; o campo de status de revisão de um exemplo recém-gerado começa como pendente, mesmo quando a geração automática parece plausível.

### Recall@K

Recall@K mede, para uma query, se ao menos um (ou todos, dependendo da definição adotada) dos passages relevantes conhecidos aparece entre os k primeiros resultados retornados pelo retrieval. É a métrica mais direta para avaliar se o sistema de retrieval está encontrando a informação certa, mas não diz nada sobre a posição exata dentro do top-k.

### Recall@10 pode enganar em corpus pequeno

Em um corpus pequeno, um retriever fraco ou até aleatório pode atingir Recall@10 artificialmente alto simplesmente porque k=10 já cobre uma fração grande do corpus total. Recomenda-se validar o tamanho do corpus antes de confiar em Recall@K: como regra prática, o corpus deve ter pelo menos cinco vezes mais chunks do que o valor de k avaliado, senão a métrica perde poder de discriminar um retriever bom de um ruim.

### MRR

MRR (Mean Reciprocal Rank) é a média, sobre todas as queries, do inverso da posição do primeiro passage relevante encontrado — quanto mais cedo o passage certo aparece no ranking, mais próximo de 1 o valor fica. Diferente de Recall@K, o MRR é sensível à posição exata do resultado correto, não apenas à sua presença dentro de um corte fixo.

### Diferença entre Recall@K e MRR

Recall@K responde "o passage certo está entre os k primeiros?" de forma binária por query; MRR responde "quão perto do topo ele está?" de forma contínua. Um sistema pode ter Recall@10 perfeito (o passage certo sempre aparece nos 10 primeiros) e ainda assim ter MRR mediano, se o passage certo geralmente aparece na posição 8 ou 9 em vez da posição 1.

### nDCG

nDCG (normalized Discounted Cumulative Gain) compara o ranking obtido com o ranking ideal possível, penalizando resultados relevantes que aparecem em posições mais baixas com um fator logarítmico. Ele é mais informativo que Recall@K quando existe mais de um passage relevante por query, porque considera a posição de todos eles, não apenas a presença de um único resultado correto.

### Faithfulness vs Answer Relevance

Answer relevance mede se a resposta gerada realmente responde à pergunta feita, independentemente de onde a informação veio; faithfulness mede se cada afirmação da resposta está de fato sustentada pelo contexto recuperado. Answer relevance alta com faithfulness baixa é um sintoma clássico de alucinação fluente: a resposta parece boa e pertinente, mas contém informação que não vem dos passages fornecidos — nesse cenário recomenda-se investigar o prompt de geração e a montagem do contexto antes de mexer no retrieval.

### Métricas de retrieval nunca devem ser misturadas com métricas de geração

Recall, MRR e nDCG avaliam se a informação certa foi encontrada; faithfulness e answer relevance avaliam a qualidade do texto gerado a partir dessa informação. É proibido combinar essas métricas em uma única pontuação composta, porque uma pontuação combinada esconde em qual estágio do pipeline está o problema real — uma queda na pontuação composta pode vir do retrieval, da geração, ou das duas ao mesmo tempo, sem forma de diferenciar.

### Gate de regressão no CI / evaluation harness

O evaluation harness roda o golden set contra toda mudança de retrieval antes do merge, comparando a configuração candidata com a configuração baseline em produção. Uma regressão de Recall@10 maior que 1 ponto percentual em relação ao baseline bloqueia o deploy automaticamente; a mudança só pode seguir adiante com uma aprovação manual explícita justificando a exceção.

## Agentes

### O que é um AI agent

Um AI agent é um sistema em que o modelo de linguagem decide, em tempo de execução, quais ações tomar e em qual ordem, em vez de seguir um fluxo fixo definido previamente em código. Isso o diferencia de um pipeline determinístico como um RAG simples, onde a sequência de etapas é sempre a mesma independentemente da pergunta.

### Tools e function calling

Tools são funções externas (busca, cálculo, chamadas de API, escrita em banco de dados) que um agente pode invocar durante sua execução, descritas ao modelo por meio de function calling: um schema que especifica nome, parâmetros e quando cada tool deve ser usada. A qualidade das descrições de tool afeta diretamente a taxa de acerto na escolha da ferramenta certa — descrições ambíguas levam o modelo a escolher a tool errada mesmo quando a correta está disponível.

### Diferença entre agente escolher tool e workflow chamar função direto

Em um workflow tradicional, o código decide explicitamente qual função chamar e quando, e o LLM (se existir) só participa de uma etapa isolada. Em um agente, é o próprio modelo que decide, a cada passo, se e qual tool chamar com base no estado atual da conversa — isso dá flexibilidade para lidar com casos não previstos no fluxo, mas também introduz um ponto de falha novo: o modelo pode escolher a tool errada ou chamar a tool certa com parâmetros incorretos.

### Planning em agentes

Planning é a capacidade de um agente decompor uma tarefa complexa em uma sequência de sub-passos antes (ou durante) a execução, em vez de tentar resolver tudo em uma única chamada. Usa-se planning explícito (o agente descreve seu plano antes de agir) em tarefas de múltiplos passos, porque isso facilita auditar e depurar decisões erradas depois do fato.

### Loops de agente e limite de iterações

Um agente tipicamente executa em loop: decide uma ação, observa o resultado, decide a próxima ação, até concluir a tarefa ou atingir um critério de parada. Sem um limite explícito de iterações, um agente pode entrar em loop repetindo a mesma ação sem progresso; todo agente em produção deve ter um número máximo de iterações configurado, após o qual a execução é interrompida e escalada para revisão humana.

### Tool selection

Tool selection é a decisão, feita pelo modelo, de qual ferramenta entre as disponíveis usar em um dado passo. Quando duas tools têm propósitos parecidos, a taxa de erro na seleção sobe; recomenda-se manter o conjunto de tools disponível para um agente o mais enxuto possível para a tarefa em questão, em vez de expor todas as tools do sistema a todo agente.

### Memória em agentes

Memória, no contexto de agentes, é a capacidade de reter informação relevante entre passos de execução ou entre sessões diferentes, além do que cabe na janela de contexto de uma única chamada. Isso pode ser implementado como um resumo do histórico, um banco de fatos estruturado, ou buscas em interações passadas via retrieval — cada abordagem tem trade-offs diferentes entre custo, fidelidade e latência de acesso.

### Quando não usar agentes

Um agente é a escolha errada quando a sequência de passos necessária já é conhecida e fixa: nesses casos, um workflow determinístico é mais confiável, mais barato e mais fácil de testar do que deixar o modelo decidir a cada execução algo que já poderia estar codificado. Um agente só é recomendado quando a tarefa exige decisões condicionais que dependem do conteúdo específico de cada execução, e não apenas de um fluxo previsível.

## Observabilidade

### Tracing de requisições

Tracing registra, para cada requisição, a sequência completa de estágios executados (retrieval, reranking, geração, chamadas de tool) com seus tempos individuais, permitindo reconstruir exatamente o que aconteceu em uma execução específica. Sem tracing, depurar uma resposta ruim em produção exige adivinhar em qual estágio o problema ocorreu, o que é lento e pouco confiável.

### Logs de prompts e respostas

O prompt final efetivamente enviado ao modelo (já com o contexto montado) e a resposta recebida devem ser registrados, não apenas o template do prompt, porque o comportamento observado depende do conteúdo específico injetado naquela chamada. Esses logs são a principal fonte de evidência ao investigar por que uma resposta específica saiu errada.

### Monitoramento de tokens e custo

Cada chamada de modelo tem um custo proporcional ao número de tokens de entrada e de saída, então monitora-se consumo de tokens por rota de produto, não apenas o custo agregado da conta, para identificar rapidamente qual funcionalidade específica é responsável por um aumento de custo. Picos de custo sem aumento proporcional de tráfego geralmente indicam contexto inflado (prompts muito longos ou muitos documentos recuperados desnecessariamente).

### Latência — o que medir

Latência de um sistema de IA deve ser medida por estágio (retrieval, reranking, geração), não apenas como tempo total de resposta, porque cada estágio tem causas de lentidão diferentes e exige mitigação diferente. A latência do primeiro token também deve ser acompanhada (time to first token) separadamente da latência total, já que isso afeta diretamente a percepção de velocidade em interfaces com streaming.

### Falhas de retrieval — como detectar

Uma falha de retrieval se manifesta como resposta genérica, incompleta ou incorreta mesmo quando a informação existe no corpus; a forma mais confiável de detectar isso é inspecionar diretamente quais passages foram recuperados para aquela query, não apenas a resposta final gerada. Recomenda-se sempre logar os IDs dos passages recuperados junto com cada resposta, exatamente para viabilizar esse tipo de investigação depois do fato.

### Falhas de geração — como detectar

Uma falha de geração acontece quando o contexto recuperado estava correto, mas a resposta final ainda assim está errada, incompleta ou não fundamentada nesse contexto. Isso é identificado comparando o conjunto de passages recuperados (que estava certo) com o conteúdo da resposta (que não reflete esse conteúdo corretamente) — sintoma que aponta para o prompt de geração ou para a montagem do contexto, não para o retrieval.

### Monitoramento em produção (dashboards)

Além de métricas offline calculadas sobre o golden set, mantêm-se dashboards de produção com taxa de erro, latência por estágio, custo por rota e amostras de tracing para revisão manual periódica. Métricas offline validam uma mudança antes do deploy; métricas de produção validam que o comportamento observado no golden set continua valendo sob tráfego real, que é sempre mais variado do que qualquer conjunto de teste.

## Produção

### Retries e backoff

Retries automáticos reexecutam uma chamada de modelo que falhou por erro transitório (timeout, erro 5xx, limite de taxa momentâneo), tipicamente com backoff exponencial entre tentativas para não sobrecarregar ainda mais um serviço já instável. Usa-se no máximo 3 tentativas por chamada; falhas que persistem além disso são tratadas como falha definitiva e acionam o fallback configurado, não uma quarta tentativa.

### Timeouts

Um timeout define o tempo máximo que o sistema espera por uma resposta do modelo antes de desistir da chamada. Usa-se um timeout padrão de 30 segundos para chamadas de geração; chamadas que excedem esse limite são canceladas e tratadas como falha, entrando no fluxo de retry e, se necessário, de fallback, em vez de deixar o usuário esperando indefinidamente.

### Fallback

Fallback é o comportamento acionado quando a chamada principal falha mesmo após os retries configurados: pode ser uma resposta padrão, um modelo mais simples e mais rápido, ou uma mensagem explícita informando indisponibilidade temporária. Todo caminho crítico de produção deve ter um fallback definido explicitamente — nunca deixar uma falha de modelo propagar como erro não tratado até o usuário final.

### Rate limits

Rate limits são o número máximo de chamadas que um sistema pode fazer a um provedor de modelo em um intervalo de tempo, e ultrapassá-los gera erros ou lentidão adicional. Controle de taxa deve ser implementado no próprio lado do cliente (antes de a chamada sair) para distribuir requisições de forma previsível, em vez de depender apenas do tratamento de erro quando o limite do provedor já foi excedido.

### Caching

Caching armazena a resposta de uma chamada de modelo (ou de um estágio como embeddings) para reaproveitar em chamadas futuras equivalentes, reduzindo custo e latência. Faz-se cache de embeddings de chunks (que não mudam entre queries) de forma quase irrestrita, mas aplica cache de respostas de geração com cautela, porque uma resposta cacheada pode ficar desatualizada se o corpus subjacente mudar.

### Controle de custo

Controle de custo envolve decisões deliberadas sobre quando usar um modelo maior e mais caro versus um modelo menor e mais barato, e sobre limitar o tamanho do contexto enviado por chamada. Definem-se orçamentos de custo por funcionalidade de produto, não apenas um teto global, para que uma funcionalidade cara não consuma silenciosamente o orçamento de outras.

### Versionamento de modelos

Assim como prompts, modelos usados em produção têm uma versão fixa referenciada explicitamente (nunca "a versão mais recente" de forma implícita), porque provedores podem atualizar um modelo e mudar seu comportamento sem aviso equivalente a uma mudança de versão maior. A migração para uma nova versão de modelo só ocorre depois de rodar o golden set completo contra ela e comparar os resultados com a versão em produção.

### Feature flags e testes antes do deploy

Feature flags permitem ativar uma mudança de pipeline (novo prompt, novo modelo, novo reranker) para uma fração do tráfego antes de um rollout completo, limitando o impacto de uma regressão não detectada pelo golden set. A passagem pelo evaluation harness deve ser tratada como pré-requisito para abrir uma feature flag em produção, não como substituto dela — o golden set valida qualidade offline, a feature flag valida comportamento sob tráfego real.

## Segurança

### Prompt injection na perspectiva de segurança

Do ponto de vista de segurança, prompt injection é considerado um vetor de ataque: um documento malicioso no corpus, ou um input de usuário cuidadosamente construído, pode tentar fazer o modelo ignorar suas instruções originais e executar uma ação não autorizada, como revelar informação sensível ou chamar uma tool indevida. Qualquer conteúdo vindo de fora do system prompt deve ser tratado (contexto recuperado, input do usuário, saída de uma tool) como não confiável por padrão.

### Vazamento de dados sensíveis

Vazamento de dados ocorre quando informação sensível (credenciais, dados pessoais, segredos internos) presente no contexto ou nos dados de treino acaba aparecendo na resposta para um usuário que não deveria ter acesso a ela. Pipelines de RAG devem aplicar filtragem de dados sensíveis na etapa de ingestão, antes da indexação, porque é muito mais barato impedir que o dado entre no índice do que tentar filtrá-lo depois em toda resposta gerada.

### Controle de acesso a documentos

Controle de acesso garante que um usuário só receba, via retrieval, passages que ele já teria permissão de ver diretamente na fonte original. Isso é implementado propagando os metadados de permissão do documento original para o chunk indexado, e aplicando esse filtro de metadata antes do cálculo de similaridade — nunca depois, para não correr o risco de um passage restrito aparecer no candidate set em nenhum momento do processo.

### Tool permissions e confirmação humana

Nem toda tool disponível a um agente deveria poder ser executada sem supervisão: tools que causam efeitos irreversíveis ou visíveis externamente (enviar uma comunicação, processar um pagamento, excluir um registro) exigem confirmação humana explícita antes da execução, mesmo que o agente tenha decidido de forma correta que aquela ação é necessária. Cada tool deve ser classificada por nível de risco no momento em que ela é registrada no sistema, e tools de alto risco nunca são marcadas como "auto-executáveis" por padrão.

### Validação de output antes de agir

Antes de um agente usar a saída de uma chamada de modelo como entrada para uma ação real (chamar outra tool, gravar em um banco de dados), essa saída deve ser validada estruturalmente — schema correto, campos obrigatórios presentes, valores dentro de limites esperados. A saída de um LLM deve ser tratada como um dado externo não confiável para fins de validação, exatamente como trataria um input vindo de um usuário anônimo.

### Execução de ações externas por agentes

Ações externas (chamadas de API que alteram estado fora do próprio sistema de IA) representam o maior risco em sistemas de agentes, porque um erro de decisão do modelo se torna um efeito real fora do ambiente controlado da conversa. Recomenda-se que toda ação externa irreversível passe por uma camada de confirmação ou por um limite de escopo (por exemplo, um valor máximo de operação) que não pode ser alterado pelo próprio agente durante a execução.

### Dados sensíveis em logs

Logs de observabilidade capturam prompts e respostas completas para fins de depuração, o que significa que dados sensíveis presentes em uma conversa podem acabar armazenados em sistemas de log se nenhum tratamento for aplicado. É exigido mascaramento de campos sensíveis conhecidos (documentos de identificação, dados de pagamento) antes de qualquer prompt ou resposta ser persistido em sistema de observabilidade, mesmo quando esse sistema é interno.
