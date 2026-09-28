# Projetos pessoais de Pedro Castanheira Costa

Este documento reúne somente projetos pessoais descritos no currículo fornecido. As descrições distinguem o que está em desenvolvimento do que faz parte de um roadmap planejado.

## Cast Review

O Cast Review é uma plataforma open-source, em desenvolvimento, para revisar pull requests com apoio de agentes de IA especializados. A proposta é dividir uma revisão ampla em análises menores — por exemplo, testes, API, arquitetura, segurança, documentação, banco de dados, performance e qualidade — e consolidá-las em um Merge Report. Antes das revisões especializadas, o fluxo prevê um Change Analyzer para entender o escopo das alterações e uma Implementation Spec para organizar o que deve ser verificado.

A arquitetura planejada é um monólito modular composto por dois serviços com responsabilidades distintas. O serviço principal em NestJS e TypeScript cuidará de autenticação, integração com GitHub e gateway de WebSocket. Um serviço Python/FastAPI stateless executará o pipeline dos agentes e transmitirá o progresso por Server-Sent Events (SSE). Essa separação permite que a aplicação mantenha as responsabilidades de produto e integração organizadas, enquanto o pipeline de IA pode executar e reportar suas etapas de forma independente.

O roadmap contempla indexação estrutural do código, arquivos de convenção do repositório, histórico de revisões e um marketplace de agentes. A arquitetura prevista também considera múltiplos provedores de modelos, incluindo modelos locais. Esses itens são planos de evolução do projeto, não devem ser interpretados como funcionalidades já entregues.

## Cast Code

O Cast Code é um agente de codificação acessado por CLI, com subagentes e skills reutilizáveis. Seu objetivo é automatizar partes do fluxo de desenvolvimento: criar Conventional Commits, separar mudanças por escopo em branches e preparar pull requests com descrições detalhadas.

O fluxo pode executar tarefas de ponta a ponta, incluindo pesquisa na web, edição de arquivos e revisão do código produzido. Também se integra ao GitHub e ao Azure DevOps, conectando a execução local do agente aos repositórios e processos de desenvolvimento usados pelo time.

## Tracecast

O Tracecast é um framework open-source de observabilidade para agentes e aplicações que usam modelos de linguagem. Ele registra gastos e traces completos das execuções, tornando visível como uma resposta foi produzida e quais etapas ocorreram durante o fluxo.

O projeto também permite que cada equipe defina avaliadores por projeto. Isso cria um lugar para acompanhar a qualidade de diferentes aplicações com critérios próprios, além de inspecionar traces quando uma execução precisa ser investigada.

## Cast Skills

O Cast Skills é um pacote de skills reutilizáveis para acelerar o desenvolvimento com assistência de IA. A proposta é reunir instruções e fluxos que possam ser reaproveitados em tarefas diferentes, reduzindo trabalho repetitivo e ajudando a melhorar custo e qualidade do código gerado.

O pacote também inclui explicações visuais sobre o que foi construído. A intenção é que as skills não apenas ajudem a produzir mudanças, mas apoiem a compreensão do resultado e facilitem sua revisão posterior.
