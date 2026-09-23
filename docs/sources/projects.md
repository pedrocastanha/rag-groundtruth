# Projetos e aplicações de IA

Este documento reúne projetos profissionais e pessoais descritos no currículo de Pedro Castanheira Costa. Os projetos mostram aplicações de agentes, RAG, visão computacional, observabilidade e ferramentas de engenharia.

## Projetos profissionais na Inova Soluções Educacionais

### Clara AI

Backend de agente de IA multi-tenant, com treinamento distinto por módulo e cliente. As bases de conhecimento RAG são particionadas por módulo de cliente usando Qdrant. O sistema avalia aproximadamente 140.000 submissões de alunos por mês e reduziu em aproximadamente 99% o custo por atividade corrigida em comparação com a correção humana.

### Converte AI

Agente de vendas construído com LangGraph que orquestra três subagentes especializados, do pitch à coleta de dados. Usa function calling e gerenciamento de contexto conversacional. Opera 24 horas por dia e matricula entre 40 e 70 novos alunos por dia para uma filial.

### Documents Validator

Modelo de visão computacional baseado em ConvNeXt para classificar tipos de documentos brasileiros e validar sua autenticidade. Rejeita submissões não oficiais ou com tipo incorreto antes do processamento pago de OCR e verificação, reduzindo em aproximadamente 60% os custos de processamento do Document AI.

### Chatbots educacionais

Agentes com acesso a tools em sistemas internos. Além de responder perguntas, podem executar ações diretamente nos sistemas, reduzindo o volume de solicitações de suporte atendidas pela equipe.

### Observabilidade e guardrails compartilhados

Os projetos do time foram instrumentados com LangSmith para registrar tokens e custos, acompanhar taxas de alucinação e erros, e inspecionar traces. O trabalho também inclui testes com Pytest, mitigação de prompt injection, validação de entrada e integração com PostgreSQL, MongoDB e Redis.

## Projetos pessoais

### Cast Review

Plataforma open-source, em desenvolvimento, para revisar pull requests com agentes de IA especialistas. O fluxo prevê Change Analyzer, Implementation Spec e agentes dedicados a testes, API, arquitetura, segurança, documentação, banco de dados, performance e qualidade, reunidos em um Merge Report.

A arquitetura planejada é um monólito modular: um serviço NestJS/TypeScript cuida de autenticação, integração com GitHub e gateway WebSocket; um serviço Python/FastAPI stateless executa o pipeline de agentes e transmite progresso via SSE. O roadmap contempla indexação estrutural de código, arquivos de convenção, histórico de revisões e marketplace de agentes com múltiplos provedores, inclusive modelos locais.

### Cast Code

Agente de codificação via CLI com subagentes e skills. Automatiza Conventional Commits, divisão de branches por escopo e criação de pull requests com descrições detalhadas. Integra-se ao GitHub e Azure DevOps, pesquisa na web e pode executar tarefas de ponta a ponta, incluindo edição e revisão de código.

### Tracecast

Framework open-source de observabilidade para agentes e aplicações com LLMs. Registra gastos, permite que equipes definam avaliadores por projeto e exibe traces completos de execução.

### Cast Skills

Pacote de skills reutilizáveis para acelerar o desenvolvimento, reduzir custos e melhorar a qualidade do código gerado por IA. Inclui explicações visuais sobre o que foi construído.
