# Industrial Dual-Context RAG — Slide Outline

> **Duração alvo:** 15–20 minutos de apresentação + 10 minutos de Q&A.

## 1. Título e Contexto (1 slide)
- Nome do projeto, autores, disciplina, data
- Motivação: diagnósticos ciber-físicos exigem contexto híbrido (manuais + telemetria)

## 2. Problema & Objetivos (1–2 slides)
- Desafios de manutenção em máquinas rotativas
- Objetivo: comparar três níveis de contexto para LLMs (Baseline, RAG estático, Dual)

## 3. Arquitetura Geral (2 slides)
- Visão Docker (simulador, API, web, Ollama)
- Fluxos: Telemetria MQTT e Diagnóstico On-Demand

## 4. Pipeline RAG Dual (2 slides)
- Ingestão de PDFs (chunking + embeddings + ChromaDB)
- Fusão de contexto com telemetria em tempo real e montagem do prompt

## 5. Tecnologias e Modelos (1 slide)
- LLMs (Groq Llama3, Gemini, Ollama)
- Stack Python/Streamlit/MQTT

## 6. Experimentos & Metodologia (2 slides)
- Cenários avaliados, métricas planejadas (accuracy, latência, feedback do operador)
- Procedimento de teste (injeção de falhas, coleta de logs)

## 7. Resultados Esperados / Parciais (2 slides)
- Tabelas/plots comparando respostas entre cenários
- Casos de sucesso e falhas (quando não há chunk relevante, atraso MQTT etc.)

## 8. Demonstração Rápida (1 slide)
- Sequência: upload manual → injeção de falha → geração do relatório com logs

## 9. Limitações e Trabalhos Futuros (1 slide)
- Dependência de rede, necessidade de métricas mais robustas, prompt templates por falha

## 10. Conclusão & Takeaways (1 slide)
- Benefícios do contexto dual
- Próximos passos para produção/linha de pesquisa

## 11. Referências (1 slide)
- Citar papers, ferramentas, datasets utilizados
