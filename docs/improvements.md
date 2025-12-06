# Plano de Melhoria Contínua

Este documento consolida os próximos incrementos necessários para concluir o projeto e preparar a entrega final da disciplina. Classifiquei as ações por trilhas e indiquei dependências quando pertinente.

## 1. Experimentos e Métricas *(executar por último)*
- [ ] Definir métricas objetivas para cada cenário (ex.: accuracy de recuperação, avaliação humana 1-5, latência fim-a-fim).
- [ ] Instrumentar scripts automatizados para rodar as três condições (Baseline, RAG estático, Dual) com as mesmas perguntas.
- [ ] Persistir resultados agregados (ex.: CSV + gráficos) para inclusão na seção de Experimentos do paper.

## 2. Análise / Limitações
- [ ] Usar o novo checkbox de "Gravar logs" para coletar evidências de falhas (sem chunks, timeouts MQTT, limites de tokens).
- [ ] Criar rotina que consolida esses logs em tabelas/resumos citáveis no relatório.
- [ ] Documentar limitações operacionais (dependência do broker público, estabilidade da API externa, etc.).

## 3. Originalidade e Contribuições
- [ ] Definir templates de prompt por tipo de falha (normal, superaquecimento, desbalanceamento) e expor escolha/edição no frontend.
- [ ] Adicionar heurísticas de seleção de sensores: permitir marcar quais sinais entram no prompt ou aplicar filtros baseados em thresholds.
- [ ] Avaliar integração com bibliotecas de diagnóstico vibracional para enriquecer o contexto dinâmico.

## 4. Relatório e Slides
- [ ] Completar `docs/paper_draft.tex` (Introdução → Conclusão) com citações das fontes dos simuladores/LLMs.
- [ ] Incorporar tabelas/figuras dos experimentos (assim que a etapa 1 estiver concluída).
- [ ] Preparar os slides da apresentação em `docs/presentation.md` (estrutura disponível neste repositório) e gerar o PDF final.
