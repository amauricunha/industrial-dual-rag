# Plano de Melhoria Contínua

Este documento consolida os próximos incrementos necessários para concluir o projeto e preparar a entrega final da disciplina. Classifiquei as ações por trilhas e indiquei dependências quando pertinente.

## 1. Experimentos e Métricas
- [ ] Incluir os campos no UI e no API: 
      "instructions": [
    "1. Sempre priorize valores de telemetria frente ao texto do contexto.",
    "2. Use o contexto apenas como referência para limites e recomendações.",
    "3. Classifique o estado geral entre NORMAL, ALERTA ou FALHA.",
    "4. Justifique todas as decisões usando valores numéricos e trechos do contexto.",
    "5. Nunca invente valores ou limites que não estiverem na telemetria ou no contexto.",
    "6. Sempre cite a fonte do contexto entre colchetes quando possível.",
    "7. Responda obrigatoriamente no formato JSON especificado a seguir."
    ],"response_format": {
    "estado_geral": "NORMAL | ALERTA | FALHA",
    "avaliacao_telemetria": {
      "temperatura": { "valor": "", "analise": "" },
      "vibracao_rms": { "valor": "", "analise": "" },
      "corrente_motor": { "valor": "", "analise": "" },
      "rpm": { "valor": "", "analise": "" }
    },
    "diagnostico_resumido": "",
    "causas_provaveis": [""],
    "acoes_recomendadas": [""],
    "limites_referenciados": [
      { "variavel": "", "limite": "", "fonte_contexto": "" }
    ],
    "justificativa": "",
    "trechos_utilizados": [""]
  }
- [ ] Implementar outros Rags como opção e liberar como parametros no UI, como FAISS, Weaviate e Pinecone.
- [ ] garanta que temos no sistema document chunking, embedding via transformer encoders.
- [ ] Definir métricas objetivas para cada cenário: accuracy de recuperação, BLEU, ROUGE, avaliação humana 1-5, latência fim-a-fim, logar isso no relatorio a cada experimento, incluindo o vetor de embending comparativo nao sei, para vermos os vetores comparativos de alguma forma nos chunks.
- [ ] Instrumentar scripts automatizados para rodar as três condições (Baseline, RAG estático, Dual) com as mesmas perguntas (precisamos habilitar na interface o base_system variavel (sempre carrega o padroa que esta hoje fixo no main mas podemos editar e o api recebe como parametro).
- [ ] Persistir resultados agregados (ex.: CSV + gráficos) para inclusão na seção de Experimentos do paper.

## 2. Análise / Limitações
- [ ] Usar o novo checkbox de "Gravar logs" para coletar evidências de falhas (sem chunks, timeouts MQTT, limites de tokens).
- [ ] Criar rotina que consolida esses logs em tabelas/resumos citáveis no relatório.
- [ ] Documentar limitações operacionais (dependência do broker público, estabilidade da API externa, etc.).

## 3. Originalidade e Contribuições
- [ ] Definir templates de prompt por tipo de falha (normal, superaquecimento, desbalanceamento) e expor escolha/edição no frontend.
- [ ] Adicionar heurísticas de seleção de sensores: permitir marcar quais sinais entram no prompt.

## 4. Relatório e Slides
- [ ] Completar `docs/paper_draft.tex` (Introdução → Conclusão) com citações das fontes dos simuladores/LLMs.
- [ ] Incorporar tabelas/figuras dos experimentos (assim que a etapa 1 estiver concluída).
- [ ] Preparar os slides da apresentação em `docs/presentation.md` (estrutura disponível neste repositório) e gerar o PDF final.
