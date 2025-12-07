# Plano de Melhoria Contínua

Este documento consolida os próximos incrementos necessários para concluir o projeto e preparar a entrega final da disciplina. Classifiquei as ações por trilhas e indiquei dependências quando pertinente.

## 1. Experimentos e Métricas
- [x] Incluir no UI e na API campos editáveis para `base_system`, `instructions` e `response_format`, aplicando o seguinte template por padrão:

  ```json
  {
    "instructions": [
      "1. Sempre priorize valores de telemetria frente ao texto do contexto.",
      "2. Use o contexto apenas como referência para limites e recomendações.",
      "3. Classifique o estado geral entre NORMAL, ALERTA ou FALHA.",
      "4. Justifique todas as decisões usando valores numéricos e trechos do contexto.",
      "5. Nunca invente valores ou limites que não estiverem na telemetria ou no contexto.",
      "6. Sempre cite a fonte do contexto entre colchetes quando possível.",
      "7. Responda obrigatoriamente no formato JSON especificado a seguir."
    ],
    "response_format": {
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
  }
  ```
- [x] Expor na UI um controle para configurar o timeout de chamadas ao LLM local (e persistir no .env/API), permitindo valores superiores a 180s conforme o tamanho do prompt.
- [x] Implementar outros Rags como opção e liberar como parametros no UI, como FAISS, Weaviate e Pinecone.
- [x] garanta que temos no sistema document chunking, embedding via transformer encoders.
- [x] Definir métricas objetivas para cada cenário: accuracy de recuperação, BLEU, ROUGE, avaliação humana 1-5, latência fim-a-fim, logar isso no relatorio a cada experimento, incluindo o vetor de embending comparativo nao sei, para vermos os vetores comparativos de alguma forma nos chunks e confirme que tudo isso seja gravado na pasta compartilhada com o host para o relatorio em formato adequado para o artigo. 
- [x] Persistir resultados agregados (ex.: CSV + gráficos) para inclusão na seção de Experimentos do paper quando checkbox marcado.

## 2. Análise / Limitações
- [x] Usar o novo checkbox de "Gravar logs" para coletar evidências de falhas (sem chunks, timeouts MQTT, limites de tokens), além dos logs do sistema para comprovação dos experimentos. Tokens de prompt/resposta passaram a ser registrados automaticamente quando o checkbox está ativo.
- [x] Criar rotina que consolida esses logs em tabelas/resumos citáveis no relatório e que seja acionada por um botão no dashboard (ex.: exportar CSV/tabela direto da UI).
- [x] Documentar limitações operacionais (dependência do broker público, estabilidade da API externa, limites de tokens) no `README.md` e no `docs/paper_draft.tex`.

## 3. Originalidade e Contribuições
- [x] Adicionar heurísticas de seleção de sensores: permitir marcar quais sinais entram no prompt para testes e experimentos.

## 4. Relatório e Slides
- [x] Completar `docs/paper_draft.tex` (Introdução → Conclusão) com citações das fontes dos simuladores/LLMs.
- [x] Incorporar tabelas/figuras dos experimentos (assim que a etapa 1 estiver concluída).
- [x] Preparar os slides da apresentação em `docs/presentation.md` (estrutura disponível neste repositório) e gerar o PDF final.
- [x] Comentarios no codigo.
