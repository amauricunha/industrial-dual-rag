# Gabarito Experimental – Torno ROMI T 240

Este arquivo fornece respostas de referência ("gabaritos") para validar as métricas de accuracy/BLEU/ROUGE do painel Streamlit. Copie o texto do cenário desejado para o campo **"Gabarito (referência para métricas)"** antes de executar o diagnóstico com o checkbox "Gravar logs de experimentos" ativo.

> Os textos abaixo foram sintetizados a partir do **Manual de Operação e Manutenção – ROMI T 240**. Ajuste-os se você modificar os limites operacionais ou incluir novos componentes.

## Como usar
1. Escolha o cenário a ser avaliado (Normal, Falha Térmica ou Desbalanceamento).
2. Injete a falha correspondente no simulador (ou mantenha em operação normal).
3. Cole o gabarito abaixo no campo da UI.
4. Gere o diagnóstico. A API registrará as métricas comparando a resposta do LLM com o gabarito.

## Cenário 1 – Operação Normal
```
Estado geral: NORMAL
Telemetria dentro das faixas especificadas na Seção 4.2 do manual (temperatura < 65 °C, vibração < 2.8 mm/s, corrente até 32 A). Recomenda-se apenas manter o ciclo de inspeção diário e verificar níveis de lubrificante do cabeçote.
```

## Cenário 2 – Falha Térmica / Superaquecimento
```
Estado geral: FALHA TÉRMICA
Temperatura do cabeçote excede 90 °C, ultrapassando o limite de segurança descrito na Seção 5.1.1.
Evidências:
- Telemetria aponta aumento sustentado de temperatura apesar de rotações moderadas.
- Manual cita que acima de 85 °C deve-se parar o eixo e acionar o circuito de refrigeração auxiliar.
Ações recomendadas:
1. Parar imediatamente o torno e manter o eixo principal em repouso.
2. Abrir a tampa de inspeção e verificar obstrução do trocador de calor.
3. Resetar o intertravamento térmico após o resfriamento e preencher o relatório de manutenção.
```

## Cenário 3 – Desbalanceamento / Vibração Elevada
```
Estado geral: ALERTA DE DESBALANCEAMENTO
Vibração RMS acima de 10 mm/s, ultrapassando o limite para eixos em rotação descrito na Seção 6.3.
Evidências:
- Telemetria indica pico de vibração com temperatura normal.
- Manual recomenda comparação com tabela ISO 10816 antes de retomar a produção.
Ações recomendadas:
1. Reduzir a rotação para 50% e executar inspeção visual de fixação da placa.
2. Conferir se há cavaco preso no prato e reapertar castanhas conforme torque nominal.
3. Caso o nível permaneça alto, realizar balanceamento dinâmico do conjunto árvore-cartucho.
```

## Personalização
- Se incluir outras falhas (ex.: queda de corrente, ruído no spindle), replique este formato com **Estado geral → Evidências → Ações** citando a seção do manual e limites numéricos.
- Para testes com textos em inglês, traduza o gabarito e mantenha as métricas; BLEU/ROUGE continuarão válidas.
