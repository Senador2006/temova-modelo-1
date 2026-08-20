# Relatório de Desempenho — Modelo 1 (v1.2.0)

**Data de referência:** Agosto/2026  
**Alvo:** `Crescimento_Acumulado_cm` (crescimento acumulado da grama em cm)  
**Limite operacional de poda:** 10 cm  
**Alerta preventivo:** 9 cm

---

## 1. Resumo executivo

O Modelo 1 é um **LSTM de série temporal** treinado para prever o crescimento acumulado da vegetação em trechos do Rodoanel. A versão **1.2.0** implementa as três recomendações prioritárias da v1.1.0: retreino com **50 epochs**, **oversampling 3×** da faixa ≥ 10 cm e ajuste do limite operacional (**alerta em 9 cm**, **poda em 10 cm**).

| Indicador | v1.1.0 | v1.2.0 | Evolução |
|---|---|---|---|
| MAE (teste) | 0,89 cm | **0,47 cm** | −47% |
| R² (teste) | 0,889 | **0,948** | +6,6% |
| MAE faixa ≥ 10 cm | 1,82 cm | **1,50 cm** | −18% |
| Acurácia alerta poda | 90,8% | **95,2%** | +4,4 pp |
| Recall alerta poda | 86,7% | **98,8%** | +12,1 pp |
| Precisão alerta poda | 57,6% | **71,8%** | +14,2 pp |

**Conclusão:** A v1.2.0 apresenta **melhoria substancial** em todas as métricas principais. O alerta de poda mantém **recall quase perfeito** (98,8%) com **precisão significativamente maior** (71,8%). O limite preventivo em 9 cm reforça a detecção antecipada de casos críticos.

---

## 2. Configuração do modelo

| Parâmetro | v1.1.0 | v1.2.0 |
|---|---|---|
| Arquitetura | LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1) | Igual |
| Versão | 1.1.0 | **1.2.0** |
| Backend | Keras 3 + PyTorch | Igual |
| Otimizador | Adam | Igual |
| Loss | Mean Squared Error | Igual |
| Janela temporal (`look_back`) | 20 dias | Igual |
| Número de features | 72 | Igual |
| Epochs de treino | 15 | **50** |
| Batch size | 32 | 32 |
| Validation split | 20% | 20% |
| Oversampling faixa ≥ 10 cm | Não | **Sim (fator 3×, só treino)** |
| Limite alerta preventivo | — | **9 cm** |
| Limite poda | 10 cm | **10 cm** |

---

## 3. Dataset

| Métrica | Valor |
|---|---|
| Arquivo | `dataset_rodoanel_10k.csv` |
| Registros totais | 10.950 |
| Pontos (trechos) | 30 |
| Período | 365 dias (2023) |
| Crescimento mínimo | 0,2 cm |
| Crescimento máximo | 15,0 cm |
| Crescimento médio | 6,02 cm |
| Registros acima de 10 cm | 11,4% (1.244 registros) |

### Regra de poda no dataset

- **Limite:** 10 cm
- Quando a grama ultrapassa 10 cm, o dataset registra valores como **10,2 · 11 · 12 · 12,5 · 13,8 · 15,0 cm** antes da poda
- Após a poda, o crescimento acumulado é resetado para ~0 cm

---

## 4. Divisão dos dados

| Conjunto | Sequências | Proporção |
|---|---|---|
| Total | 10.350 | 100% |
| Treino (base) | 8.280 | 80% |
| Treino (após oversampling) | ~10.044 | — |
| Teste | 2.070 | 20% |

Cada sequência contém **20 dias** de histórico para prever o **21º dia**. O oversampling triplica sequências com alvo ≥ 10 cm **apenas no conjunto de treino** (248 → +496 amostras extras).

---

## 5. Métricas de regressão (conjunto de teste)

| Métrica | v1.1.0 | v1.2.0 | Descrição |
|---|---|---|---|
| **MAE** | 0,885 cm | **0,471 cm** | Erro médio absoluto |
| **RMSE** | 1,209 cm | **0,828 cm** | Raiz do erro quadrático médio |
| **R²** | 0,889 | **0,948** | Coeficiente de determinação |
| **MAPE** | 23,8% | **8,7%** | Erro percentual absoluto médio |
| **Viés (erro médio)** | +0,071 cm | **+0,201 cm** | Leve tendência a superestimar |

### Distribuição do erro absoluto (v1.2.0)

| Faixa de erro | v1.1.0 | v1.2.0 |
|---|---|---|
| < 1 cm | 70,5% | **88,1%** |
| < 2 cm | 90,9% | **95,0%** |
| < 3 cm | 96,2% | **98,1%** |

---

## 6. Desempenho por faixa de crescimento

| Faixa | Amostras (teste) | MAE v1.1.0 | MAE v1.2.0 |
|---|---|---|---|
| **≤ 10 cm** (normal) | 1.822 | 0,757 cm | **0,332 cm** |
| **≥ 10 cm** (zona de poda) | 248 | 1,824 cm | **1,498 cm** |

O oversampling e o retreino prolongado reduziram o erro na faixa crítica em **~18%**, mantendo excelente desempenho na operação normal.

---

## 7. Desempenho do alerta de poda

Classificação binária: previsão **≥ 10 cm** = poda recomendada.

| Métrica | v1.1.0 | v1.2.0 |
|---|---|---|
| Acurácia | 90,8% | **95,2%** |
| Precisão | 57,6% | **71,8%** |
| Recall | 86,7% | **98,8%** |
| F1-Score | 0,692 | **0,832** |

### Matriz de confusão (v1.2.0)

|  | Previsto < 10 cm | Previsto ≥ 10 cm |
|---|---|---|
| **Real < 10 cm** | 1.726 (VN) | 96 (FP) |
| **Real ≥ 10 cm** | 3 (FN) | 245 (VP) |

### Alerta preventivo (≥ 9 cm)

| Métrica | v1.2.0 |
|---|---|
| Acurácia | 97,0% |
| Precisão | 90,7% |
| Recall | 95,1% |
| F1-Score | 0,928 |

### Interpretação

- **Recall quase perfeito (98,8%):** Apenas 3 casos críticos passaram despercebidos no teste (vs. 33 na v1.1.0).
- **Precisão melhorada (71,8%):** Falsos positivos caíram de 158 para 96 (−39%).
- **Alerta em 9 cm:** Camada adicional de monitoramento com F1 de 0,928, capturando casos antes do limite de poda.

---

## 8. Comparação entre versões

| Aspecto | v1.0.0 | v1.1.0 | v1.2.0 |
|---|---|---|---|
| Crescimento máximo | ~50 cm | 15 cm | 15 cm |
| MAE (teste) | ~8,47 cm | 0,89 cm | **0,47 cm** |
| Epochs | — | 15 | **50** |
| Oversampling | Não | Não | **Sim (3×)** |
| Alerta preventivo | Não | Não | **9 cm** |
| Limite poda | 120 dias | 10 cm | **10 cm** |

---

## 9. Pontos fortes

1. **MAE reduzido pela metade** — De 0,89 cm para 0,47 cm no conjunto de teste.
2. **R² de 0,948** — Modelo explica ~95% da variância do crescimento.
3. **Recall de poda quase perfeito** — 98,8% dos casos críticos detectados.
4. **Menos falsos alarmes** — Precisão de poda subiu de 57,6% para 71,8%.
5. **Pipeline MLOps completo** — API, interface web, artefatos versionados e Docker.

---

## 10. Pontos de atenção

1. **Erro ainda maior acima de 10 cm** — MAE de 1,50 cm na zona de poda (melhorou, mas permanece ~4,5× o MAE abaixo de 10 cm).
2. **Viés positivo (+0,20 cm)** — Modelo tende levemente a superestimar após retreino.
3. **96 falsos positivos de poda** — Redução significativa, mas ainda presentes.
4. **Dados sintéticos** — Validação com dados reais do Rodoanel ainda pendente.

---

## 11. Recomendações implementadas (v1.2.0)

| Prioridade | Ação | Status | Resultado |
|---|---|---|---|
| Alta | Retreinar com **50 epochs** | ✅ Implementado | MAE −47% |
| Alta | **Oversampling 3×** faixa ≥ 10 cm | ✅ Implementado | MAE ≥10 cm −18%, recall poda +12 pp |
| Média | Alerta em **9 cm**, poda em **10 cm** | ✅ Implementado | F1 alerta preventivo 0,928 |

---

## 12. Próximos passos sugeridos

1. Implementar **early stopping** com patience no retreino.
2. ~~Implementar logging de predições na API para monitoramento contínuo.~~ ✅ Endpoints `/monitoring/logs` e `/monitoring/stats`
3. Validar com dados reais do Rodoanel quando disponíveis.
4. Avaliar modelo GRU (também testado no notebook) como alternativa.
5. Hyperparameter tuning (Keras Tuner, como no notebook).

---

*Relatório atualizado com base na avaliação do modelo LSTM v1.2.0 sobre o dataset `dataset_rodoanel_10k.csv` (10.950 registros, 30 pontos, limite de poda 10 cm, alerta preventivo 9 cm).*
