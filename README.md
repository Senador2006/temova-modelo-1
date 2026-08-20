# Modelo 1 — Previsão de Crescimento da Grama (Rodoanel)

Protótipo de **série temporal** para prever o crescimento acumulado da vegetação (cm) em trechos do Rodoanel, com **API MLOps** e interface web de teste.

## Objetivo

Prever o **Crescimento_Acumulado_cm** com base em variáveis climáticas, solo, índices de vegetação e histórico de poda. O limite operacional de poda é **10 cm** — valores acima disso indicam necessidade de intervenção.

## Arquitetura do modelo

| Aspecto | Detalhe |
|---|---|
| Tipo | LSTM (série temporal) |
| Camadas | LSTM(50) → Dropout → LSTM(50) → Dropout → Dense(1) |
| Janela (`look_back`) | 20 dias consecutivos |
| Features | 72 variáveis |
| Alvo | `Crescimento_Acumulado_cm` |
| Versão atual | 1.2.0 |

## Estrutura do projeto

```
Modelo 1/
├── ml/
│   ├── config.py           # Constantes e caminhos
│   ├── preprocessing.py    # Pipeline de features (igual ao notebook)
│   ├── model_builder.py    # Arquitetura LSTM
│   ├── predictor.py        # Classe de inferência
│   └── prediction_logger.py # Log JSONL de predições
├── logs/                   # Histórico de predições (gerado em runtime)
│   └── predictions.jsonl
├── api/
│   ├── main.py             # FastAPI (endpoints MLOps)
│   └── schemas.py          # Validação Pydantic
├── static/
│   └── index.html          # Interface web de teste
├── artifacts/              # Modelo treinado + scalers + metadata
├── train_model.py          # Treino e exportação de artefatos
├── run_api.py              # Atalho para subir a API
├── requirements.txt
├── Dockerfile
├── notebook_1.ipynb        # Notebook original de exploração
├── RELATORIO.md            # Relatório de desempenho do modelo
└── README.md
```

## Pré-requisitos

- Python 3.13+ (ou 3.14 com Keras 3 + PyTorch)
- Dataset: `data/dataset_rodoanel_10k.csv`

> **Nota:** TensorFlow não suporta Python 3.14 neste ambiente. O serviço usa **Keras 3 + PyTorch** como backend.

## Instalação

```powershell
cd "Modelo 1"
..\venv\Scripts\pip.exe install -r requirements.txt
```

## Treinar o modelo

```powershell
cd "Modelo 1"
$env:KERAS_BACKEND = "torch"
..\venv\Scripts\python.exe train_model.py --epochs 50
```

Artefatos gerados em `artifacts/`:
- `lstm_model.keras`
- `scaler_features.joblib`
- `scaler_target.joblib`
- `metadata.json`

## Subir a API

```powershell
cd "Modelo 1"
$env:KERAS_BACKEND = "torch"
..\venv\Scripts\python.exe run_api.py
```

Acesse:
- **Interface web:** http://127.0.0.1:8000
- **Swagger:** http://127.0.0.1:8000/docs

## Endpoints da API

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/health` | Status do serviço e versão do modelo |
| `GET` | `/model/info` | Metadados do modelo |
| `POST` | `/predict` | Predição a partir de 20+ registros diários (registra log automaticamente) |
| `GET` | `/monitoring/logs` | Histórico de predições para dashboards |
| `GET` | `/monitoring/stats` | Agregados de monitoramento (alertas, médias, por ponto) |
| `GET` | `/sample/{id_ponto}` | Amostra pronta do dataset para teste |
| `GET` | `/` | Interface web |

### Exemplo de predição

```powershell
$body = Get-Content "sample_rodo_km_01.json" -Raw
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST -ContentType "application/json; charset=utf-8" `
  -Headers @{ "X-Source-System" = "dashboard-rodoanel" } -Body $body
```

Resposta esperada:

```json
{
  "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "crescimento_acumulado_cm": 12.43,
  "limite_alerta_cm": 9.0,
  "limite_poda_cm": 10.0,
  "acima_limite_poda": true,
  "nivel_alerta": "laranja",
  "id_ponto": "RODO_KM_01",
  "data_referencia": "2023-01-20",
  "latency_ms": 42.5,
  "model_version": "1.2.0",
  "look_back": 20,
  "registros_utilizados": 20,
  "unidade": "cm"
}
```

## Monitoramento e dashboard

Cada chamada a `POST /predict` grava uma linha JSON em `logs/predictions.jsonl` (configurável via `PREDICTION_LOG_PATH`).

### Consultar histórico

```powershell
# Últimas 50 predições
Invoke-RestMethod "http://127.0.0.1:8000/monitoring/logs?limit=50"

# Filtrar por trecho e nível de alerta
Invoke-RestMethod "http://127.0.0.1:8000/monitoring/logs?id_ponto=RODO_KM_01&nivel_alerta=laranja"
```

### Estatísticas para painel

```powershell
# Agregados das últimas 24 h (padrão)
Invoke-RestMethod "http://127.0.0.1:8000/monitoring/stats"

# Janela de 7 dias
Invoke-RestMethod "http://127.0.0.1:8000/monitoring/stats?window_hours=168"
```

Campos úteis para dashboard externo:
- `by_nivel_alerta_window` — contagem por cor de alerta
- `latest_by_ponto` — última predição de cada trecho
- `alerta_preventivo_rate_pct` / `poda_necessaria_rate_pct` — taxas de alerta
- `avg_crescimento_cm` — média de crescimento previsto na janela

Header opcional `X-Source-System` identifica qual sistema consumidor fez a predição (ex.: `"dashboard-rodoanel"`).

## Arquivos de teste

| Arquivo | Alerta | Previsão esperada |
|---|---|---|
| `sample_alerta_verde.json` | Normal (< 6 cm) | ~1,5 cm |
| `sample_alerta_amarelo.json` | Atenção (6–9 cm) | ~6,0 cm |
| `sample_alerta_vermelho.json` | Alerta preventivo (9–10 cm) | ~9,4 cm |
| `sample_alerta_laranja.json` | Poda necessária (≥ 10 cm) | ~13,9 cm |
| `sample_rodo_km_01.json` | Payload completo RODO_KM_01 | — |
| `samples/manifest.json` | Metadados de todas as amostras | — |

Endpoints de amostra por alerta: `GET /sample-alerta/{verde|amarelo|vermelho|laranja}`

## Interface web

1. Selecione o **ID do ponto** (ex.: `RODO_KM_01`)
2. Clique em **Carregar amostra (20 dias)**
3. Clique em **Prever crescimento**

Na interface, cole **somente o array JSON** (sem `"records":`).

## Docker

Execute a partir da pasta **`Modelo 1`**, usando o contexto da pasta pai (`prototype`):

```powershell
cd "Modelo 1"
docker build -f Dockerfile -t modelo1-grama ..
docker run -p 8000:8000 -v modelo1-logs:/app/logs modelo1-grama
```

> **Importante:** o `..` no final é necessário — o dataset (`dataset_rodoanel_10k.csv`) fica na pasta `prototype`, não dentro de `Modelo 1`.

## Dataset

Gerado por `../dataset_creator.py` com:
- 30 pontos × 365 dias = 10.950 registros
- Limite de poda: **10 cm**
- Valores acima do limite: 10,2 · 11 · 12 · 12,5 · 13,8 · 15 cm

Regenerar o dataset:

```powershell
cd ..
..\venv\Scripts\python.exe dataset_creator.py
```

## Documentação adicional

Consulte [RELATORIO.md](./RELATORIO.md) para métricas de desempenho, matriz de confusão do alerta de poda e recomendações de melhoria.
