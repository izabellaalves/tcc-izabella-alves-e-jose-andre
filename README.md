# TCC — Izabella Alves Pereira e José André Rabelo Rocha — UnB

Pipeline para priorização de casos de teste usando bugs do Defects4J 3.0.1 em seis projetos:
**Lang**, **Chart**, **Math**, **Time**, **Mockito** e **Compress**.

O pipeline gera `data/processed/features.csv` (294 bugs, 122.560 instâncias teste-bug), treina um
Random Forest e compara com baselines Random, History-based e Same-Package. Os resultados finais
ficam em `results/`.

## Features

| Feature | Descrição |
|---|---|
| `history` | Quantas vezes o teste detectou falhas em bugs anteriores do mesmo projeto |
| `same_package` | Se a classe de teste está no mesmo pacote de alguma classe modificada |
| `modified_classes_count` | Número de classes modificadas no bug |

## Como rodar — Parte 1: geração da base de dados (Docker)

**Pré-requisitos:** Docker e Docker Compose.

```bash
docker-compose build
docker-compose up tcc-pipeline
```

Opções úteis:

```bash
# Reutilizar checkouts existentes em data/raw
docker-compose run --rm tcc-pipeline python3 scripts/prepare_dataset.py --skip-checkout

# Recalcular features a partir de intermediate.csv (pula checkout, metadados e enumeração)
docker-compose run --rm tcc-pipeline python3 scripts/prepare_dataset.py --skip-to-features

# Processar apenas alguns projetos
docker-compose run --rm tcc-pipeline python3 scripts/prepare_dataset.py --projects Lang,Chart

# Recuperar bugs faltantes de forma incremental (merge seguro em features.csv)
docker-compose run --rm tcc-pipeline python3 scripts/add_missing_bugs.py
```

O resultado será salvo em `data/processed/features.csv`. Checkouts ficam em `data/raw/`,
a tabela intermediária em `data/intermediate/intermediate.csv` e logs em `logs/`.

Para executar scripts com o código-fonte atualizado sem rebuild da imagem, monte o diretório `src`:

```bash
docker-compose run --rm -v ./src:/app/src tcc-pipeline python3 -m src.ml.random_forest_pipeline
```

## Como rodar — Parte 2: experimentos de priorização

Execute na raiz do projeto, nesta ordem. Pode ser dentro ou fora do container
(`docker-compose run --rm tcc-pipeline <comando>`).

1. **Validação do dataset**

   ```bash
   python3 -m src.utils.validate_dataset
   ```

2. **Baseline Random** — 30 seeds (0–29) por bug

   ```bash
   python3 -m src.baselines.random_baseline
   ```

3. **Baseline History-based** — ordenação por histórico de detecção

   ```bash
   python3 -m src.baselines.history_baseline
   ```

4. **Baseline Same-Package** — ordenação por `same_package` decrescente

   ```bash
   python3 -m src.baselines.same_package_baseline
   ```

5. **Random Forest** — split cronológico 70/30, GridSearchCV (5-fold por bug) e APFD

   ```bash
   python3 -m src.ml.random_forest_pipeline
   ```

6. **Consolidação dos resultados** — estatísticas descritivas e Wilcoxon (RF, Random, History)

   ```bash
   python3 -m src.metrics.consolidate_results
   ```

## Split treino/teste

- Split **por bug** (todas as instâncias de um bug ficam do mesmo lado)
- **Cronológico 70/30** por id de bug dentro de cada projeto:
  - Bugs **sem trigger tests** (`label=0` em todos os testes) → treino
  - Dos demais, os primeiros 70% por id → treino; os últimos 30% → teste
- Lista completa em `results/train_test_split.json`

| Projeto | Treino | Teste | Sem trigger (treino) |
|---|---:|---:|---:|
| Lang | 43 | 18 | 1 |
| Chart | 18 | 8 | 0 |
| Math | 75 | 30 | 6 |
| Time | 17 | 8 | 0 |
| Mockito | 22 | 9 | 2 |
| Compress | 33 | 13 | 4 |

**Total:** 208 bugs de treino, 86 bugs de teste (todos com trigger tests).

## Resultados (86 bugs de teste)

| Estratégia | APFD médio | Mediana |
|---|---:|---:|
| Same-Package | **0.6530** | 0.7000 |
| Random Forest | 0.6248 | 0.6757 |
| Random | 0.5741 | 0.5318 |
| History-based | 0.5461 | 0.5327 |

APFD médio por projeto (Random Forest):

| Projeto | APFD |
|---|---:|
| Time | 0.7895 |
| Compress | 0.7758 |
| Math | 0.6495 |
| Chart | 0.5892 |
| Mockito | 0.5321 |
| Lang | 0.4636 |

Testes de Wilcoxon pareados (86 bugs, p < 0,05):

| Comparação | p-valor | Significativo |
|---|---:|---|
| History-based vs Random Forest | 0,0005 | Sim (RF melhor) |
| Random vs Random Forest | 0,115 | Não |
| Random vs History-based | 0,434 | Não |
| Same-Package vs Random Forest | 0,250 | Não |

Hiperparâmetros escolhidos pelo GridSearchCV: `n_estimators=100`, `max_depth=null`,
`min_samples_leaf=1`, `min_samples_split=2`, `class_weight=balanced`.

## Arquivos em `results/`

| Arquivo | Conteúdo |
|---|---|
| `random_baseline_apfd.csv` | APFD por bug da baseline Random (média e desvio de 30 seeds) |
| `history_baseline_apfd.csv` | APFD por bug da baseline History-based |
| `same_package_baseline_apfd.csv` | APFD por bug da baseline Same-Package |
| `random_forest_apfd.csv` | APFD por bug do Random Forest |
| `train_test_split.json` | Bugs de treino/teste por projeto |
| `rf_model.joblib` | Modelo Random Forest treinado |
| `rf_hyperparameters.json` | Hiperparâmetros e score de validação cruzada |
| `apfd_long_format.csv` | Resultados de RF, Random e History em formato longo |
| `descriptive_statistics.csv` | Estatísticas descritivas do APFD (geral e por projeto) |
| `wins_by_bug.csv` | APFD lado a lado por bug, com melhor/pior estratégia |
| `statistical_tests.csv` | Testes de Wilcoxon pareados entre RF, Random e History |
