# TCC — Izabella Alves Pereira e José André Rabelo Rocha — UnB

Pipeline para priorização de casos de teste usando bugs do Defects4J 3.0.1 em seis projetos:
**Lang**, **Chart**, **Math**, **Time**, **Mockito** e **Compress**.

O pipeline gera `data/processed/features.csv`, treina um Random Forest e compara com baselines
Random e History-based. Os resultados finais ficam em `results/`.

## Features

| Feature | Descrição |
|---|---|
| `history` | Quantas vezes o teste detectou falhas em bugs anteriores do mesmo projeto |
| `same_package` | Se a classe de teste está no mesmo pacote de alguma classe modificada |
| `modified_classes_count` | Número de classes modificadas no bug |
| `historical_failure_rate` | Taxa histórica de falha do teste em bugs anteriores |
| `last_failure_distance` | Distância (em bugs) desde a última falha do teste |
| `test_name_similarity` | Similaridade de Jaccard entre tokens do nome do teste e das classes modificadas |

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
```

O resultado será salvo em `data/processed/features.csv`. Checkouts ficam em `data/raw/`,
a tabela intermediária em `data/intermediate/intermediate.csv` e logs em `logs/`.

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

4. **Random Forest** — split por bug, GridSearchCV (5-fold estratificado) e APFD

   ```bash
   python3 -m src.ml.random_forest_pipeline
   python3 -m src.ml.random_forest_pipeline roc_auc
   ```

5. **Consolidação dos resultados** — estatísticas descritivas e Wilcoxon

   ```bash
   python3 -m src.metrics.consolidate_results
   ```

## Split treino/teste

- Split **por bug** (todas as instâncias de um bug ficam do mesmo lado), **seed 42**
- **Lang-1** entra fixo no treino (sem trigger tests; APFD indefinido)
- Lista completa em `results/train_test_split.json`

| Projeto | Treino | Teste |
|---|---|---|
| Lang | 41 | 17 |
| Chart | 18 | 8 |
| Math | 71 | 31 |
| Time | 18 | 7 |
| Mockito | 22 | 9 |
| Compress | 32 | 14 |

## Resultados (81 bugs válidos de teste)

86 bugs no conjunto de teste; 5 são excluídos da avaliação de APFD por não terem trigger
tests (`Compress-28`, `Math-22`, `Math-41`, `Math-78`, `Mockito-22`).

| Estratégia | APFD médio | Mediana |
|---|---|---|
| Random Forest | **0.6741** | 0.7857 |
| Random | 0.5896 | 0.5397 |
| History-based | 0.5087 | 0.4677 |

APFD médio por projeto (Random Forest):

| Projeto | APFD |
|---|---|
| Compress | 0.8383 |
| Time | 0.7770 |
| Mockito | 0.7028 |
| Math | 0.6400 |
| Chart | 0.6090 |
| Lang | 0.5794 |

Testes de Wilcoxon pareados (p < 0,05): Random Forest supera History-based e Random;
Random supera History-based.

Hiperparâmetros escolhidos pelo GridSearchCV: `n_estimators=100`, `max_depth=10`,
`min_samples_leaf=5`, `min_samples_split=2`, `class_weight=balanced`.

## Arquivos em `results/`

| Arquivo | Conteúdo |
|---|---|
| `random_baseline_apfd.csv` | APFD por bug da baseline Random (média e desvio de 30 seeds) |
| `history_baseline_apfd.csv` | APFD por bug da baseline History-based |
| `random_forest_apfd.csv` | APFD por bug do Random Forest |
| `train_test_split.json` | Bugs de treino/teste por projeto |
| `rf_model.joblib` | Modelo Random Forest treinado |
| `rf_hyperparameters.json` | Hiperparâmetros e score de validação cruzada |
| `apfd_long_format.csv` | Resultados das três estratégias em formato longo |
| `descriptive_statistics.csv` | Estatísticas descritivas do APFD (geral e por projeto) |
| `wins_by_bug.csv` | APFD lado a lado por bug, com melhor/pior estratégia |
| `statistical_tests.csv` | Testes de Wilcoxon pareados entre estratégias |
