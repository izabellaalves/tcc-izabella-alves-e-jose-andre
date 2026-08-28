# TCC

Pipeline para priorização de testes com base em bugs do Defects4J (Lang e Chart).

A etapa de **feature engineering** (checkout dos bugs, extração de features e geração do
dataset em `data/processed/features.csv`) está concluída. As três estratégias de
priorização — **Random**, **History-based** e **Random Forest** — e a **consolidação dos
resultados** (estatísticas descritivas e testes de Wilcoxon) também já estão
implementadas, com os resultados finais em `results/`.

## Como rodar — Parte 1: geração da base de dados (Docker)

Esta parte cobre apenas o início do pipeline: checkout dos bugs no Defects4J e geração
do `data/processed/features.csv`.

**Pré-requisitos:** Docker e Docker Compose.

```bash
docker-compose build
docker-compose up tcc-pipeline
```

O resultado será salvo em `data/processed/features.csv`.

## Como rodar — Parte 2: experimentos de priorização (fora do container)

As etapas abaixo rodam localmente com Python 3 e as dependências de `requirements.txt`
(`pip install -r requirements.txt`), partindo do pressuposto de que
`data/processed/features.csv` já existe. Execute na raiz do projeto, nesta ordem:

1. **Validação do dataset** — confere linhas, bugs, labels e nulos contra os metadados
   esperados (Tabela 6 do capítulo 4):

   ```bash
   python3 -m src.utils.validate_dataset
   ```

2. **Baseline Random** — 30 seeds (0–29) por bug; gera `results/random_baseline_apfd.csv`:

   ```bash
   python3 -m src.baselines.random_baseline
   ```

3. **Baseline History-based** — ordenação determinística por histórico de detecção de
   falhas (empates por ordem alfabética); gera `results/history_baseline_apfd.csv`:

   ```bash
   python3 -m src.baselines.history_baseline
   ```

4. **Random Forest** — split por bug, GridSearchCV (5-fold estratificado, só no treino) e
   priorização por `predict_proba`; gera `results/random_forest_apfd.csv` (e o split,
   modelo e hiperparâmetros, ver lista abaixo). Aceita um argumento opcional com a
   métrica de otimização do GridSearchCV (`f1`, o default, ou `roc_auc` — saídas com
   sufixo `_rocauc`):

   ```bash
   python3 -m src.ml.random_forest_pipeline          # scoring f1 (default)
   python3 -m src.ml.random_forest_pipeline roc_auc  # scoring roc_auc
   ```

5. **Consolidação dos resultados** — junta os três resultados nos mesmos 26 bugs de
   teste, calcula estatísticas descritivas e testes de Wilcoxon pareados:

   ```bash
   python3 -m src.metrics.consolidate_results
   ```

### Notas de reprodutibilidade

- **Bug Lang-1:** não possui nenhum trigger test (label=1) no dataset, então o APFD é
  indefinido para ele. Ele foi **excluído da avaliação de APFD nas três estratégias**
  (aparece com APFD vazio/NaN nos CSVs das baselines), mas **mantido no treinamento** do
  Random Forest — entra fixo no conjunto de treino, já que suas instâncias (todas
  label=0) são válidas para aprendizado, e não faria sentido gastar uma vaga de teste
  num bug sem métrica de ranking possível.
- **Split treino/teste:** feito **por bug** (todas as instâncias de um bug ficam do mesmo
  lado), com **seed fixa 42**. Tamanhos: Lang 43 treino / 18 teste, Chart 18 treino /
  8 teste. A lista exata de bugs de cada lado fica registrada em
  `results/train_test_split.json`.

### Arquivos gerados em `results/`

| Arquivo | Conteúdo |
|---|---|
| `random_baseline_apfd.csv` | APFD por bug da baseline Random (média e desvio de 30 seeds, n_tests, n_trigger_tests). |
| `history_baseline_apfd.csv` | APFD por bug da baseline History-based (determinística, 1 valor por bug). |
| `random_forest_apfd.csv` | APFD por bug do Random Forest nos 26 bugs de teste (variante f1). |
| `random_forest_apfd_rocauc.csv` | Idem, variante roc_auc (valores idênticos à f1 — a busca escolheu os mesmos hiperparâmetros). |
| `train_test_split.json` | Lista auditável dos bugs de treino/teste por projeto, com a seed usada. |
| `rf_model.joblib` / `rf_model_rocauc.joblib` | Modelos Random Forest treinados (cada variante). |
| `rf_hyperparameters.json` / `rf_hyperparameters_rocauc.json` | Hiperparâmetros escolhidos pelo GridSearchCV e score de validação cruzada. |
| `apfd_long_format.csv` | Os três resultados empilhados em formato longo (strategy, project, bug, apfd, ...) — insumo para os gráficos. |
| `descriptive_statistics.csv` | Média, mediana, desvio, mín/máx e quartis do APFD por estratégia (geral e por projeto). |
| `wins_by_bug.csv` | APFD das três estratégias lado a lado por bug, com melhor/pior estratégia de cada bug. |
| `statistical_tests.csv` | Testes de Wilcoxon pareados entre as três estratégias (estatística, p-valor, significância a p<0.05). |
