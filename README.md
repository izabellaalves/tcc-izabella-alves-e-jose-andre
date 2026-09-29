# TCC — Izabella Alves Pereira e José André Rabelo Rocha — UnB

Pipeline para priorização de casos de teste usando bugs do Defects4J 3.0.1 em seis projetos:
**Lang**, **Chart**, **Math**, **Time**, **Mockito** e **Compress**.

O pipeline gera `data/processed/features.csv` (297 bugs, 133.789 instâncias teste-bug), treina um
Random Forest e compara com baselines Random, History-based e Same-Package. Os resultados finais
ficam em `results/`.

Os 297 bugs são todos os 304 bugs ativos do Defects4J nesses projetos, menos sete de Mockito
(16, 17 e 34 a 38) que não compilam neste ambiente. O motivo de cada ausência fica registrado em
`data/processed/dataset_build_report.csv`. Todos os 638 trigger tests de referência do Defects4J
estão rotulados, e nenhum bug ficou sem trigger.

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

### Geração em fluxo, com retomada

`scripts/prepare_dataset.py` faz o checkout de todos os bugs antes de enumerar, o que exige manter
mais de 300 workspaces compilados em disco. A alternativa processa um bug por vez e descarta o
checkout em seguida, grava as linhas à medida que ficam prontas e retoma de onde parou:

```bash
# Base inteira, um bug por vez
docker-compose run --rm tcc-pipeline python3 scripts/build_dataset_streaming.py

# Em paralelo, um processo por projeto ou faixa de bugs
docker-compose run --rm tcc-pipeline python3 scripts/build_dataset_streaming.py \
  --projects Math --bugs 1-53 --suffix _MathA

# Junta as partes e calcula as features de uma vez
docker-compose run --rm tcc-pipeline python3 scripts/merge_dataset_parts.py
```

As features saem do merge, não de cada parte: `history` acumula ao longo de todos os bugs
anteriores do projeto, então só faz sentido depois que as partes se juntam.

### Datas de commit, necessárias para o split cronológico

```bash
python3 scripts/export_bug_dates.py
```

Gera `data/processed/bug_commit_dates.csv`, que é versionado. Precisa dos repositórios do
Defects4J clonados (`defects4j/init.sh`), mas só uma vez: o pipeline de ML lê o CSV.

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

5. **Random Forest** — split cronológico 70/30, GridSearchCV (GroupKFold por project+bug) e APFD

   ```bash
   python3 -m src.ml.random_forest_pipeline
   # opcional: otimiza a AUC em vez do F1, mais coerente com o objetivo de ordenar
   python3 -m src.ml.random_forest_pipeline roc_auc
   ```

   Cria `results/train_test_split.json`, então precisa rodar antes da consolidação.

6. **Consolidação dos resultados** — estatísticas descritivas e Wilcoxon entre as quatro estratégias

   ```bash
   python3 -m src.metrics.consolidate_results
   ```

7. **Ablação de features** — o que o modelo acrescenta à heurística `same_package`

   ```bash
   python3 -m src.ml.ablation
   ```

## Split treino/teste

- Split **por bug**: todas as instâncias de um bug ficam do mesmo lado.
- **Cronológico 70/30 pela data do commit de correção**, dentro de cada projeto:
  - Bugs **sem trigger tests** → treino. Com o enumerador corrigido não existe mais nenhum,
    mas a regra fica no código como garantia.
  - Dos demais, os 70% mais antigos → treino; os 30% mais recentes → teste.
- Lista completa em `results/train_test_split.json`.

A ordem vem de `data/processed/bug_commit_dates.csv`, **não do id do bug**. O id do Defects4J não
acompanha o tempo de forma uniforme: em Lang, Chart, Math, Time e Mockito ele cresce para trás
(Lang-1 é o LANG-747, de 2011, e Lang-65 é o LANG-59, de 2002), e só em Compress cresce para
frente. São 257 dos 304 bugs na ordem invertida, então ordenar por id treinaria no futuro e
testaria no passado na maior parte da base. A feature `history` usa a mesma ordem, pelo mesmo motivo.

| Projeto | Treino | Teste |
|---|---:|---:|
| Lang | 43 | 18 |
| Chart | 18 | 8 |
| Math | 74 | 32 |
| Time | 18 | 8 |
| Mockito | 22 | 9 |
| Compress | 33 | 14 |

**Total:** 208 bugs de treino, 89 bugs de teste, todos com trigger tests.

## Resultados (89 bugs de teste)

| Estratégia | APFD médio | Mediana | Desvio |
|---|---:|---:|---:|
| Same-Package | **0.7412** | 0.8500 | 0.2599 |
| Random Forest | 0.7314 | 0.8333 | 0.2569 |
| History-based | 0.6112 | 0.6307 | 0.2662 |
| Random | 0.5901 | 0.5436 | 0.1405 |

APFD médio por projeto:

| Projeto | n | Random | History | Same-Package | Random Forest |
|---|---:|---:|---:|---:|---:|
| Lang | 18 | 0.5723 | 0.6311 | 0.7387 | **0.7389** |
| Chart | 8 | 0.6062 | 0.7383 | **0.7813** | 0.7383 |
| Math | 32 | 0.5540 | 0.5432 | **0.6266** | 0.6213 |
| Time | 8 | 0.6443 | 0.5341 | 0.7307 | **0.7582** |
| Mockito | 9 | 0.7459 | 0.7244 | **0.9139** | 0.9094 |
| Compress | 14 | 0.5552 | 0.6395 | **0.8784** | 0.8398 |

Testes de Wilcoxon pareados (89 bugs):

| Comparação | p-valor | Significativo |
|---|---:|---|
| Random vs History-based | 0,4803 | Não |
| Random vs Same-Package | <0,0001 | Sim (Same-Package melhor) |
| Random vs Random Forest | <0,0001 | Sim (RF melhor) |
| History-based vs Same-Package | <0,0001 | Sim (Same-Package melhor) |
| History-based vs Random Forest | <0,0001 | Sim (RF melhor) |
| Same-Package vs Random Forest | 0,0001 | Sim (**Same-Package melhor**) |

Hiperparâmetros escolhidos pelo GridSearchCV (`f1`, GroupKFold 5-fold por project+bug):
`n_estimators=100`, `max_depth=null`, `min_samples_leaf=1`, `min_samples_split=2`,
`class_weight=balanced`. F1 médio na validação cruzada: 0,0224.

AUC da validação cruzada no treino e no teste:

| Esquema | AUC |
|---|---:|
| StratifiedKFold(5) por instância | 0.6998 |
| GroupKFold(5) por bug | 0.6884 |
| GroupKFold(5) por project+bug | 0.6917 |
| Teste, por instância | 0.6316 |

Importância das features no modelo final:

| Feature | Importância |
|---|---:|
| `same_package` | 0.8096 |
| `modified_classes_count` | 0.1615 |
| `history` | 0.0289 |

## Ablação: o que o modelo acrescenta à heurística

`python3 -m src.ml.ablation` gera `results/ablation/`. Três resultados:

1. **A heurística `same_package` sozinha supera o Random Forest de três features**, com p de 0,0001,
   vencendo em 36 bugs, perdendo em 6 e empatando em 47.
2. **`modified_classes_count` não tem poder de ordenação nenhum.** Ela é constante dentro de cada um
   dos 297 bugs, porque conta as classes modificadas do bug e não do teste. Como o APFD depende só
   da ordem dos testes dentro de um bug, uma feature constante no bug nunca muda a posição relativa
   de dois testes. A importância de 0,16 mede o quanto ela ajuda a classificar, não a ordenar.
3. **Por isso o modelo tem, na prática, duas features úteis.** Um Random Forest treinado só com
   `same_package` reproduz a baseline Same-Package exatamente, nos 89 bugs, sem uma única diferença;
   e um treinado só com `history` reproduz a baseline History-based do mesmo modo. Sem
   `same_package`, o modelo cai de 0.7314 para 0.6139, o nível da History-based.

| Variante | APFD médio | Mediana |
|---|---:|---:|
| RF 3 features | 0.7314 | 0.8333 |
| RF só `same_package` (idêntico à baseline Same-Package) | 0.7412 | 0.8500 |
| RF sem `same_package` | 0.6139 | 0.6307 |
| RF só `history` (idêntico à baseline History-based) | 0.6112 | 0.6307 |

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
| `apfd_long_format.csv` | Resultados das quatro estratégias em formato longo |
| `descriptive_statistics.csv` | Estatísticas descritivas do APFD (geral e por projeto) |
| `wins_by_bug.csv` | APFD lado a lado por bug, com melhor/pior estratégia |
| `statistical_tests.csv` | Wilcoxon pareado entre as quatro estratégias, com vitórias e empates |
| `ablation/apfd_by_bug.csv` | APFD por bug de cada variante de features |
| `ablation/summary.csv` | Média, mediana e desvio por variante |
| `ablation/wilcoxon.csv` | Wilcoxon entre todas as variantes e baselines |

## Arquivos em `data/processed/`

| Arquivo | Conteúdo |
|---|---|
| `features.csv` | Base final: uma linha por par (bug, método de teste) |
| `bug_commit_dates.csv` | Data do commit de correção de cada bug, usada pela ordem cronológica |
| `dataset_build_report.csv` | Por bug: entrou na base ou não, e o motivo da falha |
