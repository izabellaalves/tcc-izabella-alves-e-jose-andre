# TCC

Pipeline para priorização de testes com base em bugs do Defects4J (Lang e Chart).

Atualmente, o projeto está preparado para a etapa de **feature engineering**: checkout dos bugs, extração de features e geração do dataset em `data/processed/features.csv`.

As próximas etapas (treinamento de modelos, baselines, métricas e visualização) ainda não estão implementadas, mas a estrutura em `src/` já está organizada para recebê-las.

## Como rodar

**Pré-requisitos:** Docker e Docker Compose.

```bash
docker-compose build
docker-compose up tcc-pipeline
```

O resultado será salvo em `data/processed/features.csv`.
