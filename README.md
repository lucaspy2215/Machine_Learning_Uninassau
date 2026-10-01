# Classificação de Iris

Projeto da disciplina **Aprendizagem de Máquina** (Uninassau).

Classifica as espécies de flores Iris (*setosa*, *versicolor*, *virginica*) a partir das medidas de sépala e pétala, comparando dois modelos supervisionados:

- Árvore de Decisão (`DecisionTreeClassifier`)
- Regressão Logística (`LogisticRegression`)

## O que o script faz

1. Carrega o dataset `iris_alterado.csv` e renomeia as colunas
2. Identifica e remove linhas com dados faltantes
3. Codifica as espécies e cria categorias (pequeno, médio, grande)
4. Divide em 80% treino e 20% teste (estratificado)
5. Treina e avalia os dois modelos
6. Gera gráficos: árvore de decisão, pairplot e boxplots

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

## Tecnologias

Python, pandas, scikit-learn, matplotlib, seaborn.

## Autor

Lucas Rocha de Farias