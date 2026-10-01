"""
Classificação de espécies de Iris
Disciplina: Aprendizagem de Máquina (Uninassau)

Etapas:
1. Carga e preparação dos dados
2. Divisão em treino e teste
3. Árvore de Decisão
4. Regressão Logística
5. Visualizações exploratórias
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
URL = (
    "https://github.com/danielmarquesvg/Uninassau_AprendizagemDeMaquina_20262"
    "/raw/refs/heads/main/dados/iris_alterado.csv"
)

COLUNAS = ["comp_sepala", "larg_sepala", "comp_petala", "larg_petala"]
ESPECIES = {"setosa": 1, "versicolor": 2, "virginica": 3}
NOMES = {v: k for k, v in ESPECIES.items()}

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


# ---------------------------------------------------------------------------
# 1. Carga e preparação
# ---------------------------------------------------------------------------
def carregar_dados(url: str) -> pd.DataFrame:
    dados = pd.read_csv(url)
    return dados.rename(
        columns={
            "Id": "id",
            "SepalLengthCm": "comp_sepala",
            "SepalWidthCm": "larg_sepala",
            "PetalLengthCm": "comp_petala",
            "PetalWidthCm": "larg_petala",
            "Species": "especie",
        }
    )


def preparar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    print(dados)

    # Linhas com dados faltantes
    print("\nLinhas com dados incompletos:")
    print(dados[dados.isna().any(axis=1)])

    # Codifica a espécie como número e cria a coluna com o nome
    dados["especie"] = dados["especie"].map(ESPECIES)
    dados["especie_nome"] = dados["especie"].map(NOMES)

    # O dataset é "alterado": remove linhas incompletas para os modelos funcionarem
    dados = dados.dropna(subset=COLUNAS + ["especie"]).copy()
    dados["especie"] = dados["especie"].astype(int)

    print("\nContagem por espécie:")
    print(dados["especie"].value_counts())

    # Categorias em texto (pequeno / médio / grande)
    for col in COLUNAS:
        dados[col + "_cat"] = pd.cut(
            dados[col], bins=3, labels=["pequeno", "médio", "grande"]
        )

    print(dados.head())
    return dados


# ---------------------------------------------------------------------------
# 2. Treino e teste
# ---------------------------------------------------------------------------
def dividir_dados(dados: pd.DataFrame):
    X = dados[COLUNAS]
    y = dados["especie"]
    print("X:", X.shape, "| y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,   # 20% teste / 80% treino
        random_state=42,  # partição reproduzível
        stratify=y,       # mantém a proporção das espécies
    )
    print("Treino:", X_train.shape, "| Teste:", X_test.shape)
    print("\nTreino por espécie:\n", y_train.value_counts())
    print("\nTeste por espécie:\n", y_test.value_counts())
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. Árvore de Decisão
# ---------------------------------------------------------------------------
def treinar_arvore(X_train, X_test, y_train, y_test):
    arvore = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
    arvore.fit(X_train, y_train)

    y_pred = arvore.predict(X_test)
    print(f"\nAcurácia da Árvore de Decisão: {accuracy_score(y_test, y_pred):.4f}")

    plt.figure(figsize=(25, 10))
    plot_tree(
        arvore,
        feature_names=X_train.columns,
        class_names=list(ESPECIES.keys()),
        filled=True,
        rounded=True,
        fontsize=14,
    )
    plt.show()
    return arvore


# ---------------------------------------------------------------------------
# 4. Regressão Logística
# ---------------------------------------------------------------------------
def treinar_regressao_logistica(X_train, X_test, y_train, y_test):
    print("\nTreinando o modelo de Regressão Logística...")
    modelo = LogisticRegression(max_iter=200, random_state=42)
    modelo.fit(X_train, y_train)

    print("Realizando predições no conjunto de teste...")
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== Resultados da Avaliação ===")
    print(f"Acurácia no conjunto de teste: {acc * 100:.2f}%\n")
    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=list(ESPECIES.keys())))
    return modelo


# ---------------------------------------------------------------------------
# 5. Visualizações
# ---------------------------------------------------------------------------
def visualizar(dados: pd.DataFrame):
    sns.pairplot(dados, vars=COLUNAS, hue="especie", height=2)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(data=dados, y="comp_sepala", x="especie_nome", ax=axes[0, 0])
    sns.boxplot(data=dados, y="comp_petala", x="especie_nome", ax=axes[0, 1])
    sns.boxplot(data=dados, y="larg_sepala", x="especie_nome", ax=axes[1, 0])
    sns.boxplot(data=dados, y="larg_petala", x="especie_nome", ax=axes[1, 1])
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def main():
    dados = preparar_dados(carregar_dados(URL))
    X_train, X_test, y_train, y_test = dividir_dados(dados)
    treinar_arvore(X_train, X_test, y_train, y_test)
    treinar_regressao_logistica(X_train, X_test, y_train, y_test)
    visualizar(dados)


if __name__ == "__main__":
    main()