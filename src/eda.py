# src/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def basic_info(df: pd.DataFrame):
    """Exibe informações básicas do DataFrame."""
    print("\nInformações do DataFrame:")
    print(df.info())
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nEstatísticas descritivas:")
    print(df.describe())

def missing_values(df: pd.DataFrame):
    """Exibe a contagem e proporção de valores ausentes."""
    print("\nValores ausentes por coluna:")
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Total Missing': missing, '%': percent})
    print(missing_df[missing_df['Total Missing'] > 0])

def plot_distributions(df: pd.DataFrame, numeric_cols: list):
    """Plota a distribuição de variáveis numéricas."""
    print("\nPlotando distribuições numéricas...")
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')
        plt.tight_layout()
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: list):
    """Plota a matriz de correlação entre variáveis numéricas."""
    print("\nMatriz de correlação:")
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.show()
