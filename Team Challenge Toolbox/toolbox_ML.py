import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def describe_df(df: pd.DataFrame):
    """
     Devuelve un resumen completo del DataFrame incluyendo:
    - Tipo de dato
    - Número de nulos
    - Porcentaje de nulos
    - Número de valores únicos
    - Estadísticas descriptivas (si es numérica)
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_pct': df.isnull().mean() * 100,
        'unique_values': df.nunique(),
        'cardinality_pct': df.nunique() / len(df) * 100
    })
    return summary.T


def get_features_num_regression(df, target_col, corr_threshold=0.1):
    """
     Devuelve una lista de variables numéricas cuya correlación con el target
    es mayor en valor absoluto que umbral_corr.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la variable objetivo numérica.
    umbral_corr (float): Umbral mínimo de correlación (0-1).
    pvalue (float, optional): Nivel de significación estadística.

    Retorna:
    list: Lista de columnas numéricas que cumplen los criterios.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if target_col not in num_cols:
        return []

    num_cols.remove(target_col)
    selected_features = []

    for col in num_cols:
        corr = df[col].corr(df[target_col])
        if abs(corr) >= corr_threshold:
            selected_features.append(col)

    return selected_features


def tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float):
    """
     Clasifica las variables del DataFrame en:
    - numerica_continua
    - numerica_discreta
    - categorica
    - binaria

    Args:
        df (pd.DataFrame)
        umbral_categoria (int): número máximo de valores únicos
                                 para considerar una variable numérica como discreta

    Returns:
        pd.DataFrame con el tipo asignado
    """
    resultados = []

    for col in df.columns:
        n_unicos = df[col].nunique()
        pct_cardinalidad = n_unicos / len(df) * 100

        if n_unicos == 2:
            tipo = "Binaria"
        elif n_unicos < umbral_categoria:
            tipo = "Categórica"
        else:
            if pct_cardinalidad >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"

        resultados.append({
            'nombre_variable': col,
            'tipo_sugerido': tipo
        })

    return pd.DataFrame(resultados)


def plot_features_num_regression(df, features, target_col):
    """
     Genera pairplots entre target y variables numéricas
    que cumplan el umbral de correlación y significación.

    Retorna:
    list: columnas que cumplen condiciones.
    
    """
    for col in features:
        plt.figure()
        plt.scatter(df[col], df[target_col])
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.title(f"{col} vs {target_col}")
        plt.show()


def get_features_cat_regression(df, target_col, cardinality_threshold=0.1):
    """
   Devuelve variables categóricas cuya relación con target
    sea estadísticamente significativa.

    Argumentos:
    df (pd.DataFrame)
    target_col (str)
    pvalue (float)

    Retorna:
    list: columnas categóricas que cumplen condiciones.
    
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    selected_features = []

    for col in cat_cols:
        card = df[col].nunique() / len(df)
        if card <= cardinality_threshold:
            selected_features.append(col)

    return selected_features


def plot_features_cat_regression(df, features, target_col):
    """
    Genera boxplots entre variables categóricas y la variable objetivo.
    """
    for col in features:
        plt.figure()
        sns.boxplot(x=df[col], y=df[target_col])
        plt.title(f"{col} vs {target_col}")
        plt.show()