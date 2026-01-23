import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def describe_df(df: pd.DataFrame):
    """
    Genera un resumen estadístico detallado del DataFrame.
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_pct': df.isnull().mean() * 100,
        'unique_values': df.nunique(),
        'cardinality_pct': df.nunique() / len(df) * 100
    })
    return summary.T


def get_features_num_regression(df, target_col, corr_threshold=0.1):
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
    Sugiere el tipo de variable según su cardinalidad.
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
    Genera scatter plots entre cada variable numérica y la variable objetivo.
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
    Devuelve columnas categóricas significativas.
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