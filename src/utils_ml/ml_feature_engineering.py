# ml_feature_engineering.py

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


def calcular_outliers_porcentaje(sku_data):
    """Calcula el porcentaje de outliers en una serie de datos."""
    if sku_data.empty:
        return 0.0
    q1 = sku_data.quantile(0.25)
    q3 = sku_data.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = sku_data[(sku_data < limite_inferior) | (sku_data > limite_superior)]
    porcentaje_outliers = (len(outliers) / len(sku_data)) * 100

    return porcentaje_outliers


def create_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Agrega variables temporales contextualizadas a partir de una columna de fecha.

    Parámetros:
    - df: DataFrame con una columna de fecha.
    - date_col: Nombre de la columna de fecha (por defecto 'date').

    Retorna:
    - DataFrame con nuevas columnas de variables temporales.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["day_of_week"] = df[date_col].dt.weekday
    df["day_of_month"] = df[date_col].dt.day
    df["is_weekend"] = df[date_col].dt.weekday.isin([5, 6]).astype(int)
    df["week_of_month"] = (df[date_col].dt.day - 1) // 7 + 1

    # Primeros 5 días del mes
    df["is_start_of_month"] = (df[date_col].dt.day <= 5).astype(int)

    # Proximidad a quincena: ±2 días del 15 o del 30/31
    df["is_near_quincena"] = df[date_col].dt.day.apply(
        lambda d: int(any(abs(d - q) <= 2 for q in [15, 30, 31]))
    )

    # Vacaciones escolares: julio y diciembre
    df["is_vacation"] = df[date_col].dt.month.isin([7, 12]).astype(int)

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    date_col: str = "fecha",
    max_daily_lag: int = 14,
    weekday_lags: int = 4,
) -> pd.DataFrame:
    """
    Crea columnas de lags diarios y lags del mismo día de la semana para la variable target, por grupo (SKU).

    Parámetros:
    - df: DataFrame con columnas de fecha y target.
    - target_col: Nombre de la columna objetivo (e.g., 'pedidos').
    - group_col: Nombre de la columna de agrupación (e.g., 'sku').
    - date_col: Nombre de la columna de fecha (por defecto 'fecha').
    - max_daily_lag: Número de lags diarios hacia atrás a crear.
    - weekday_lags: Número de lags semanales (del mismo día de la semana).

    Retorna:
    - DataFrame con columnas de lags agregadas.
    """
    df = df.copy()
    df = df.sort_values(by=[group_col, date_col])

    # Función para agregar lags por grupo
    def _add_lags(group):
        for lag in range(1, max_daily_lag + 1):
            group[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)
        for i in range(1, weekday_lags + 1):
            lag_days = i * 7
            group[f"{target_col}_weekday_lag_{i}"] = group[target_col].shift(lag_days)
        return group

    # Aplicar la función de lags por grupo
    df = df.groupby(group_col, group_keys=False).apply(_add_lags)
    return df


def create_rolling_features(
    df: pd.DataFrame, target_col: str, group_col: str, date_col: str = "fecha"
) -> pd.DataFrame:
    """
    Crea variables de tipo promedio móvil por grupo (SKU), excluyendo ceros.

    Promedios calculados:
    - Últimos 2 días
    - Últimos 7 días
    - Semana anterior completa (días -14 a -7)
    - Mismo día de la semana en las dos y tres semanas anteriores

    Parámetros:
    - df: DataFrame con columna target y columna de fecha.
    - target_col: Nombre de la columna objetivo.
    - date_col: Nombre de la columna de fecha.

    Retorna:
    - DataFrame con nuevas columnas de promedio móvil.
    """
    df = df.copy()
    df = df.sort_values(by=[group_col, date_col])

    def _add_rolling_features(group):
        # reemplazar ceros por NaN para evitar promedios erróneos
        target_no_zero = group[target_col].replace(0, np.nan)

        ## Promedios móviles
        group[f"{target_col}_rolling_2"] = (
            target_no_zero.shift(1).rolling(window=2, min_periods=1).mean()
        )

        group[f"{target_col}_rolling_7"] = (
            target_no_zero.shift(1).rolling(window=7, min_periods=1).mean()
        )

        group[f"{target_col}_prev_week_avg"] = (
            target_no_zero.shift(7).rolling(window=7, min_periods=1).mean()
        )

        # Promedio de los mismos días de la semana en las 2 semanas anteriores
        same_dow_lags_2wks = [7, 14]
        same_dow_values_2wks = [target_no_zero.shift(lag) for lag in same_dow_lags_2wks]
        group[f"{target_col}_dow_avg_2wks"] = pd.concat(
            same_dow_values_2wks, axis=1
        ).mean(axis=1)

        # Promedio de los mismos días de la semana en las 3 semanas anteriores
        same_dow_lags_3wks = [7, 14, 21]
        same_dow_values_3wks = [target_no_zero.shift(lag) for lag in same_dow_lags_3wks]
        group[f"{target_col}_dow_avg_3wks"] = pd.concat(
            same_dow_values_3wks, axis=1
        ).mean(axis=1)

        return group

    df = df.groupby(group_col, group_keys=False).apply(_add_rolling_features)
    return df


def create_stl_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    date_col: str = "fecha",
    seasonal: int = 7,
    stl_lags: int = 14,
) -> pd.DataFrame:
    """
    Aplica descomposición STL por grupo (SKU) y genera lags de los componentes.

    Parámetros:
    - df: DataFrame con fecha y target.
    - target_col: Columna de demanda.
    - group_col: Columna del SKU.
    - date_col: Columna de fecha.
    - seasonal: Periodo estacional esperado (7 = semanal).
    - stl_lags: Número de lags a generar para cada componente.

    Retorna:
    - DataFrame con columnas de tendencia y estacionalidad + sus lags.
    """
    df = df.copy()
    df = df.sort_values(by=[group_col, date_col])

    # Función para aplicar STL (descomposición) por grupo
    def _apply_stl(group):
        group = group.copy()
        series = group[target_col].fillna(0).values
        try:
            stl = STL(series, period=seasonal, robust=True)
            res = stl.fit()
            group["stl_trend"] = res.trend
            group["stl_seasonal"] = res.seasonal
        except Exception:
            group["stl_trend"] = np.nan
            group["stl_seasonal"] = np.nan
        return group

    df = df.groupby(group_col, group_keys=False).apply(_apply_stl)

    # Crear lags de cada componente
    for comp in ["stl_trend", "stl_seasonal"]:
        for lag in range(1, stl_lags + 1):
            df[f"{comp}_lag_{lag}"] = df.groupby(group_col)[comp].shift(lag)

    # eliminar el trend y la estacionalidad originales
    df.drop(columns=["stl_trend", "stl_seasonal"], inplace=True)

    return df
