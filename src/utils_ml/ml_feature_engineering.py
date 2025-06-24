# ml_feature_engineering.py

import pandas as pd
import numpy as np


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

    def _add_lags(group):
        for lag in range(1, max_daily_lag + 1):
            group[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)
        for i in range(1, weekday_lags + 1):
            lag_days = i * 7
            group[f"{target_col}_weekday_lag_{i}"] = group[target_col].shift(lag_days)
        return group

    df = df.groupby(group_col, group_keys=False).apply(_add_lags)
    return df


def create_rolling_features(
    df: pd.DataFrame, target_col: str, group_col: str, date_col: str = "date"
) -> pd.DataFrame:
    """
    Crea variables de tipo promedio móvil por grupo (SKU), excluyendo ceros.

    Promedios calculados:
    - Últimos 2 días
    - Últimos 7 días
    - Semana anterior completa (días -14 a -7)
    - Mismo día de la semana en las dos semanas anteriores

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
        target_no_zero = group[target_col].replace(0, np.nan)

        group[f"{target_col}_rolling_2"] = (
            target_no_zero.shift(1).rolling(window=2, min_periods=1).mean()
        )

        group[f"{target_col}_rolling_7"] = (
            target_no_zero.shift(1).rolling(window=7, min_periods=1).mean()
        )

        group[f"{target_col}_prev_week_avg"] = (
            target_no_zero.shift(7).rolling(window=7, min_periods=1).mean()
        )

        same_dow_lags = [7, 14]
        same_dow_values = []

        for lag in same_dow_lags:
            same_dow_values.append(target_no_zero.shift(lag))

        group[f"{target_col}_dow_avg_2wks"] = pd.concat(same_dow_values, axis=1).mean(
            axis=1
        )

        return group

    df = df.groupby(group_col, group_keys=False).apply(_add_rolling_features)
    return df
