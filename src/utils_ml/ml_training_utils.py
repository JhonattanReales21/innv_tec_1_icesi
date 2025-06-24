import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

import optuna
from typing import Callable, Dict, Any
from sklearn.base import RegressorMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import ast


def trans_pred(predicciones):
    """
    Transforma las predicciones negativas en cero.

    """
    return np.maximum(predicciones, 0)


def smape(y_true, y_pred):
    """
    Calcula el SMAPE (Symmetric Mean Absolute Percentage Error).

    Parámetros:
    - y_true: Valores reales.
    - y_pred: Valores predichos.

    Retorna:
    - SMAPE como porcentaje.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Evita división por cero
    return 200 * np.mean(diff)


def mape_mod(y_true, y_pred):
    """
    Calcula el MAPE modificado (Mean Absolute Percentage Error).

    Parámetros:
    - y_true: Valores reales.
    - y_pred: Valores predichos.

    Retorna:
    - MAPE modificado como porcentaje.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Evita división por cero: solo considera valores donde y_true ≠ 0
    non_zero = y_true != 0

    # Calcula el MAPE sobre esos valores
    return (
        np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    )


def evaluate_model_with_recursive_window(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    val_iter: int = 3,
    start_train: int = 0,
    train_size: int = 100,
    val_size: int = 7,
) -> tuple:
    """
    Evalúa un modelo usando ventana recursiva.

    Parámetros:
    - model: Modelo ya inicializado (con .fit() y .predict()).
    - X: Variables predictoras ordenadas por fecha.
    - y: Variable objetivo ordenada por fecha.
    - val_iter: Número de ventanas de evaluación.
    - start_train: Índice inicial para la ventana de entrenamiento.
    - train_size: Tamaño de la ventana de entrenamiento.
    - val_size: Tamaño de la ventana de validación.

    Retorna:
    - SMAPE promedio en validación.
    - GAP promedio (|SMAPE_train - SMAPE_val|).
    """
    smape_vals = []
    gap_vals = []

    for i in range(val_iter):
        # Define los índices de inicio y fin para entrenamiento y validación
        step = i * val_size
        end_train = step + train_size
        end_val = end_train + val_size

        # print(
        #     f"Iteración {i + 1}: Entrenamiento de {start_train} a {end_train}, Validación de {end_train} a {end_val}"
        # )

        if end_val > len(X):
            break  # Evita salir de rango

        # Divide los datos en entrenamiento y validación
        X_train, y_train = X.iloc[start_train:end_train], y.iloc[start_train:end_train]
        X_val, y_val = X.iloc[end_train:end_val], y.iloc[end_train:end_val]

        # print(
        #     f"Tamaño de entrenamiento: {len(X_train)} , Tamaño de validación: {len(X_val)}"
        # )

        # Entrena el modelo y realiza predicciones
        model.fit(X_train, y_train)
        y_pred_val = trans_pred(model.predict(X_val))
        y_pred_train = trans_pred(model.predict(X_train))

        # calcula el smape y el gap
        smape_val = smape(y_val, y_pred_val)
        smape_train = smape(y_train, y_pred_train)
        gap = np.abs(smape_train - smape_val)

        # print(
        #     f"SMAPE Entrenamiento: {smape_train:.4f}, SMAPE Validación: {smape_val:.4f}, GAP: {gap:.4f}"
        # )

        # Almacena los resultados
        smape_vals.append(smape_val)
        gap_vals.append(gap)

    # Calcula el SMAPE promedio y el GAP promedio
    avg_smape = np.mean(smape_vals)
    avg_gap = np.mean(gap_vals)

    return avg_smape, avg_gap


def optimize_model_with_optuna(
    model_class: Callable[..., RegressorMixin],
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    val_iter: int = 3,
    start_train: int = 0,
    train_size: int = 100,
    val_size: int = 7,
    show_progress_bar: bool = True,
    random_seed: int = 42,
) -> optuna.study.Study:
    """
    Optimiza hiperparámetros con Optuna usando evaluación multiobjetivo: SMAPE y GAP.

    Parámetros:
    - model_class: Clase del modelo (ej: XGBRegressor, CatBoostRegressor).
    - param_grid: Diccionario con los espacios de búsqueda (usar trial.suggest_...).
    - X, y: Datos de entrada ya ordenados por fecha.
    - n_trials: Número de iteraciones.
    - val_iter, start_train, train_size, val_size: Parámetros para la evaluación recursiva.
    - show_progress_bar: Mostrar barra de progreso de Optuna.
    - random_seed: Semilla para reproducibilidad.

    Retorna:
    - study: Objeto de estudio Optuna.
    """

    def objective(trial: optuna.Trial):
        # Construcción dinámica de hiperparámetros
        trial_params = {}
        for param_name, suggestion in param_grid.items():
            trial_params[param_name] = suggestion(trial)

        model = model_class(**trial_params)

        avg_smape, avg_gap = evaluate_model_with_recursive_window(
            model=model,
            X=X,
            y=y,
            val_iter=val_iter,
            start_train=start_train,
            train_size=train_size,
            val_size=val_size,
        )

        return round(avg_smape, 2), round(avg_gap, 2)

    # Crear estudio multiobjetivo
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )

    # Ejecutar optimización
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    # Mostrar mejores resultados
    print("\n📌 Mejores Trials:")
    for i, trial in enumerate(study.best_trials):
        if i < 5:
            print(
                f"Trial {i} - SMAPE: {trial.values[0]:.4f}, GAP: {trial.values[1]:.4f}"
            )

    print("\nNúmero total de Best Trials ('Frente de pareto'):", len(study.best_trials))
    print("\n🏆 Best Trial Params:")
    print(study.best_trials[0].params)

    # print(
    #     f"\n💡 Best Values: SMAPE={study.best_trials[0].values[0]:.4f}, GAP={study.best_trials[0].values[1]:.4f}"
    # )

    return study


def evaluate_model_test(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """
    Evalúa un modelo en el conjunto de prueba y calcula SMAPE y GAP.

    Parámetros:
    - model: Modelo ya entrenado.
    - X_test: Variables predictoras del conjunto de prueba.
    - y_test: Variable objetivo del conjunto de prueba.

    Retorna:
    - SMAPE en el conjunto de prueba.
    - GAP entre entrenamiento y prueba.
    """

    return "f"


def evaluate_models_on_test(
    df: pd.DataFrame,
    df_best_models: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    sku_col: str = "sku",
    test_size: int = 7,
) -> pd.DataFrame:
    """
    Reentrena y evalúa el mejor modelo por SKU en el set de test (últimos test_size días).

    Parámetros:
    - df: DataFrame completo con las series.
    - df_best_models: DataFrame con columnas 'sku', 'model', 'best_params'.
    - target_col: Nombre de la columna objetivo.
    - feature_cols: Lista de variables predictoras.
    - sku_col: Columna identificadora del SKU.
    - test_size: Número de días para el test set final.

    Retorna:
    - DataFrame con métricas por SKU y predicciones.
    """

    resultados_test = []

    for sku in df_best_models[sku_col].unique():
        print(f"\n🔍 Reentrenando y evaluando en test para SKU: {sku}")

        df_sku = df[df[sku_col] == sku].copy()
        df_test = df_sku[-test_size:].copy()
        df_sku = df_sku[:-test_size]

        X_train = df_sku[feature_cols]
        y_train = df_sku[target_col]
        X_test = df_test[feature_cols]
        y_test = df_test[target_col]

        model_name = df_best_models.loc[df_best_models[sku_col] == sku, "model"].values[
            0
        ]
        raw_params = df_best_models.loc[
            df_best_models[sku_col] == sku, "best_params"
        ].values[0]
        parsed_params = ast.literal_eval(raw_params)

        # Limpieza para pipeline
        if model_name in ["ElasticNet", "KNeighborsRegressor"]:
            best_params = {
                k.replace("model__", ""): v
                for k, v in parsed_params.items()
                if k.startswith("model__")
            }
        else:
            best_params = parsed_params

        # Inicializamos el modelo
        if model_name == "XGBRegressor":
            model = XGBRegressor(**best_params)
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(**best_params)
        elif model_name == "ElasticNet":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(**best_params)),
                ]
            )
        elif model_name == "KNeighborsRegressor":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor(**best_params)),
                ]
            )
        else:
            raise ValueError(f"Modelo desconocido: {model_name}")

        # Entrenar y predecir
        model.fit(X_train, y_train)
        y_pred = trans_pred(model.predict(X_test))

        # Métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mape_mod(y_test, y_pred)
        smape_val = smape(y_test, y_pred)

        resultados_test.append(
            {
                "SKU": sku,
                "Modelo": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "SMAPE": smape_val,
                "forecast": y_pred.tolist(),
            }
        )

    return pd.DataFrame(resultados_test).sort_values(by="SKU")
