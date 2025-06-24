import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

import optuna
from typing import Callable, Dict, Any
from sklearn.base import RegressorMixin


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
        model.fit(X_train, y_train, njobs=-1, verbose=False)
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
        print(f"Trial {i} - SMAPE: {trial.values[0]:.4f}, GAP: {trial.values[1]:.4f}")

    print("\n🏆 Best Trial Params:")
    print(study.best_trials[0].params)

    # print(
    #     f"\n💡 Best Values: SMAPE={study.best_trials[0].values[0]:.4f}, GAP={study.best_trials[0].values[1]:.4f}"
    # )

    return study
