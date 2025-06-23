# evaluacion_recursiva.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import ARIMA
from itertools import product
from pmdarima import auto_arima
import optuna

import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Evita división por cero: solo considera valores donde y_true ≠ 0
    non_zero = y_true != 0

    # Calcula el MAPE sobre esos valores
    return (
        np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    )


def evaluate_regression_model_with_recursive_window(X, y, model, window_size=252, horizon=7):
    rmse_list, mae_list, mape_list = [], [], []
    last_forecast = None

    for start in range(0, len(y) - window_size - horizon + 1):
        end = start + window_size
        X_train, y_train = X[start:end], y[start:end]
        X_test, y_test = X[end:end+horizon], y[end:end+horizon]

        try:
            cloned_model = clone(model)
            cloned_model.fit(X_train, y_train)
            y_pred = cloned_model.predict(X_test)

            if any(np.isnan(y_pred)) or len(y_pred) != len(y_test):
                continue

            rmse_list.append(mean_squared_error(y_test, y_pred, squared=False))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            mape_list.append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
            last_forecast = y_pred
        except Exception:
            continue

    return {
        "model": model.__class__.__name__,
        "rmse": float(np.mean(rmse_list)) if rmse_list else np.inf,
        "mae": float(np.mean(mae_list)) if mae_list else np.inf,
        "mape": float(np.mean(mape_list)) if mape_list else np.inf,
        "forecast": last_forecast
    }

def evaluate_autoarima_with_recursive_window(series, window_size=252, horizon=7):
    rmse_list, mae_list, mape_list = [], [], []
    last_forecast = None

    for start in range(0, len(series) - window_size - horizon + 1, 8):
        train = series[:start+window_size]
        test = series[start+window_size:start+window_size+horizon]

        model = auto_arima(train, seasonal=False, suppress_warnings=True, error_action="ignore", m=1, stepwise=True, trace=False)
        forecast = model.predict(n_periods=horizon)
        forecast = np.where(forecast < 0, 0, forecast)

        rmse_list.append(np.sqrt(mean_squared_error(test, forecast)))
        mae_list.append(mean_absolute_error(test, forecast))
        mape_list.append(np.mean(np.abs((test - forecast) / test)) * 100)
        last_forecast = forecast

    return {
        "model": "AutoARIMA",
        "params": None,
        "rmse": float(np.mean(rmse_list)) if rmse_list else np.inf,
        "mae": float(np.mean(mae_list)) if mae_list else np.inf,
        "mape": float(np.mean(mape_list)) if mape_list else np.inf,
        "forecast": last_forecast
    }
def evaluate_arima_with_recursive_window(series, window_size=252, horizon=7):
    rmse_list, mae_list, mape_list = [], [], []
    last_forecast = None


    #print(series)

    best_result = None  # Para almacenar el mejor resultado
    result=[]
    # Prueba combinaciones de parámetros (p, d, q) desde 0 hasta 2
    for p, d, q in product(range(3), repeat=3):
        try:

            #print(len(series) - window_size  -horizon+ 1)
            #print(len(series))
            for start in range(0, len(series) - window_size -horizon + 1, 7):
                train = series[:start+window_size]
                test = series[start+window_size:start+window_size+horizon]
                #print(start+window_size)
                #print(start+window_size+horizon)
                #print(train)
                model =  ARIMA(train, order=(p, d, q)).fit() #auto_arima(train, seasonal=False, suppress_warnings=True, error_action="ignore", m=1, stepwise=True, trace=False)
                forecast = model.forecast(steps=7)
                forecast = np.where(forecast < 0, 0, forecast)
                # Calcula métricas de error
                mae = mean_absolute_error(test, forecast)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                mape = mean_absolute_percentage_error(test, forecast)
                last_forecast = forecast

                # Guarda configuración del modelo
                config = f"ARIMA({p},{d},{q})"
                #print(config)
                #print(rmse)

                result.append({
                    "model": config,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "forecast": forecast.tolist(),
                    "params" : (p,d,q)
                })
        except Exception as e:
                print(f'error{p},{d},{q}')
                continue  # o guarda el error y sigue con otros modelos
    #print(result)

    if not result:  # lista vacía
        return None

    resumen = pd.DataFrame(result).groupby(["model","params"])[["mae", "rmse", "mape"]].mean().reset_index()

    mejor_modelo = resumen.loc[resumen["rmse"].idxmin()]
    #print(mejor_modelo)
    return mejor_modelo

def evaluate_holtwinters_with_recursive_window(series, params, window_size=252, horizon=7, seasonal_periods=7):
    trend, seasonal = params['trend'], params['seasonal']
    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']

    rmse_list, mae_list, mape_list = [], [], []
    last_forecast = None

    #for start in range(0, len(series) - window_size - horizon + 1, 8):
    for start in range(0, len(series) - window_size -horizon + 1, 7):
        train = series[:start+window_size]
        test = series[start+window_size:start+window_size+horizon]

        #train = series[:start+window_size]
        #test = series[start+window_size:start+window_size+horizon]
        #print(start)
        #print(start+window_size)
        #print(start+window_size+horizon)


        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        ).fit(
            smoothing_level=alpha,
            smoothing_slope=beta,
            smoothing_seasonal=gamma,
            optimized=False
        )
        forecast = model.forecast(horizon)
        forecast = np.where(forecast < 0, 0, forecast)

        rmse_list.append(np.sqrt(mean_squared_error(test, forecast)))
        mae_list.append(mean_absolute_error(test, forecast))
        mape_list.append(np.mean(np.abs((test - forecast) / test)) * 100)
        last_forecast = forecast

    return {
        "model": "Holt-Winters",
        "params": params,
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "mape": float(np.mean(mape_list)),
        "forecast": last_forecast
    }

def optimize_holtwinters_with_optuna(series, seasonal_periods=7, n_trials=25):
    has_nonpositive = any(series <= 0)
    trend_choices = ["add"] if has_nonpositive else ["add", "mul"]
    seasonal_choices = ["add"] if has_nonpositive else ["add", "mul"]

    def objective(trial):
        params = {
            "trend": trial.suggest_categorical("trend", trend_choices),
            "seasonal": trial.suggest_categorical("seasonal", seasonal_choices),
            "alpha": trial.suggest_float("alpha", 0.05, 0.95),
            "beta": trial.suggest_float("beta", 0.05, 0.95),
            "gamma": trial.suggest_float("gamma", 0.05, 0.95),
        }
        result = evaluate_holtwinters_with_recursive_window(series, params, seasonal_periods=seasonal_periods)
        return result["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if len(study.trials) > 0 and study.best_trial.value is not None and not np.isnan(study.best_trial.value):
        return study.best_params
    else:
        print("⚠️  Error en estudio Optuna: No se encontró un modelo válido")
        return None
