# evaluacion_recursiva.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import optuna
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_regression_model_with_recursive_window(X, y, model, window_size=210, horizon=7):
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

def evaluate_autoarima_with_recursive_window(series, window_size=210, horizon=7):
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

def evaluate_holtwinters_with_recursive_window(series, params, window_size=210, horizon=7, seasonal_periods=7):
    trend, seasonal = params['trend'], params['seasonal']
    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']

    rmse_list, mae_list, mape_list = [], [], []
    last_forecast = None

    for start in range(0, len(series) - window_size - horizon + 1, 8):
        train = series[:start+window_size]
        test = series[start+window_size:start+window_size+horizon]

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
