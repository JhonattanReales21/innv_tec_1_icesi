import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
import optuna.visualization as vis
import matplotlib.pyplot as plt


def graficar_serie_con_descomposicion(df_sku, sku, periodo=7):
    """
    Grafica la serie de tiempo de pedidos junto con su tendencia y estacionalidad
    usando la descomposición STL. La tendencia y estacionalidad se grafican en un eje secundario.

    Parámetros:
    - df_sku: DataFrame con columnas 'fecha' y 'pedidos'
    - sku: identificador del SKU (para el título del gráfico)
    - periodo: periodicidad estacional (por defecto 7 para datos semanales)
    """

    # Aplicar descomposición STL
    stl = STL(df_sku["pedidos"], period=periodo)
    result = stl.fit()

    # Añadir columnas al DataFrame
    df_sku["tendencia"] = result.trend
    df_sku["estacionalidad"] = result.seasonal

    # Crear figura
    fig = go.Figure()

    # Pedidos (eje primario)
    fig.add_trace(
        go.Scatter(
            x=df_sku["fecha"],
            y=df_sku["pedidos"],
            mode="lines",
            name="Pedidos",
            line=dict(width=2),
            yaxis="y1",
        )
    )

    # Tendencia (eje secundario)
    fig.add_trace(
        go.Scatter(
            x=df_sku["fecha"],
            y=df_sku["tendencia"],
            mode="lines",
            name="Tendencia",
            line=dict(width=2, dash="dash"),
            yaxis="y2",
        )
    )

    # Estacionalidad (eje secundario)
    fig.add_trace(
        go.Scatter(
            x=df_sku["fecha"],
            y=df_sku["estacionalidad"],
            mode="lines",
            name="Estacionalidad",
            line=dict(width=2, dash="dot"),
            yaxis="y2",
        )
    )

    # Layout del gráfico
    fig.update_layout(
        title=f"Serie de tiempo de Pedidos para {sku} con Tendencia y Estacionalidad",
        xaxis=dict(title="Fecha", showgrid=True, gridcolor="LightGray"),
        yaxis=dict(title="Pedidos", showgrid=True, gridcolor="LightGray"),
        yaxis2=dict(
            title="Tendencia / Estacionalidad",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99),
        template="simple_white",
    )

    fig.show()


def mostrar_graficas_optuna(study, model_name="Modelo"):
    """
    Genera y muestra múltiples visualizaciones del estudio de Optuna:
    - Frontera de Pareto
    - Importancia de parámetros
    - Coordenadas paralelas (para SMAPE)
    - Historia de optimización (para GAP)

    Se asume que el estudio es multiobjetivo con `values[0]` = SMAPE y `values[1]` = GAP.
    """
    try:
        print("📊 Mostrando gráfica de frontera de Pareto...")
        vis.plot_pareto_front(study, target_names=["SMAPE", "GAP"]).show()
    except Exception as e:
        print(f"❌ Error al mostrar la gráfica de Pareto: {e}")

    try:
        print("📊 Mostrando importancia de parámetros...")
        vis.plot_param_importances(study).show()
    except Exception as e:
        print(f"❌ Error al mostrar importancia de parámetros: {e}")

    try:
        if model_name == "XGBRegressor":
            print("📊 Mostrando coordenadas paralelas (SMAPE)...")
            vis.plot_parallel_coordinate(
                study,
                params=[
                    "max_depth",
                    "learning_rate",
                    "n_estimators",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                    "min_child_weight",
                ],
                target=lambda trial: trial.values[0],
                target_name="SMAPE",
            ).show()

        elif model_name == "RandomForestRegressor":
            print("📊 Mostrando coordenadas paralelas (SMAPE)...")
            vis.plot_parallel_coordinate(
                study,
                params=[
                    "n_estimators",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "max_features",
                ],
                target=lambda trial: trial.values[0],
                target_name="SMAPE",
            ).show()
    except Exception as e:
        print(f"❌ Error al mostrar coordenadas paralelas: {e}")

    try:
        print("📊 Mostrando historia de optimización (GAP)...")
        vis.plot_optimization_history(
            study,
            target=lambda trial: trial.values[1],
            target_name="GAP",
        ).show()
    except Exception as e:
        print(f"❌ Error al mostrar historia de optimización: {e}")
