import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL


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
