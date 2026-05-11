"""
Módulo para visualizaciones interactivas con Plotly
"""
import base64

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import io
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import statsmodels.api as sm


def create_dashboard_layout():
    """
    Devuelve un diccionario con estilos comunes para gráficas Plotly,
    con paleta al estilo Data México (oscuro, naranja, blanco).
    """
    layout = dict(
        # Tema general
        template="plotly_dark",  # 👈 Activa el modo oscuro

        # Colores de fondo
        paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente del papel
        plot_bgcolor='rgba(0,0,0,0)',   # Fondo transparente del área de la gráfica

        # Colores del texto
        font=dict(color="#C5CAE9"),  # Gris claro para texto (como en Data México)

        # Colores de los ejes
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',  # Grilla sutil
            color="#C5CAE9"  # Color del eje X
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',  # Grilla sutil
            color="#C5CAE9"  # Color del eje Y
        ),

        # Hover
        hovermode="x unified",  # Muestra info de todas las líneas en un punto X
        hoverlabel=dict(
            bgcolor="rgba(10, 25, 47, 0.9)",  # Fondo del hover
            font_color="white"  # Color del texto del hover
        ),

        # Margen
        margin=dict(l=50, r=30, t=50, b=50)
    )
    return layout


def plot_correlation_heatmap(df, title="Matriz de Correlación"):
    """Mapa de calor de correlaciones"""
    corr = df.select_dtypes(include='number').corr()
    
    fig = px.imshow(
        corr, 
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title=title
    )
    fig.update_layout(**create_dashboard_layout())
    return fig


def plot_time_series(df, columns=None, title="Serie Temporal",
                    yaxis_title="Valor", show_legend=True, 
                    secondary_y=None):
    """
    Gráfica de líneas para series temporales múltiples con soporte de eje secundario.
    
    Args:
        df: DataFrame con series temporales
        columns: Lista de columnas a graficar (None = todas numéricas)
        title: Título de la gráfica
        yaxis_title: Título del eje Y principal
        show_legend: Mostrar leyenda
        secondary_y: Lista de columnas para eje Y secundario (None = detectar automáticamente)
    """
    cols = columns or df.select_dtypes(include='number').columns
    fig = go.Figure()
    
    # Detectar automáticamente series para eje secundario si no se especifica
    if secondary_y is None:
        # Detectar series con escala muy diferente (ratio > 10x)
        median_values = df[cols].median()
        primary_series = []
        secondary_series = []
        
        for col in cols:
            if col in df.columns:
                median_val = median_values[col]
                # Si la mediana es > 10x la mínima, usar eje secundario
                if median_val > 10 * median_values.min() or median_val < median_values.max() / 10:
                    secondary_series.append(col)
                else:
                    primary_series.append(col)
        
        secondary_y = secondary_series if secondary_series else None
    else:
        primary_series = [c for c in cols if c not in secondary_y]
        secondary_series = secondary_y
    
    # Graficar series primarias (eje Y izquierdo)
    for col in primary_series:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[col],
                name=col,
                mode='lines',
                line=dict(width=2),
                yaxis='y'  # Eje principal
            ))
    
    # Graficar series secundarias (eje Y derecho)
    for col in secondary_series:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[col],
                name=col,
                mode='lines',
                line=dict(width=2, dash='dash'),
                yaxis='y2'  # Eje secundario
            ))
    
    # Configurar layout con dos ejes Y
    layout_config = create_dashboard_layout()
    
    if secondary_series:
        # Agregar configuración de eje secundario
        layout_config.update({
            'yaxis2': dict(
                title=yaxis_title + " (Secundario)",
                overlaying='y',
                side='right',
                gridcolor='rgba(255, 255, 255, 0.05)',  # Grilla más sutil
                color="#FF9800"  # Naranja para diferenciar
            ),
            'legend': dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        })
        
        # Agregar anotación para aclarar ejes
        fig.add_annotation(
            text=" Eje Izquierdo: Series primarias | Eje Derecho: Series secundarias",
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            showarrow=False,
            font=dict(size=10, color="#C5CAE9")
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=yaxis_title,
        showlegend=show_legend,
        **layout_config
    )
    
    return fig


def plot_indicators_cards(df, metrics_dict):
    """Tarjetas de KPIs para dashboard"""
    cards = []
    
    for label, (value, suffix, delta) in metrics_dict.items():
        card = go.Figure()
        card.add_trace(go.Indicator(
            mode="number+delta",
            value=value,
            title={'text': label},
            delta={'reference': delta, 'suffix': suffix} if delta else {'suffix': suffix},
            number={'prefix': '$' if '$' in suffix else '', 
                   'suffix': suffix.replace('$','')}
        ))
        card.update_layout(height=150, margin=dict(t=20, b=20, l=20, r=20))
        cards.append(card)
    
    return cards


def plot_regression_results(X, y, predictions, residuals=None, var_name="Variable"):
    """Visualización de resultados de regresión"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Ajuste del Modelo", "Residuos"],
        horizontal_spacing=0.1
    )
    
    # Gráfica de ajuste
    fig.add_trace(
        go.Scatter(x=X, y=y, mode='markers', name='Observado',
                  marker=dict(color='blue', opacity=0.6)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=X, y=predictions, mode='lines', name='Predicho',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Gráfica de residuos
    if residuals is not None:
        fig.add_trace(
            go.Scatter(x=predictions, y=residuals, mode='markers',
                      name='Residuos', marker=dict(color='gray')),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dot", line_color="red", row=1, col=2)
    
    fig.update_layout(
        title=f"Análisis de Regresión - {var_name}",
        height=400,
        showlegend=True,
        **create_dashboard_layout()
    )
    
    return fig

def plot_log_scatter(df: pd.DataFrame, y_col: str, x_col: str = None):
    """Gráfica de dispersión ln(Y) vs ln(X) con ajuste."""
    if x_col is None:
        x_col = [c for c in df.columns if 'ln_' in c and c != y_col][0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col],
                             mode='markers', name='Datos',
                             marker=dict(size=6, color='blue')))
    
    # Ajuste lineal simple para ilustrar
    mask = df[[x_col, y_col]].notna().all(axis=1)
    x = df.loc[mask, x_col].values
    y = df.loc[mask, y_col].values
    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)
    fig.add_trace(go.Scatter(x=x, y=y_fit, mode='lines', name='Ajuste', line=dict(color='red')))

    fig.update_layout(
        title=f"{y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        **create_dashboard_layout()
    )
    return fig
    pass

def plot_residuals(residuals: np.ndarray, index):
    """Gráfica de residuos vs tiempo y QQ-plot profesional."""
    from statsmodels.graphics import gofplots
    
    # 1. Crear subplots con proporciones iguales
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=["Residuos vs Tiempo", "QQ-Plot"],
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": False}, {"secondary_y": False}]] # Evita conflictos de ejes
    )
    
    # 2. Panel Izquierdo: Residuos
    fig.add_trace(
        go.Scatter(
            x=index, y=residuals, 
            mode='lines+markers', 
            name='Residuos',
            marker=dict(size=4, color='#64B5F6')
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # 3. Panel Derecho: QQ-Plot (Generado manualmente para control total)
    # Calculamos cuantiles teóricos y muestrales
    qq = sm.ProbPlot(residuals, fit=True)
    theoretical = qq.theoretical_quantiles
    sample = qq.sample_quantiles
    
    # Línea de referencia 45 grados
    min_q = min(theoretical.min(), sample.min())
    max_q = max(theoretical.max(), sample.max())
    
    fig.add_trace(
        go.Scatter(
            x=theoretical, y=sample,
            mode='markers',
            name='Cuantiles',
            marker=dict(size=6, color='#FF6B35'),
            text=[f"Teórico: {t:.2f}<br>Muestral: {s:.2f}" for t, s in zip(theoretical, sample)],
            hoverinfo='text+x+y'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[min_q, max_q], y=[min_q, max_q],
            mode='lines',
            name='Referencia',
            line=dict(color='gray', width=1, dash='dash')
        ),
        row=1, col=2
    )

    # 4. Layout Final y Configuración
    layout_base = create_dashboard_layout()
    layout_base['margin'] = dict(l=60, r=40, t=60, b=60)
    layout_base['height'] = 450
    
    # Actualizamos el layout
    fig.update_layout(**layout_base)
    
    # Ajuste manual del eje X del segundo panel para que sea cuadrado
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=2)
    fig.update_yaxes(row=1, col=2, scaleanchor="x", scaleratio=1) # Fuerza proporción 1:1
    
    return fig

def plot_forecast(actual: pd.Series, forecast: np.ndarray, conf_int=None, title: str = "Pronóstico") -> go.Figure:
    """
    Genera gráfica interactiva de pronóstico con Plotly.

    Args:
        actual: Serie histórica (pd.Series con índice datetime)
        forecast: Valores pronosticados (np.ndarray o list)
        conf_int: DataFrame con columnas 'lower' y 'upper' (opcional)
        title: Título de la gráfica

    Returns:
        fig: Figura de Plotly
    """
    fig = go.Figure()

    # 1. Serie histórica
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode='lines',
        name='Histórico',
        line=dict(color='blue', width=2)
    ))

    # 2. Generar índice para el pronóstico
    # Corrección crítica: usar 'ME' en lugar de 'M' (pandas ≥ 2.0)
    last_date = actual.index[-1]
    if hasattr(last_date, 'to_period'):
        # Si es Timestamp, avanzar al siguiente período
        next_period = last_date.to_period('M').to_timestamp(how='end') + pd.offsets.MonthEnd(1)
    else:
        next_period = last_date + pd.offsets.MonthEnd(1)

    forecast_index = pd.date_range(
        start=next_period,
        periods=len(forecast),
        freq='ME'  # ✅ 'ME' = mes final (reemplaza a antiguo 'M')
    )

    # 3. Pronóstico
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast,
        mode='lines',
        name='Pronóstico',
        line=dict(color='red', width=2, dash='dash')
    ))

    # 4. Intervalo de confianza
    if conf_int is not None and 'lower' in conf_int.columns and 'upper' in conf_int.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_index, forecast_index[::-1]]),
            y=pd.concat([conf_int['upper'], conf_int['lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='IC 95%'
        ))

    # 5. Layout
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Valor",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(t=50, b=50, l=50, r=30)
    )

    return fig