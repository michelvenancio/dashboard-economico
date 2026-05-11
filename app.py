"""
Dashboard Principal - Análisis de Datos Económicos
Fundamentos de Análisis de Datos - Proyecto Final
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# === STYLING: Diseño al estilo Data México ===
st.markdown("""
<style>
    /* Paleta Data México */
    :root {
        --primary: #FF6B35;      /* Naranja institucional */
        --secondary: #4CAF50;    /* Verde para positivos */
        --dark: #0A192F;         /* Fondo oscuro profundo */
        --dark-alt: #121826;
        --light: #FFFFFF;
        --gray: #C5CAE9;
        --gray-light: #E0E7FF;
    }

    body {
        background-color: var(--dark);
        color: var(--light);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #E55A2A;
        transform: translateY(-1px);
    }

    .stMetric {
        background: var(--dark-alt);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border-left: 4px solid var(--primary);
    }

    .stAlert {
        background-color: rgba(10, 25, 47, 0.7);
        border-left: 4px solid var(--primary);
        color: var(--light);
    }

    h1, h2, h3 {
        color: var(--light);
        margin-top: 0;
    }

    .header-title {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, var(--dark) 0%, #1E293B 100%);
        border-radius: 0 0 16px 16px;
        margin-bottom: 24px;
    }

    .footer {
        text-align: center;
        color: var(--gray);
        font-size: 0.9em;
        padding: 20px;
        margin-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# === FUNCIÓN DE TARJETA DE KPI PERSONALIZADA ===
def metric_card(title: str, value: str, delta: float = None, icon: str = "📈"):
    """delta debe ser número (no string)"""
    color = "var(--secondary)" if delta is not None and delta > 0 else "var(--gray)"
    delta_str = f"{delta:+.2f}%" if delta is not None else ""
    st.markdown(f"""
    <div class="stMetric">
        <div style="font-size: 0.9em; color: var(--gray);">{title}</div>
        <div style="font-size: 1.8em; font-weight: bold; margin: 8px 0;">{value}</div>
        {f'<div style="color: {color}; font-size: 0.9em;">{icon} {delta_str}</div>' if delta is not None else ''}
    </div>
    """, unsafe_allow_html=True)


# Importar módulos locales
from src.data_extraction import get_banxico_data, get_fred_series, get_inegi_csv_data  # Añadido get_inegi_csv_data si lo usas aquí
from src.data_processing import (
    clean_economic_data, calculate_returns, add_technical_indicators,
    apply_log_transform, add_volatility_column, add_post_2020_dummy  # Añadidas nuevas funciones
)
from src.models import (
    fit_arima_model, test_stationarity,
    multiple_regression, auto_arima_optimization  # Añadida nueva función
)
from src.visualization import (
    plot_time_series, plot_correlation_heatmap,
    plot_indicators_cards, create_dashboard_layout,
    plot_log_scatter, plot_residuals, plot_forecast  # Añadidas nuevas funciones
)

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Dashboard Económico",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejor presentación
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CONFIGURACIÓN
# =============================================================================
with st.sidebar:
    st.title("⚙️ Configuración")

# Selección de fuente de datos — SIN ESPACIOS
    data_source = st.radio(
        "📡 Fuente de Datos",
        ["🇲🇽 Banxico", "🇲🇽 INEGI (CSV)", "🇺🇸 FRED", "🔄 Combinar"],  # Añadida opción INEGI CSV
        index=0
    )
    st.divider()

    # Parámetros de fecha
    st.subheader("📅 Rango de Fechas")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Inicio", value=pd.to_datetime("2007-01-01"))  # o la fecha mínima real de tu CSV
    with col2:
        end_date = st.date_input("Fin", value=pd.to_datetime("today"))

    st.divider()

    # Series disponibles — definir opciones limpias (sin espacios)
    banxico_options = {
        "SF43718": "Tipo de Cambio USD/MXN (Fix)",
        "SF331451": "TIIE de fondeo a 1 día",
        "SE28528": "Remesas Totales",
        "SP30577": "INPC (Inflación)",
        "SR17622": "PIB a precios constantes (MDP)"
    }

    fred_options = {
        "DEXMXUS": "Tipo de Cambio MXN/USD",
        "CPIAUCSL": "IPC Estados Unidos",
        "GDPC1": "PIB Real",
        "UNRATE": "Tasa de Desempleo"
    }

    inegi_csv_options = {
        "EXP_SLP": "Exportaciones San Luis Potosí (Trimestrales)"
    }

# === Series disponibles — siempre mostradas, pero con lógica de habilitación ===
    st.subheader("📋 Series Disponibles")

# Banxico
    with st.expander("🇲🇽 Banxico", expanded=("Banxico" in data_source or "Combinar" in data_source)):
        banxico_series = st.multiselect(
        "Series Banxico",
        options=list(banxico_options.keys()),
        format_func=lambda x: banxico_options[x],
        default=["SF43718"],
        key="banxico_series"
    )

# FRED
    with st.expander("🇺🇸 FRED", expanded=("FRED" in data_source or "Combinar" in data_source)):
        fred_series = st.multiselect(
        "Series FRED",
        options=list(fred_options.keys()),
        format_func=lambda x: fred_options[x],
        default=["DEXMXUS", "CPIAUCSL"],
        key="fred_series"
    )

# INEGI (CSV)
    with st.expander("🇲🇽 INEGI (CSV)", expanded=("INEGI" in data_source or "Combinar" in data_source)):
        inegi_csv_series = st.multiselect(
        "Series INEGI (CSV)",
        options=list(inegi_csv_options.keys()),
        format_func=lambda x: inegi_csv_options[x],
        default=["EXP_SLP"],
        key="inegi_csv_series"
    )

    st.divider()

    # Opciones de análisis
    st.subheader("🔍 Análisis")
    show_correlation = st.checkbox("📊 Matriz de Correlación", value=True)
    show_forecast = st.checkbox("🔮 Pronóstico ARIMA", value=True)
    show_regression = st.checkbox("📈 Regresión Lineal", value=False)

    # Botón de actualización
    if st.button("🔄 Actualizar Datos", type="primary", use_container_width=True):
        st.session_state['refresh'] = True

# =============================================================================
# HEADER PRINCIPAL
# =============================================================================
st.markdown("""
<div class="header-title">
    <h1>📊 Impacto del Tipo de Cambio en las Exportaciones de San Luis Potosí</h1>
    <p style="color: var(--gray); margin: 8px 0;">
        Proyecto de Investigación | Fundamentos de Análisis de Datos
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# EXTRACCIÓN Y PROCESAMIENTO DE DATOS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="📡 Extrayendo datos...")
def load_data(source, series_dict, start, end):
    """Función cacheada para cargar datos"""
    all_dfs = []

    # Banxico: solo si hay series y fuente válida
    if source in ["🇲🇽 Banxico", "🔄 Combinar"] and series_dict.get('banxico') and len(series_dict['banxico']) > 0:
        df_banxico = get_banxico_data(
            series_ids=list(series_dict['banxico'].keys()),
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d')
        )
        if df_banxico is not None and not df_banxico.empty:
            df_banxico.columns = [series_dict['banxico'].get(c, c) for c in df_banxico.columns]
            all_dfs.append(df_banxico)

    # FRED
    if source in ["🇺🇸 FRED", "🔄 Combinar"] and series_dict.get('fred') and len(series_dict['fred']) > 0:
        df_fred = get_fred_series(
            series_ids=list(series_dict['fred'].keys()),
            start_date=start.strftime('%Y-%m-%d')
        )
        if df_fred is not None and not df_fred.empty:
            # Solo columnas numéricas (quitar metadatos)
            value_cols = [c for c in df_fred.columns if c in series_dict['fred']]
            df_fred = df_fred[value_cols]
            df_fred.columns = [series_dict['fred'].get(c, c) for c in df_fred.columns]
            all_dfs.append(df_fred)

    # INEGI (CSV)
    if source in ["🇲🇽 INEGI (CSV)", "🔄 Combinar"] and series_dict.get('inegi_csv') and len(series_dict['inegi_csv']) > 0:
        for csv_series_id in series_dict['inegi_csv']:
            CSV_FILE_PATH = "data/inegi_exportaciones_slp.csv"
            df_inegi_csv = get_inegi_csv_data(
                CSV_FILE_PATH, 
                csv_series_id, 
                start_date=start, 
                end_date=end
            )
            if df_inegi_csv is not None and not df_inegi_csv.empty:
                all_dfs.append(df_inegi_csv)

    if all_dfs:
        df_combined = pd.concat(all_dfs, axis=1).sort_index()
        return clean_economic_data(df_combined)

    return None

# === Construir series_config de forma segura ===
# === EXTRAER SERIES DIRECTAMENTE DESDE st.session_state ===
# Esto funciona incluso si el multiselect no se mostró (gracias a key=)
banxico_ids = st.session_state.get('banxico_series', [])
fred_ids = st.session_state.get('fred_series', [])
inegi_csv_ids = st.session_state.get('inegi_csv_series', [])

banxico_ids = [s.strip() for s in banxico_ids if s and s.strip()]
fred_ids = [s.strip() for s in fred_ids if s and s.strip()]
inegi_csv_ids = [s.strip() for s in inegi_csv_ids if s and s.strip()]

series_dict = {
    'banxico': {sid: sid for sid in banxico_ids},
    'fred': {sid: sid for sid in fred_ids},
    'inegi_csv': {sid: sid for sid in inegi_csv_ids}
}

# Cargar datos
with st.spinner("🔄 Procesando datos..."):
    df = load_data(data_source, series_dict, start_date, end_date)

# =============================================================================
# MANEJO DE ERRORES Y ESTADO VACÍO
# =============================================================================
if df is None or df.empty:
    st.warning("""
    ### ⚠️ No hay datos para mostrar
    
    **Posibles causas:**
    - Tokens de API no configurados (revisar `.streamlit/secrets.toml`)
    - Series seleccionadas no disponibles en el rango de fechas
    - Error de conexión con las APIs
    
    **Solución:** Verifica tu configuración y presiona "Actualizar Datos"
    """)
    st.stop()

# =============================================================================
# PANEL DE MÉTRICAS PRINCIPALES
# =============================================================================
st.subheader("📌 Indicadores Clave")

# Calcular métricas dinámicas
metrics = {}
for col in df.select_dtypes(include='number').columns:
    latest = df[col].iloc[-1]
    prev = df[col].iloc[-2] if len(df) > 1 else latest
    change = ((latest - prev) / prev * 100) if prev != 0 else 0
    
    # Determinar unidad
    # Determinar unidad y tipo de cambio
if 'tipo de cambio' in col.lower() or 'exchange' in col.lower():
    suffix = ' MXN/USD'
    # Para tipo de cambio, el cambio absoluto sí es relevante
    change_val = latest - prev
elif 'cpi' in col.lower() or 'inflación' in col.lower():
    # CPI es un índice (base 100), no un % → no añadir '%'
    suffix = ''  # ✅ Correcto: solo "330.29", no "330.29%"
    # Cambio relativo en % (inflación interanual o trimestral)
    change_val = ((latest - prev) / prev * 100) if prev != 0 else 0
elif 'unrate' in col.lower():
    # Tasa de desempleo sí es % → mantener
    suffix = '%'
    change_val = latest - prev  # puntos porcentuales
else:
    suffix = ''
    change_val = latest - prev
    
    metrics[col] = (
    round(latest, 2),
    suffix,
    round(prev, 2) if change_val != 0 else None,
    round(change_val, 2)  # <-- nuevo: guardamos el cambio numérico para delta
)

# Mostrar tarjetas de métricas
if metrics:  # Solo si hay métricas
    metric_cols = st.columns(len(metrics))
    for i, (col_name, (value, suffix, prev, delta_num)) in enumerate(metrics.items()):
        with metric_cols[i]:
            st.metric(
                label=col_name,
                value=f"{value}{suffix}",
                delta=f"{delta_num:+.2f}{suffix}" if prev is not None else None,
                delta_color="inverse" if 'inflación' in col_name.lower() or 'unrate' in col_name.lower() else "normal"
            )
else:
    st.info("⚠️ No hay indicadores para mostrar. Verifica que las series estén disponibles en el rango de fechas.")

st.divider()

# =============================================================================
# VISUALIZACIONES PRINCIPALES
# =============================================================================
col_chart, col_stats = st.columns([3, 1])

with col_chart:
    st.subheader("📈 Evolución Temporal")
    
    # Selector de variables para graficar
    vars_to_plot = st.multiselect(
        "Selecciona variables:",
        options=df.select_dtypes(include='number').columns.tolist(),
        default=df.select_dtypes(include='number').columns.tolist()[:3]
    )
    
    # === NUEVO: Opción para eje secundario ===
    if len(vars_to_plot) > 1:
        secondary_y = st.multiselect(
            "Variables para eje Y secundario (escalas diferentes):",
            options=vars_to_plot,
            default=[],
            help="Selecciona variables con valores muy diferentes para usar el eje derecho"
        )
    else:
        secondary_y = []
    
    if vars_to_plot:
        fig_ts = plot_time_series(
            df[vars_to_plot], 
            title="Serie Temporal de Indicadores",
            yaxis_title="Valor",
            secondary_y=secondary_y if secondary_y else None  # Pasar al gráfico
        )
        st.plotly_chart(fig_ts, use_container_width=True)

with col_stats:
    st.subheader("📊 Estadísticas")
    
    # Resumen estadístico
    st.dataframe(
        df[vars_to_plot].describe().round(2),
        use_container_width=True,
        hide_index=True
    )
    
    # Estacionariedad (ADF test)
    if len(vars_to_plot) > 0:
        st.markdown("##### 🧪 Prueba ADF")
        for col in vars_to_plot[:2]:  # Limitar a 2 para no saturar
            result = test_stationarity(df[col].dropna())
            status = "✅ Estacionaria" if result['is_stationary'] else "❌ No estacionaria"
            st.caption(f"{col}: {status} (p={result['p_value']:.3f})")

# =============================================================================
# ANÁLISIS AVANZADO
# =============================================================================
if show_correlation and len(df.select_dtypes(include='number').columns) > 1:
    st.subheader("🔗 Análisis de Correlación")
    fig_corr = plot_correlation_heatmap(df, title="Matriz de Correlación de Pearson")
    st.plotly_chart(fig_corr, use_container_width=True)

# =============================================================================
# PRONÓSTICO ARIMA
# =============================================================================
if show_forecast:
    st.subheader("🔮 Pronóstico con ARIMA")
    
    col_model, col_plot = st.columns([1, 2])
    
    with col_model:
        forecast_var = st.selectbox(
            "Variable a pronosticar:",
            options=df.select_dtypes(include='number').columns.tolist()
        )
        
        # === MODO MANUAL vs AUTOMÁTICO ===
        optimization_mode = st.radio(
            "📊 Método de selección:",
            options=["🎯 Manual", "🤖 Automático (Grid Search)"],
            index=1  # Default: Automático
        )
        
        if optimization_mode == "🎯 Manual":
            # Modo manual (parámetros fijos)
            col_p, col_d, col_q = st.columns(3)
            with col_p: p = st.number_input("p (AR)", min_value=0, max_value=5, value=1)
            with col_d: d = st.number_input("d (Diferenciación)", min_value=0, max_value=2, value=1)
            with col_q: q = st.number_input("q (MA)", min_value=0, max_value=5, value=1)
            order = (p, d, q)
            
        else:
            # Modo automático
            st.info("ℹ️ El sistema probará combinaciones de (p,d,q) y seleccionará la óptima")
            
            # Parámetros avanzados (opcional)
            with st.expander("⚙️ Parámetros de búsqueda"):
                max_p = st.slider("Máximo p", 0, 5, 3)
                max_d = st.slider("Máximo d", 0, 2, 2)
                max_q = st.slider("Máximo q", 0, 5, 3)
            order = None  # Se determinará automáticamente
        
        steps = st.slider("Periodos a pronosticar", 1, 24, 12)
        
        # === BOTÓN DE GENERAR ===
        if st.button("🚀 Generar Pronóstico", type="primary", use_container_width=True):
            with st.spinner(" Buscando mejores parámetros..." if optimization_mode == "🤖 Automático (Grid Search)" else "Ajustando modelo ARIMA..."):
                
                if optimization_mode == "🤖 Automático (Grid Search)":
                    # Usar optimización automática
                    result = auto_arima_optimization(
                        df[forecast_var].dropna(),
                        max_p=max_p,
                        max_d=max_d,
                        max_q=max_q,
                        forecast_steps=steps
                    )
                    
                    if 'error' not in result:
                        st.session_state['forecast_result'] = result
                        st.session_state['forecast_var'] = forecast_var
                        st.session_state['optimization_mode'] = 'auto'
                        
                        st.success(f"✅ Modelo ARIMA{result['best_order']} seleccionado")
                        st.caption(f"AIC: {result['best_aic']:.2f} | BIC: {result['best_bic']:.2f}")
                    else:
                        st.error(f"❌ Error: {result['error']}")
                
                else:
                    # Usar modo manual
                    result = fit_arima_model(
                        df[forecast_var].dropna(),
                        order=order,
                        forecast_steps=steps
                    )
                    
                    if 'error' not in result:
                        st.session_state['forecast_result'] = result
                        st.session_state['forecast_var'] = forecast_var
                        st.session_state['optimization_mode'] = 'manual'
                        st.success("✅ Modelo ajustado exitosamente")
                    else:
                        st.error(f"❌ Error: {result['error']}")
    
    # === MOSTRAR RESULTADOS ===
    with col_plot:
        if 'forecast_result' in st.session_state:
            res = st.session_state['forecast_result']
            var = st.session_state['forecast_var']
        
            # Verificar que el resultado tenga las claves necesarias
            if 'forecast' not in res or 'conf_int' not in res:
                st.error("❌ El modelo no generó un pronóstico válido")
                st.stop()
        
            fig_fc = plot_forecast(
                actual=df[var],
                forecast=res['forecast'],
                conf_int=res['conf_int'],
                title=f"Pronóstico ARIMA({res.get('order', 'N/A')}) - {var}"
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        
            # Métricas del modelo (con validación)
            with st.expander("📋 Métricas del Modelo", expanded=True):
                if 'metrics' in res:
                    metrics = res['metrics']
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        st.metric("AIC", f"{metrics.get('aic', 'N/A'):.2f}" if isinstance(metrics.get('aic'), (int, float)) else "N/A")
                    
                    with col2:
                        st.metric("BIC", f"{metrics.get('bic', 'N/A'):.2f}" if isinstance(metrics.get('bic'), (int, float)) else "N/A")
                    
                    with col3:
                        rmse = metrics.get('rmse_insample')
                        st.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "N/A")
                else:
                    st.warning("⚠️ Las métricas del modelo no están disponibles")
                    st.json(res)  # Mostrar lo que sí tenemos
                
                # Tabla comparativa (solo en modo automático)
                if st.session_state.get('optimization_mode') == 'auto' and 'all_results' in res:
                    st.markdown("##### 📊 Top 5 Modelos Evaluados")
                    top_models = res['all_results'].head(5)[['order', 'aic', 'bic']].copy()
                    top_models['order'] = top_models['order'].astype(str)
                    st.dataframe(top_models.round(2), use_container_width=True)

# =============================================================================
# REGRESIÓN LINEAL (OPCIONAL)
# =============================================================================
if show_regression and df is not None and not df.empty:
    st.subheader("📊 Modelo Econométrico (OLS Multivariable)")

    # === Paso 1: Procesar datos usando funciones modularizadas ===
    df_with_logs = df.copy()

    # 1. Identificar columnas numéricas > 0 para logaritmos
    cols_for_log = [
        col for col in df.select_dtypes(include='number').columns 
        if col != 'fecha' and df[col].min() > 0
    ]

    # 2. Aplicar transformaciones usando funciones de data_processing
    df_with_logs = apply_log_transform(df_with_logs, cols=cols_for_log)

    if 'SF43718' in df.columns:
        df_with_logs = add_volatility_column(df_with_logs, col='SF43718', window=4)

    df_with_logs = add_post_2020_dummy(df_with_logs, date_col=None)  # None = usar índice

    # 3. Asegurar frecuencia válida (pandas ≥ 2.0)
    if not hasattr(df_with_logs.index, 'freq') or df_with_logs.index.freq is None:
        try:
            df_with_logs = df_with_logs.asfreq('ME')
        except:
            df_with_logs = df_with_logs.asfreq('M')  # Fallback para pandas antiguos

    # === Paso 2: Seleccionar variables
    all_cols = df_with_logs.select_dtypes(include='number').columns.tolist()

    # Variable dependiente: buscar columnas con 'exp' o 'slp'
    y_candidates = [c for c in all_cols if 'exp' in c.lower() or 'slp' in c.lower() or 'export' in c.lower()]
    y_options = [c for c in y_candidates if 'ln_' in c]  # prioriza logaritmos
    if not y_options:
        y_options = y_candidates
    if not y_options and all_cols:
        y_options = [all_cols[0]]

    y_var = st.selectbox(
        "Variable dependiente (ln):",
        options=y_options,
        index=0 if y_options else 0,
        disabled=len(y_options) == 0
    )

    # Variables independientes
    x_options = [c for c in all_cols if c != y_var] if y_var else all_cols[1:]
    x_vars = st.multiselect(
        "Variables independientes:",
        options=x_options,
        default=[c for c in x_options if 'ln_' in c or 'vol' in c.lower() or 'DPOST' in c]
    )

    # === Paso 3: Estimar modelo
    if y_var and x_vars:
        try:
            result = multiple_regression(df_with_logs, y_var, x_vars)
            if 'error' in result:
                st.error(f"❌ Error en modelo: {result['error']}")
            else:
                # Mostrar resultados
                st.markdown("##### 📈 Coeficientes (Elasticidades)")
                for var, coef in result['coefficients'].items():
                    if var != 'const':
                        text = f"**{var}**: {coef:+.4f}"
                        if 'ln_' in y_var and 'ln_' in var:
                            text += " → Elasticidad"
                        st.caption(text)

                st.metric("R²", f"{result['r2']:.3f}")
                st.metric("RMSE", f"{result['rmse']:.4f}")

        except Exception as e:
            st.error(f"❌ Error en regresión: {e}")
    else:
        st.warning("⚠️ Seleccione una variable dependiente y al menos una independiente.")


# === Diagnóstico del modelo (versión robusta) ===
if 'diagnostics' in result:
    diag = result['diagnostics']
    
    st.markdown("##### 📊 Pruebas de supuestos")
    
    # Verificar que los valores existan
    bp_p = diag.get('bp_p', None)
    jb_p = diag.get('jb_p', None)
    dw_stat = diag.get('dw_stat', None)
    vif_df = diag.get('vif', pd.DataFrame(columns=['feature', 'VIF']))

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "p-value β₁ (TC)",
            f"{diag['p_values'].get('ln_SF43718', 'N/A'):.4f}",
            delta="✅ < 0.05" if diag['p_values'].get('ln_SF43718', 1) < 0.05 else "⚠️ ≥ 0.05",
            help="Significancia del coeficiente principal"
        )
    
    with col2:
        st.metric(
            "VIF (TC)",
            f"{vif_df.loc[vif_df['feature'] == 'ln_SF43718', 'VIF'].iloc[0]:.2f}" 
            if 'ln_SF43718' in vif_df['feature'].values else "N/A",
            delta="✅ < 5" if vif_df.loc[vif_df['feature'] == 'ln_SF43718', 'VIF'].iloc[0] < 5 else "⚠️ > 5",
            help="Multicolinealidad (VIF > 10 es problemático)"
        )
    
    with col3:
        st.metric(
            "Durbin-Watson",
            f"{dw_stat:.2f}" if dw_stat is not None else "N/A",
            delta="✅ ≈2.0" if dw_stat is not None and 1.5 < dw_stat < 2.5 else "⚠️ Autocorrelación",
            help="Valores cercanos a 2 indican ausencia de autocorrelación"
        )

    # Gráfica de residuos (solo si está definida y hay datos)
    try:
        if 'residuals' in result:
            fig_resid = plot_residuals(result['residuals'], df_with_logs.index)
            st.plotly_chart(fig_resid, use_container_width=True)
        else:
            st.info("📊 Residuos no disponibles para gráfica.")
    except Exception as e:
        st.warning(f"⚠️ No se pudo generar gráfica de residuos: {e}")
else:
    st.info("ℹ️ El modelo no incluye diagnóstico. Asegúrese de que `multiple_regression` devuelva `'diagnostics'`.")

# =============================================================================
# FOOTER Y CRÉDITOS
# =============================================================================
st.markdown("""
<div class="footer">
    <strong>Dashboard de Investigación Económica</strong> | Fundamentos de Análisis de Datos  
    📊 Desarrollado con Python • Streamlit • Plotly • StatsModels  
    🏛️ Inspirado en el diseño de <a href="https://www.datamexico.org/" target="_blank" style="color: var(--primary);">Data México</a>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# DESCARGA DE DATOS (BONUS)
# =============================================================================
with st.expander("💾 Descargar Datos Procesados"):
    st.markdown("Descarga el dataset limpio en formato CSV:")
    
    csv = df.to_csv(index=True)
    st.download_button(
        label="📥 Descargar CSV",
        data=csv,
        file_name=f"datos_economicos_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )