"""
Módulo para limpieza, transformación y feature engineering
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def clean_economic_data(df, method='interpolate'):
    """
    Limpieza básica de series económicas
    
    Args:
        df: DataFrame con series temporales
        method: Método para manejar valores faltantes
    
    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()
    
    # Convertir índices a datetime si es necesario
    if not pd.api.types.is_datetime64_any_dtype(df_clean.index):
        df_clean.index = pd.to_datetime(df_clean.index)
    
    # Ordenar por fecha
    df_clean = df_clean.sort_index()
    
    # Eliminar duplicados
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
    
    # Manejar valores faltantes
    if method == 'interpolate':
        df_clean = df_clean.interpolate(method='time')
    elif method == 'ffill':
        df_clean = df_clean.ffill()
    elif method == 'drop':
        df_clean = df_clean.dropna()
    
    # Eliminar valores extremos (opcional, con IQR)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3*IQR
        upper_bound = Q3 + 3*IQR
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    return df_clean


def calculate_returns(df, columns=None, method='pct_change'):
    """
    Calcula rendimientos/variaciones para análisis
    
    Args:
        df: DataFrame con series
        columns: Columnas a procesar (None = todas numéricas)
        method: 'pct_change', 'diff', o 'log_return'
    
    Returns:
        DataFrame con variaciones calculadas
    """
    df_ret = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns
    
    for col in cols:
        if method == 'pct_change':
            df_ret[f'{col}_var'] = df_ret[col].pct_change() * 100
        elif method == 'diff':
            df_ret[f'{col}_var'] = df_ret[col].diff()
        elif method == 'log_return':
            df_ret[f'{col}_var'] = np.log(df_ret[col] / df_ret[col].shift(1)) * 100
    
    return df_ret


def add_technical_indicators(df, column, windows=[7, 30, 90]):
    """
    Agrega indicadores técnicos para análisis
    
    Args:
        df: DataFrame
        column: Columna base para cálculos
        windows: Ventanas para medias móviles
    
    Returns:
        DataFrame con indicadores agregados
    """
    df_ind = df.copy()
    
    # Medias móviles
    for w in windows:
        df_ind[f'{column}_ma{w}'] = df_ind[column].rolling(window=w, min_periods=1).mean()
    
    # Volatilidad móvil (desviación estándar)
    df_ind[f'{column}_vol30'] = df_ind[column].rolling(window=30, min_periods=1).std()
    
    # Momentum
    df_ind[f'{column}_mom12'] = df_ind[column].pct_change(periods=12) * 100
    
    # Z-score para detección de anomalías
    df_ind[f'{column}_zscore'] = (
        (df_ind[column] - df_ind[column].rolling(30, min_periods=1).mean()) / 
        df_ind[column].rolling(30, min_periods=1).std()
    )
    
    return df_ind


def resample_frequency(df, freq='M', agg_func='last'):
    """
    Cambia la frecuencia de la serie temporal
    
    Args:
        df: DataFrame con índice datetime
        freq: Nueva frecuencia ('D', 'W', 'M', 'Q', 'A')
        agg_func: Función de agregación
    
    Returns:
        DataFrame resampleado
    """
    agg_map = {
        'last': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
        'mean': 'mean',
        'sum': 'sum',
        'first': lambda x: x.iloc[0] if len(x) > 0 else np.nan
    }
    
    return df.resample(freq).agg(agg_map.get(agg_func, agg_func)).dropna()

def apply_log_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Aplica logaritmo natural a columnas seleccionadas (evita valores ≤ 0)."""
    df_log = df.copy()
    for col in cols:
        if col in df_log.columns:
            # Filtrar valores > 0
            mask = df_log[col] > 0
            df_log[f"ln_{col}"] = np.log(df_log[col].where(mask))
            # Opcional: rellenar NaN con 0 o None, según necesidad
    return df_log

def add_volatility_column(df: pd.DataFrame, col: str, window: int = 4) -> pd.DataFrame:
    """Agrega columna de volatilidad (desv. estándar móvil) de una serie."""
    df = df.copy()
    if col in df.columns:
        df[f"{col}_vol"] = df[col].rolling(window=window).std()
    return df

def add_post_2020_dummy(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    Agrega columna 'DPOST_2020' = 1 si fecha >= 2020-01-01.
    
    Args:
        df: DataFrame con índice datetime o columna de fecha
        date_col: Nombre de la columna de fecha (None = usar índice)
    """
    df = df.copy()
    
    # Determinar qué usar: ¿columna o índice?
    if date_col and date_col in df.columns:
        fechas = pd.to_datetime(df[date_col])
    elif hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index):
        fechas = df.index
    else:
        raise ValueError("No se encontró columna de fecha ni índice datetime")
    
    df["DPOST_2020"] = (fechas >= pd.Timestamp("2020-01-01")).astype(int)
    return df

def resample_to_quarterly(df: pd.DataFrame, agg_rules: dict) -> pd.DataFrame:
    """
    Convierte DataFrame a frecuencia trimestral con reglas de agregación personalizadas.
    
    Args:
        df: DataFrame con índice datetime
        agg_rules: Dict {columna: función_de_agregación}
                   Ej: {'SF43718': 'mean', 'EXP_SLP': 'sum', 'GDPC1': 'last'}
    
    Returns:
        DataFrame con frecuencia trimestral
    """
    # Asegurar índice datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    # Resamplear a trimestral ('QE' = final de trimestre) ✅ CORREGIDO
    df_q = df.resample('QE').agg(agg_rules).dropna()
    
    # Cambiar a frecuencia de período trimestral (opcional, pero más limpio)
    df_q.index = df_q.index.to_period('Q').to_timestamp()
    
    return df_q