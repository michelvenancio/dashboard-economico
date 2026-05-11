"""
Módulo para extracción de datos desde APIs: Banxico y FRED
INEGI ha sido removido de este script.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


# =============================================================================
# CONEXIÓN BANXICO (SIE API)
# =============================================================================
def get_banxico_data(series_ids, start_date=None, end_date=None, token=None):
    """
    Extrae series temporales del Banco de México usando banxicoapi
    Documentación oficial: https://pypi.org/project/banxicoapi/
    """
    try:
        # Import CORRECTO según PyPI
        from banxicoapi.banxico_api import BanxicoApi
        # Obtener token
        if token is None:
            token = st.secrets.get("BANXICO_TOKEN")

        if not token:
            st.error("❌ Token de Banxico no configurado")
            return None

        # Inicializar API (forma correcta)
        banxico_api = BanxicoApi(token)

        # Convertir fechas a formato ISO (YYYY-MM-DD)
        start_str = None
        end_str = None
        if start_date and end_date:
            start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # Consultar datos usando el método get()
        data_list = []

        try:
            # Llamar a la API
            response = banxico_api.get(
                series=series_ids,
                start_date=start_str,
                end_date=end_str
            )

            # Procesar respuesta
            if response:
                # La respuesta puede tener diferentes estructuras
                if isinstance(response, dict):
                    # Si tiene clave 'data' o 'series'
                    series_data = response.get('data', response.get('series', response))

                    for serie_id, datos_serie in series_data.items():
                        if isinstance(datos_serie, dict):
                            for fecha_str, valor_str in datos_serie.items():
                                if fecha_str and valor_str and str(valor_str).strip():
                                    try:
                                        fecha = pd.to_datetime(fecha_str, errors='coerce')
                                        valor = float(str(valor_str).replace(',', ''))

                                        data_list.append({
                                            'fecha': fecha,
                                            'valor': valor,
                                            'serie_id': serie_id,
                                            'titulo': serie_id
                                        })
                                    except Exception as e:
                                        st.warning(f"⚠️ Error procesando {fecha_str}: {e}")
                                        continue

                # Si la respuesta es una lista de series
                elif isinstance(response, list):
                    for item in response:
                        if isinstance(item, dict) and 'idSerie' in item:
                            serie_id = item['idSerie']
                            serie_titulo = item.get('titulo', serie_id)

                            for dato in item.get('datos', []):
                                fecha_str = dato.get('fecha', '')
                                valor_str = dato.get('dato', '')

                                if fecha_str and valor_str:
                                    try:
                                        fecha = pd.to_datetime(fecha_str, dayfirst=True, errors='coerce')
                                        valor = float(str(valor_str).replace(',', ''))

                                        data_list.append({
                                            'fecha': fecha,
                                            'valor': valor,
                                            'serie_id': serie_id,
                                            'titulo': serie_titulo
                                        })
                                    except Exception as e:
                                        st.warning(f"⚠️ Error: {e}")
                                        continue

            if not data_list:
                st.warning(f"⚠️ No se obtuvieron datos para: {series_ids}")
                st.write("💡 Respuesta recibida: ", response)
                return None

            # Crear DataFrame
            df = pd.DataFrame(data_list)

            if len(df) > 0:
                df = df.pivot(index='fecha', columns='serie_id', values='valor')

            return df

        except Exception as api_error:
            st.error(f"❌ Error en llamada a Banxico API: {api_error}")
            return None

    except ImportError as e:
        st.error(f"❌ Error de importación: {e}")
        st.error("❌ Ejecuta: pip install banxicoapi")
        return None
    except Exception as e:
        st.error(f"❌ Error general: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


# =============================================================================
# CONEXIÓN FRED (Federal Reserve)
# =============================================================================
def get_fred_series(series_ids, api_key=None, start_date=None):
    """
    Extrae series económicas de FRED
    Args:
        series_ids: Lista de IDs (ej: ['GDP', 'CPIAUCSL', 'UNRATE'])
        api_key: API Key de FRED
        start_date: Fecha inicial

    Returns:
        DataFrame con múltiples series
    """
    try:
        from fredapi import Fred

        if api_key is None:
            api_key = st.secrets.get("FRED_API_KEY")

        if not api_key:
            st.error("❌ API Key de FRED no configurada")
            return None

        fred = Fred(api_key=api_key)
        df_list = []

        for series_id in series_ids:
            try:
                series = fred.get_series(series_id)
                if start_date:
                    series = series[series.index >= pd.to_datetime(start_date)]
                df_series = series.to_frame(name=series_id)

                # Agregar metadatos
                series_info = fred.get_series_info(series_id)
                df_series['titulo'] = series_info.title
                df_series['unidades'] = series_info.units
                df_series['frecuencia'] = series_info.frequency

                df_list.append(df_series)

            except Exception as e:
                st.warning(f"⚠️ No se pudo obtener {series_id}: {e}")
                continue

        if df_list:
            return pd.concat(df_list, axis=1)
        return None

    except ImportError:
        st.warning("⚠️ Instalar: pip install fredapi")
        return None
    except Exception as e:
        st.error(f"❌ Error FRED: {str(e)}")
        return None

# =============================================================================
# CONEXIÓN INEGI (CSV LOCAL)
# =============================================================================
def get_inegi_csv_data(file_path: str, series_id: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Lee y filtra una serie temporal desde un archivo CSV local.
    Soporta filtrado por rango de fechas.
    """
    try:
        df_raw = pd.read_csv(file_path, encoding='utf-8-sig')
        df_raw.columns = df_raw.columns.str.strip()

        # Detectar columnas (robusto)
        col_fecha = next((col for col in df_raw.columns if 'periodo' in col.lower()), None)
        col_valor = next((col for col in df_raw.columns if 'export' in col.lower() or 'slp' in col.lower() or 'valor' in col.lower()), None)

        if not col_fecha or not col_valor:
            st.error(f"❌ Columnas no encontradas. Esperaba 'Periodos' y 'Exportaciones...'. Columnas: {list(df_raw.columns)}")
            return None

        df_raw = df_raw.rename(columns={col_fecha: 'fecha', col_valor: 'valor'})
        df_raw['fecha'] = pd.to_datetime(df_raw['fecha'], format='%Y/%m', errors='coerce')
        df_raw = df_raw.dropna(subset=['fecha', 'valor'])
        df_raw['valor'] = pd.to_numeric(df_raw['valor'], errors='coerce')
        df_raw = df_raw.dropna(subset=['valor'])

        df = df_raw.set_index('fecha')[['valor']].sort_index()
        df.rename(columns={'valor': series_id}, inplace=True)

        # Filtrar por rango de fechas (si se proporcionan)
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        # Si después del filtro está vacío, devolver None (no es error, solo sin datos en rango)
        if df.empty:
            st.warning(f"⚠️ No hay datos para '{series_id}' en el rango {start_date} – {end_date}")
            return None

        return df

    except FileNotFoundError:
        st.error(f"❌ Archivo CSV no encontrado: {file_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error al leer el CSV: {e}")
        return None