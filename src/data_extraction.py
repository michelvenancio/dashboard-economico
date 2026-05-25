"""
Módulo para extracción de datos desde APIs: Banxico y FRED
Incluye conexión a Banxico, FRED e INEGI (INEGIpy).
"""
import pandas as pd
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
                df = df.pivot_table(
                    index='fecha', columns='serie_id',
                    values='valor', aggfunc='mean'
                )

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
def get_fred_series(series_ids, api_key=None, start_date=None, end_date=None):  # ← Agregar end_date
    """
    Extrae series económicas de FRED
    Args:
        series_ids: Lista de IDs (ej: ['GDP', 'CPIAUCSL', 'UNRATE'])
        api_key: API Key de FRED
        start_date: Fecha inicial
        end_date: Fecha final  # ← Agregar
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
                
                # ✅ FILTROS POR FECHA COMPLETOS
                if start_date:
                    series = series[series.index >= pd.to_datetime(start_date)]
                if end_date:  # ← AGREGAR ESTO
                    series = series[series.index <= pd.to_datetime(end_date)]
                
                df_series = series.to_frame(name=series_id)

                # Agregar metadatos
                series_info = fred.get_series_info(series_id)
                df_series['titulo'] = series_info.title
                df_series['unidades'] = series_info.units
                df_series['frecuencia'] = series_info.frequency

                df_list.append(df_series)  # ← Corregir espacio en blanco

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

# ==========================================
# NUEVA FUNCIÓN: Conexión INEGI API (INEGIpy)
# ==========================================
def get_inegi_api_data(
    token: str,
    indicator_code: str,
    state_code: str = '24',  # San Luis Potosí
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Obtiene datos de INEGI usando la API oficial vía INEGIpy.
    """
    try:
        from INEGIpy import Indicadores
        
        # Inicializar cliente
        inegi = Indicadores(token)
        
        # Consultar datos
        df = inegi.obtener_df(
            indicadores=indicator_code,
            clave_area=state_code,
            inicio=start_date,
            fin=end_date,
            metadatos=False
        )
        
        # Renombrar columna para que coincida con tu dashboard
        if not df.empty:
            df.columns = ['EXP_SLP']
            
        return df
        
    except Exception as e:
        print(f"❌ Error consultando INEGI API: {e}")
        return pd.DataFrame()