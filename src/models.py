"""
Módulo con modelos estadísticos y econométricos
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


def test_stationarity(series, significance=0.05):
    """
    Prueba de raíz unitaria (ADF) para estacionariedad
    
    Returns:
        dict con resultados de la prueba
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < significance
    }


def fit_arima_model(series, order=(1,1,1), forecast_steps=12):
    try:
        model = ARIMA(series.dropna(), order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
        
        # ✅ Asegúrate de que esto exista
        metrics = {
            'aic': model_fit.aic,
            'bic': model_fit.bic,
            'rmse_insample': np.sqrt(np.mean(model_fit.resid**2))
        }
        
        return {
            'model': model_fit,
            'forecast': forecast,
            'conf_int': conf_int,
            'metrics': metrics,  # ← Clave crítica
            'order': order
        }
        
    except Exception as e:
        return {'error': str(e)}


def multiple_regression(df: pd.DataFrame, y_col: str, x_cols: list, add_constant: bool = True) -> dict:
    """
    Regresión lineal múltiple OLS.
    Args:
        df: DataFrame con todas las variables.
        y_col: nombre de la variable dependiente (ej: "ln_EXP_SLP").
        x_cols: lista de predictores (ej: ["ln_SF43718", "ln_GDP", "SF43718_vol", "DPOST_2020"]).
    Returns:
        dict con coeficientes, R², RMSE, predicciones, residuos.
    """
    X = df[x_cols].dropna()
    y = df[y_col].loc[X.index]

    # Asegurar que no haya NaN en y
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    if add_constant:
        X = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':4})
        results = {
            'model': model,
            'coefficients': model.params,
            'p_values': model.pvalues,  # Nuevo
            'r2': model.rsquared,
            'adj_r2': model.rsquared_adj,
            'rmse': np.sqrt(model.mse_resid),
            'predictions': model.fittedvalues,
            'residuals': model.resid,
            'aic': model.aic,
            'bic': model.bic
        }
        # === Diagnóstico ===
        results['diagnostics'] = model_diagnostics(model, X, model.resid)
        return results
    except Exception as e:
        return {'error': str(e)}


def model_diagnostics(model, X, residuals):
    # 1. p-values
    p_values = model.pvalues.to_dict()
    
    # 2. VIF (excluir constante para evitar VIF infinito)
    X_no_const = X.drop(columns='const', errors='ignore')
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_no_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])]
    
    # 3. Breusch-Pagan
    try:
        bp_test = het_breuschpagan(residuals, X_no_const)
        bp_p = bp_test[1]
    except:
        bp_p = None

    # 4. Jarque-Bera
    try:
        jb_test = jarque_bera(residuals)
        jb_p = jb_test[1]
    except:
        jb_p = None

    # 5. Durbin-Watson
    dw_stat = durbin_watson(residuals)

    # ✅ Retorna directamente, sin recursión
    return {
        'p_values': p_values,
        'vif': vif_data,
        'bp_p': bp_p,
        'jb_p': jb_p,
        'dw_stat': dw_stat
    }

def auto_arima_optimization(series, max_p=3, max_d=2, max_q=3, forecast_steps=12):
    """
    Búsqueda automática de mejores parámetros ARIMA mediante Grid Search.
    Prueba combinaciones de (p,d,q) y selecciona la que minimiza AIC.
    
    Args:
        series: Serie temporal (pd.Series)
        max_p: Máximo valor para p (AR)
        max_d: Máximo valor para d (diferenciación)
        max_q: Máximo valor para q (MA)
        forecast_steps: Número de periodos a pronosticar
    
    Returns:
        dict con mejores parámetros, modelo y métricas comparativas
    """
    results = []
    
    # Grid search sobre todas las combinaciones
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    # Intentar ajustar modelo
                    model = ARIMA(series.dropna(), order=(p, d, q))
                    model_fit = model.fit()
                    
                    # Verificar convergencia
                    if not np.isfinite(model_fit.aic):
                        continue
                    
                    results.append({
                        'order': (p, d, q),
                        'aic': model_fit.aic,
                        'bic': model_fit.bic,
                        'model': model_fit,
                        'converged': True
                    })
                    
                except Exception:
                    # Si no converge, saltar
                    continue
    
    if not results:
        return {'error': 'Ningún modelo ARIMA convergió'}
    
    # Ordenar por AIC (menor es mejor)
    results_df = pd.DataFrame(results).sort_values('aic')
    best = results_df.iloc[0]
    
    # Generar pronóstico con el mejor modelo
    forecast = best['model'].forecast(steps=forecast_steps)
    conf_int = best['model'].get_forecast(steps=forecast_steps).conf_int()
    
    return {
        'best_order': best['order'],
        'best_aic': best['aic'],
        'best_bic': best['bic'],
        'model': best['model'],
        'forecast': forecast,
        'conf_int': conf_int,
        'all_results': results_df,
        'forecast_steps': forecast_steps
    }

def check_cointegration(df, y_col, x_cols):
    """
    Prueba de Cointegración (Engle-Granger).
    Si los residuos de la regresión son estacionarios, hay cointegración.
    """
    try:
        # 1. Ejecutar la regresión OLS en niveles (no logaritmos, ni diferencias)
        # Nota: Cointegración se hace sobre las variables originales o log-transformadas si son I(1)
        # Aquí usamos las columnas de logaritmo que ya tienes
        X = sm.add_constant(df[x_cols])
        y = df[y_col]
        model = sm.OLS(y, X).fit()
        
        # 2. Obtener los residuos
        residuals = model.resid
        
        # 3. Aplicar prueba ADF a los residuos
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(residuals, autolag='AIC')
        
        p_value = result[1]
        is_cointegrated = p_value < 0.05
        
        return {
            'is_cointegrated': is_cointegrated,
            'p_value': p_value,
            'adf_stat': result[0],
            'interpretation': "✅ Cointegración Confirmada (Relación de Largo Plazo)" if is_cointegrated else " No Cointegradas (Regresión Espuria)"
        }
    except Exception as e:
        return {'error': str(e)}