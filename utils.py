import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar


def calcular_efectividad(t_ao, t_ao_venta):
    """
    Calcula la probabilidad de efectividad (P_EFECTIVIDAD) basada en T_AO y T_AO_VENTA.
    """
    return np.where(t_ao > 0, t_ao_venta / t_ao, 0)


def _modelo_efectividad(dotacion, t_ao_venta, params):
    """
    Modelo no lineal (sigmoide) para la P_EFECTIVIDAD en función de DOTACION y T_AO_VENTA.
    Si dotacion es 0 o negativa, la efectividad es 0.
    """
    if not isinstance(dotacion, (int, float, np.number)) or dotacion <= 0:
        return 0.0

    L = params.get('L', 1.0)
    k = params.get('k', 0.5)
    x0_base = params.get('x0_base', 5.0)
    x0_factor_t_ao_venta = params.get('x0_factor_t_ao_venta', 0.05)

    # Asegurar que t_ao_venta sea un escalar numérico para el cálculo de x0
    if not isinstance(t_ao_venta, (int, float, np.number)):
        # Si t_ao_venta no es un número, no podemos calcular x0 de forma fiable,
        # podríamos devolver 0 o un valor por defecto, o lanzar un error.
        # Por ahora, asumimos que t_ao_venta será numérico en este punto.
        # Si t_ao_venta puede ser NaN, hay que manejarlo.
        # Si es NaN, el resultado de la sigmoide podría ser NaN.
        if pd.isna(t_ao_venta):
             return 0.0 # O manejar de otra forma si t_ao_venta es NaN

    x0 = x0_base - x0_factor_t_ao_venta * t_ao_venta
    x0 = max(1.0, x0) # Asegura que x0 sea al menos 1

    # Evitar overflow en np.exp si k * (dotacion - x0) es muy grande negativo
    exp_term = -k * (dotacion - x0)
    if exp_term > np.log(np.finfo(float).max / (L if L > 0 else 1.0) ): # Aproximación para evitar overflow en exp
        return L # La efectividad tiende a L
    if exp_term < np.log(np.finfo(float).tiny): # Aproximación para evitar que 1 + exp() sea solo 1.0 y luego L/1
        return 0.0 # La efectividad tiende a 0

    return L / (1 + np.exp(exp_term))


def _modelo_efectividad_fit(X, L, k, x0_base, x0_factor_t_ao_venta):
    """
    Función auxiliar para curve_fit que toma un array X = [DOTACION, T_AO_VENTA].
    Si dotacion es 0 o negativa, la efectividad es 0.
    """
    dotacion_arr = X[0]    # Array de dotaciones
    t_ao_venta_arr = X[1]  # Array de T_AO_VENTA

    p_efectividad = np.zeros_like(dotacion_arr, dtype=float)

    # Índices donde la dotación es positiva
    idx_dotacion_positiva = (dotacion_arr > 0)

    # Solo realizar cálculos para dotaciones positivas
    if np.any(idx_dotacion_positiva):
        dotacion_pos = dotacion_arr[idx_dotacion_positiva]
        t_ao_venta_pos = t_ao_venta_arr[idx_dotacion_positiva]

        # Calcular x0 para las dotaciones positivas
        # Asegurar que t_ao_venta_pos no contenga NaNs que afecten el cálculo de x0.
        # Si t_ao_venta_pos puede tener NaNs, x0_calc podría tener NaNs.
        # Y np.maximum(1.0, x0_calc) propagará esos NaNs.
        # La sigmoide con x0=NaN dará NaN. curve_fit no maneja bien los NaNs en y_data o en el resultado de la función.
        # Asumimos que df_fit ya ha sido limpiado de NaNs en T_AO_VENTA.
        x0_calc = x0_base - x0_factor_t_ao_venta * t_ao_venta_pos
        x0_final = np.maximum(1.0, x0_calc) # Asegura que x0 sea al menos 1

        # Calcular la sigmoide solo para dotaciones positivas
        # y donde x0_final no sea NaN (si t_ao_venta_pos pudo ser NaN y no se manejó)
        # Nota: si L, k, x0_base, x0_factor pueden llevar a NaN/inf, también hay que considerarlo.

        exp_term = -k * (dotacion_pos - x0_final)

        # Aplicar la función sigmoide de forma segura
        # Evitar overflow/underflow en np.exp y división por cero
        # Esto es una forma simplificada; una librería de funciones especiales podría tener sigmoides más robustas
        sigmoide_values = np.zeros_like(dotacion_pos, dtype=float)

        # Donde exp_term es muy grande (dotacion << x0), la efectividad tiende a 0
        # Donde exp_term es muy pequeño (dotacion >> x0), la efectividad tiende a L
        # Usamos límites aproximados para evitar NaN/Inf en exp()
        # L debe ser > 0 para que esto tenga sentido. El bound para L es [0.5, 1.0]

        # Para exp_term muy negativos (dotacion >> x0), exp(exp_term) -> 0, resultado -> L
        mask_exp_large_neg = exp_term < -700 # np.exp(-709) es aprox 1e-308, cercano a 0
        sigmoide_values[mask_exp_large_neg] = L

        # Para exp_term muy positivos (dotacion << x0), exp(exp_term) -> inf, resultado -> 0
        mask_exp_large_pos = exp_term > 700 # np.exp(709) es aprox 1e308, cercano a inf
        sigmoide_values[mask_exp_large_pos] = 0.0

        # Casos intermedios
        mask_intermediate = (~mask_exp_large_neg) & (~mask_exp_large_pos)
        if np.any(mask_intermediate):
             sigmoide_values[mask_intermediate] = L / (1 + np.exp(exp_term[mask_intermediate]))

        p_efectividad[idx_dotacion_positiva] = sigmoide_values

    return p_efectividad


def estimar_parametros_efectividad(df_historico):
    """
    Estima los parámetros del modelo de efectividad (_modelo_efectividad)
    basado en datos históricos de DOTACION, T_AO y T_AO_VENTA.
    Ahora incluye T_AO_VENTA como variable para la estimación.
    """
    df_fit = df_historico[df_historico['T_AO'] > 0].copy()
    # Es importante que DOTACION no sea negativa si la lógica del modelo lo asume.
    # Si DOTACION puede ser < 0 en los datos, se debe filtrar o aclarar cómo manejarlo.
    df_fit = df_fit[df_fit['DOTACION'] >= 0] # Asegurar dotaciones no negativas para el ajuste

    if df_fit.empty:
        return {'L': 1.0, 'k': 0.5, 'x0_base': 5.0, 'x0_factor_t_ao_venta': 0.05}

    df_fit['P_EFECTIVIDAD'] = calcular_efectividad(df_fit['T_AO'], df_fit['T_AO_VENTA'])

    # Evitar NaN o Inf en los datos de entrada para curve_fit
    df_fit = df_fit.dropna(subset=['DOTACION', 'T_AO_VENTA', 'P_EFECTIVIDAD'])
    df_fit = df_fit[~np.isinf(df_fit['P_EFECTIVIDAD'])]
    df_fit = df_fit[~np.isinf(df_fit['T_AO_VENTA'])] # También para T_AO_VENTA
    df_fit = df_fit[~np.isnan(df_fit['T_AO_VENTA'])]


    # Se necesitan al menos tantos puntos como parámetros a ajustar (4)
    # y que haya variabilidad en los datos.
    if df_fit.empty or len(df_fit) < 4:
        return {'L': 1.0, 'k': 0.5, 'x0_base': 5.0, 'x0_factor_t_ao_venta': 0.05}

    try:
        X_data = np.array([df_fit['DOTACION'].values, df_fit['T_AO_VENTA'].values])
        y_data = df_fit['P_EFECTIVIDAD'].values

        # Límites para los parámetros:
        bounds = ([0.5, 0.01, 1.0, -0.1], [1.0, 2.0, 20.0, 0.1])

        mean_dotacion_positiva = df_fit[df_fit['DOTACION'] > 0]['DOTACION'].mean()
        if pd.isna(mean_dotacion_positiva) or mean_dotacion_positiva <= 0:
            # Si no hay dotaciones positivas o la media es NaN/cero, usar un valor medio de los bounds
            initial_x0_base = (bounds[0][2] + bounds[1][2]) / 2
        else:
            initial_x0_base = np.clip(mean_dotacion_positiva, bounds[0][2], bounds[1][2])

        p0 = [1.0, 0.5, initial_x0_base, 0.01]

        # Validar que p0 esté dentro de los bounds
        for i in range(len(p0)):
            p0[i] = np.clip(p0[i], bounds[0][i], bounds[1][i])


        popt, pcov = curve_fit(_modelo_efectividad_fit, X_data, y_data, p0=p0, bounds=bounds, maxfev=10000, ftol=1e-6, xtol=1e-6)

        params = {
            'L': popt[0],
            'k': popt[1],
            'x0_base': popt[2],
            'x0_factor_t_ao_venta': popt[3]
        }
        return params
    except RuntimeError:
        return {'L': 1.0, 'k': 0.5, 'x0_base': 5.0, 'x0_factor_t_ao_venta': 0.05}
    except ValueError as e:
        print(f"ValueError en curve_fit: {e}")
        print(f"p0 usado: {p0 if 'p0' in locals() else 'No definido'}")
        print(f"bounds usados: {bounds if 'bounds' in locals() else 'No definido'}")
        return {'L': 1.0, 'k': 0.5, 'x0_base': 5.0, 'x0_factor_t_ao_venta': 0.05}


def estimar_dotacion_optima(t_ao_preds, t_ao_venta_preds, efectividad_deseada=0.8, params_efectividad=None):
    """
    Estima la dotación óptima para un conjunto de predicciones de T_AO y T_AO_VENTA
    para alcanzar una P_EFECTIVIDAD deseada.
    """
    if not hasattr(t_ao_preds, "__len__") or len(t_ao_preds) == 0 or \
       not hasattr(t_ao_venta_preds, "__len__") or len(t_ao_venta_preds) == 0:
        return 0, 0.0

    if params_efectividad is None:
        print("Advertencia: Usando parámetros de efectividad por defecto para estimar dotación óptima.")
        params_efectividad = {'L': 1.0, 'k': 0.5, 'x0_base': 5.0, 'x0_factor_t_ao_venta': 0.05}

    t_ao_preds_arr = np.array(t_ao_preds)
    t_ao_venta_preds_arr = np.array(t_ao_venta_preds)

    valid_indices = (t_ao_preds_arr > 0) & (~np.isnan(t_ao_preds_arr)) & (~np.isnan(t_ao_venta_preds_arr))
    if not np.any(valid_indices):
        return 0, 0.0 # No hay predicciones válidas para operar

    # Considerar solo las predicciones válidas para la optimización
    t_ao_venta_preds_valid = t_ao_venta_preds_arr[valid_indices]


    def objective_function(dotacion_candidata):
        dotacion_candidata = max(1.0, float(dotacion_candidata)) # Asegurar que sea float y al menos 1

        # Calcular la efectividad promedio para esta dotación candidata
        efectividades_predichas_list = []
        for t_ao_vp in t_ao_venta_preds_valid: # Iterar sobre los t_ao_venta válidos
            if pd.notna(t_ao_vp): # Solo calcular si t_ao_venta_pred es válido
                 efectividades_predichas_list.append(
                     _modelo_efectividad(dotacion_candidata, t_ao_vp, params_efectividad)
                 )

        if not efectividades_predichas_list: # Si no se pudieron calcular efectividades
            return 1.0 # Penalización alta (minimizar esta función)

        avg_efectividad = np.mean(efectividades_predichas_list)

        # Penalización por no alcanzar la efectividad deseada
        costo_efectividad = (avg_efectividad - efectividad_deseada)**2 # Cuadrado para penalizar más fuerte

        # Pequeña penalización por dotación para evitar valores excesivos si múltiples dotaciones dan similar efectividad
        costo_dotacion = 0.0001 * dotacion_candidata

        return costo_efectividad + costo_dotacion


    bounds_opt = (1, 30) # Rango de búsqueda para la dotación

    try:
        # Usar 'bounded' method que es bueno para optimización escalar con límites
        res = minimize_scalar(objective_function, bounds=bounds_opt, method='bounded')
        dotacion_optima = int(np.round(res.x)) if res.success else int(np.mean(bounds_opt))
    except Exception as e:
        print(f"Excepción en minimize_scalar: {e}")
        dotacion_optima = int(np.mean(bounds_opt))

    # Asegurar dotación mínima de 1 si hay trabajo y la optimización dio 0 (aunque bounds_opt empieza en 1)
    if dotacion_optima < 1 and np.sum(t_ao_preds_arr[valid_indices]) > 0 : # Suma de T_AO predichos válidos
        dotacion_optima = 1
    elif np.sum(t_ao_preds_arr[valid_indices]) <= 0: # Si no hay T_AO predicho, no se necesita dotación
        dotacion_optima = 0


    # Calcular la efectividad promedio resultante con la dotación óptima
    efectividad_promedio_resultante = 0.0
    if dotacion_optima > 0 and np.any(valid_indices):
        efect_list = []
        for t_ao_vp in t_ao_venta_preds_valid:
             if pd.notna(t_ao_vp):
                efect_list.append(_modelo_efectividad(dotacion_optima, t_ao_vp, params_efectividad))
        if efect_list:
            efectividad_promedio_resultante = np.mean(efect_list)

    # Si la dotación óptima es 0, la efectividad resultante también debe ser 0
    if dotacion_optima == 0:
        efectividad_promedio_resultante = 0.0

    return dotacion_optima, efectividad_promedio_resultante


def calcular_metricas_historicas(df):
    """
    Calcula métricas históricas de efectividad y dotación promedio.
    """
    if df.empty:
        return 0, 0, 0

    efectividad_historica = calcular_efectividad(df["T_AO"], df["T_AO_VENTA"]).mean()
    dotacion_promedio_historica = df["DOTACION"].mean()
    t_ao_promedio_historico = df["T_AO"].mean()
    return efectividad_historica, dotacion_promedio_historica, t_ao_promedio_historico