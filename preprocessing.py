import pandas as pd
import numpy as np
import holidays


def assign_turno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna turnos basados en la hora del día (9-21).
    1: 9-11, 2: 12-14, 3: 15-17, 4: 18-21, 0: fuera de rango.
    """
    bins  = [8, 11, 14, 17, 21]
    labels = [1, 2, 3, 4]
    df['turno'] = pd.cut(
        df['HORA'], bins=bins, labels=labels,
        right=True, include_lowest=True
    )
    df['turno'] = df['turno'].cat.add_categories([0]).fillna(0).astype(int)
    return df

def prepare_features(
    df: pd.DataFrame,
    target: str,
    is_prediction: bool = False,
    include_time_features: bool = False
):
    """
    Genera un set de features enriquecido para entrenamiento o predicción.
    - Temporal básico + festivos (Chile).
    - Lags, medias móviles y desviaciones.
    - Diferencias con mismo horario días atrás.
    - Codificación cíclica de hora e indicador de pico.
    """
    df = df.copy()
    # ------------------------------------
    # 1) FECHA a datetime, asignar turno
    # ------------------------------------
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    if 'HORA' in df.columns:
        df['HORA'] = df['HORA'].fillna(-1)
        df = assign_turno(df)
    sort_cols = ['FECHA', 'HORA'] if 'HORA' in df.columns else ['FECHA']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------
    # 2) Features temporales básicas
    # ------------------------------------
    df['year']      = df['FECHA'].dt.year
    df['month']     = df['FECHA'].dt.month
    df['day']       = df['FECHA'].dt.day
    df['weekday']   = df['FECHA'].dt.weekday
    df['dayofyear'] = df['FECHA'].dt.dayofyear
    df['weekofyear']= df['FECHA'].dt.isocalendar().week.astype(int)
    df['is_weekend']= (df['weekday'] >= 5).astype(int)

    # Festivos chilenos
    years = range(df['FECHA'].dt.year.min()-1, df['FECHA'].dt.year.max()+2)
    df['is_holiday'] = df['FECHA'].dt.date.isin(holidays.Chile(years=years)).astype(int)
    # =============================================================================
    # ❶  VARIABLES DE TIEMPO Y EVENTOS DE CALENDARIO
    # =============================================================================
    if include_time_features and "HORA" in df.columns:
        # --- codificación cíclica de la hora -------------------------------------
        H = df["HORA"]
        df = df.assign(
            sin_hour=np.sin(2 * np.pi * H / 24),
            cos_hour=np.cos(2 * np.pi * H / 24),
            is_peak=H.isin([11, 12, 13]).astype(int)
        )

        # --- variables de calendario --------------------------------------------
        D = df["FECHA"]
        df = df.assign(
            dayofweek=D.dt.dayofweek,  # 0 = lunes
            is_weekend=(D.dt.dayofweek >= 5).astype(int),
            is_month_end=D.dt.is_month_end.astype(int),
            is_month_start=D.dt.is_month_start.astype(int),
            is_payday=D.dt.day.isin([15, 30, 31]).astype(int)
        )

        # Feriados de Chile (opcional, requiere workalendar)
        try:
            from workalendar.america import Chile
            cal = Chile()
            df["is_holiday_CL"] = D.dt.date.map(cal.is_holiday).astype(int)
        except Exception:
            df["is_holiday_CL"] = 0  # fallback si no está workalendar

    # ------------------------------------
    # 3) Lags, rolling mean y rolling std
    # ------------------------------------
    lag_cols = ["T_VISITAS", "T_AO", "T_AO_VENTA", "DOTACION"]
    lags     = [1,2,3,6,12,24,168]
    YEAR_HOURS  = 24 * 365  # aproximación a un año
    MONTH_HOURS = 24 * 30   # aproximación a un mes
    lags.append(YEAR_HOURS)

    lag_data = {}
    for col in lag_cols:
        if col not in df.columns:
            df[col] = np.nan
        for lag in lags:
            lag_data[f"{col}_lag{lag}"] = df[col].shift(lag)
            lag_data[f"{col}_roll{lag}_mean"] = (
                df[col].shift(1).rolling(lag, min_periods=1).mean()
            )
            lag_data[f"{col}_roll{lag}_std"] = (
                df[col].shift(1).rolling(lag, min_periods=1).std().fillna(0)
            )
        lag_data[f"{col}_prev_year_month_mean"] = (
            df[col].shift(YEAR_HOURS).rolling(MONTH_HOURS, min_periods=1).mean()
        )

    df = pd.concat([df, pd.DataFrame(lag_data, index=df.index)], axis=1)
    df = df.copy()  # defragment frame

    # ------------------------------------
    # 4) Diferencias con días anteriores
    # ------------------------------------
    if 'T_VISITAS' in df.columns:
        df['diff_24h'] = df['T_VISITAS'] - df['T_VISITAS'].shift(24)
        df['diff_168h']= df['T_VISITAS'] - df['T_VISITAS'].shift(168)
    if 'T_AO' in df.columns:
        df['ao_diff_24h'] = df['T_AO'] - df['T_AO'].shift(24)
        df['ao_diff_168h']= df['T_AO'] - df['T_AO'].shift(168)

    # --- 5) Codificación cíclica e indicador pico ---
    if include_time_features and 'HORA' in df.columns:
        H = df['HORA']
        df = df.assign(
            sin_hour=np.sin(2 * np.pi * H / 24),
            cos_hour=np.cos(2 * np.pi * H / 24),
            is_peak=H.isin([11, 12, 13]).astype(int)
        )

    # ------------------------------------
    # 6) Lista final de features
    # ------------------------------------
    features = [
        # Temporales
        "year","month","day","weekday","dayofyear","weekofyear",
        "is_weekend","is_holiday",
        # Lags y rolls (se añaden dinámicamente)
    ]
    # Añadimos dinámicamente todas las columnas generadas con *_lag*, *_roll* y *_std*
    features += [c for c in df.columns if any(s in c for s in ["_lag","roll"])]
    # Columnas que utilizan información del mismo mes del año anterior
    features += [c for c in df.columns if "prev_year" in c]

    # Diferencias
    features += [c for c in df.columns if any(s in c for s in ["diff_24h","diff_168h"])]

    # Time features opcionales
    if include_time_features:
        features += ['turno', 'HORA', 'sin_hour', 'cos_hour', 'is_peak']

    # ------------------------------------
    # 7) Preparar X e y
    # ------------------------------------
    # Asegurar que existan todas las features
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0

    if is_prediction:
        X = df[features].fillna(0).copy()
        y = pd.Series([np.nan]*len(df), index=df.index)
    else:
        if target not in df.columns:
            raise ValueError(f"Objetivo '{target}' no encontrado en DataFrame")
        valid = df[target].notna()
        X = df.loc[valid, features].fillna(0).copy()
        y = df.loc[valid, target].copy()

    return X, y
