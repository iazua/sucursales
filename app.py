import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta
import plotly.express as px
from preprocessing import assign_turno, prepare_features
from utils import calcular_efectividad, estimar_dotacion_optima, estimar_parametros_efectividad
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- CONSTANTES ---
HORIZON_DAYS = 60
HOURS_RANGE  = list(range(9, 22))  # 9,10,...,21
MODEL_DIR    = "models_lgbm"

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Predicción de Dotación Óptima (Hourly)", layout="wide")
st.markdown(
    """
    <style>
    /* Fondo general */
    .stApp, .css-1d391kg {
      background-color: #1a0033;
    }
    /* DataFrame: fondo de la tabla y de las celdas */
    .stDataFrame div[role="table"] {
      background-color: #1a0033 !important;
      color: #FFFFFF;
    }
    /* Para los encabezados de tabla */
    .stDataFrame th {
      background-color: #330066 !important;
      color: #FFFFFF;
    }
    /* Para las gráficas, el contenedor externo */
    .stPlotlyChart div {
      background-color: #1a0033 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/27/Logo_Ripley_banco_2.png",
        width=500
    )


# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    df = pd.read_excel("data/DOTACION_EFECTIVIDAD.xlsx")
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])

    # Convertir COD_SUC a string y eliminar espacios
    df['COD_SUC'] = df['COD_SUC'].astype(str).str.strip()
    return df


df = load_data()

MESES_FUTURO = 12   # horizonte de predicción (puedes cambiarlo)

# ────────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ────────────────────────────────────────────────────────────────
def _simple_forecast_monthly(series: pd.Series, horizon: int) -> list[int]:
    """
    Tendencia lineal + semidesviación estándar alterna.
    (Se utiliza para T_AO)
    """
    if series.empty:
        return [0] * horizon

    deltas = series.diff().dropna().tail(min(12, len(series) - 1))
    slope  = deltas.mean() if not deltas.empty else 0.0
    last   = series.iloc[-1]

    sigma  = series.tail(min(12, len(series))).std(ddof=0)
    noise  = 0.5 * sigma

    return [
        max(0, round(last + i * slope + (noise if i % 2 else -noise)))
        for i in range(1, horizon + 1)
    ]


# -------- NUEVO MÉTODO PARA VISITAS ----------------------------------------
def _forecast_visitas_basic(series: pd.Series, horizon: int) -> list[int]:
    """
    Proyección moderada para T_VISITAS:
      • Punto de partida: media de los últimos 6 meses.
      • Pendiente: 25 % del cambio semestral.
      • Ruido: ± 0,25 σ alterno.
    """
    if series.empty:
        return [0] * horizon

    last6  = series.tail(min(6, len(series)))
    base   = last6.mean()
    slope  = 0.25 * ((series.iloc[-1] - last6.iloc[0]) / max(1, len(last6) - 1))
    sigma  = last6.std(ddof=0)
    noise  = 0.25 * sigma

    return [
        max(0, round(base + i * slope + (noise if i % 2 else -noise)))
        for i in range(1, horizon + 1)
    ]
# ---------------------------------------------------------------------------


def _pivot_with_totals(df_values: pd.DataFrame) -> pd.DataFrame:
    """Añade columna TOTAL y fila TOTAL a un DataFrame pivot."""
    df_values = df_values.copy()
    df_values["TOTAL"] = df_values.sum(axis=1)
    tot_col = df_values.sum(axis=0).to_frame().T
    tot_col.index = ["TOTAL"]
    return pd.concat([df_values, tot_col])

# ────────────────────────────────────────────────────────────────
#  1️⃣  VISITAS HISTÓRICAS
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Visitas totales por sucursal y mes (histórico)")

def tabla_visitas_hist(df_hist: pd.DataFrame):
    data = df_hist.copy()
    data["FECHA"] = pd.to_datetime(data["FECHA"])
    data["MES"]   = data["FECHA"].dt.to_period("M").astype(str)

    pivot = (
        data.groupby(["COD_SUC", "MES"])["T_VISITAS"]
        .sum().unstack("MES").fillna(0).astype(int)
    )
    pivot = pivot[sorted(pivot.columns)]

    st.dataframe(_pivot_with_totals(pivot),
                 use_container_width=True,
                 height=min(600, 60 + 30 * len(pivot) + 60))

tabla_visitas_hist(df)

# ────────────────────────────────────────────────────────────────
#  2️⃣  VISITAS – PREDICCIÓN (prom. 6 m + 25 % pendiente ± 0,25 σ)
# ────────────────────────────────────────────────────────────────
st.subheader(f"🔮 Predicción de Visitas "
             f"(promedio 6 m + 25 % pendiente, {MESES_FUTURO} meses)")

def tabla_visitas_pred(df_hist: pd.DataFrame, horizon: int = MESES_FUTURO):
    data = df_hist.copy()
    data["FECHA"] = pd.to_datetime(data["FECHA"])
    data["MES"]   = data["FECHA"].dt.to_period("M")

    serie_men = (
        data.groupby(["COD_SUC", "MES"])["T_VISITAS"]
        .sum().unstack("COD_SUC").sort_index()
    )

    fut_idx = pd.period_range(serie_men.index.max() + 1,
                              periods=horizon, freq="M")
    pred_df = pd.DataFrame(index=fut_idx, columns=serie_men.columns)

    for suc in serie_men.columns:
        pred_df[suc] = _forecast_visitas_basic(serie_men[suc], horizon)

    pivot = _pivot_with_totals(pred_df.T.astype(int))
    pivot.columns = [c.strftime("%Y-%m") if hasattr(c, "strftime") else c
                     for c in pivot.columns]

    st.dataframe(pivot,
                 use_container_width=True,
                 height=min(600, 60 + 30 * len(pivot)))

tabla_visitas_pred(df)

# ────────────────────────────────────────────────────────────────
#  3️⃣  OFERTAS ACEPTADAS HISTÓRICAS
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Ofertas aceptadas totales por sucursal y mes (histórico)")

def tabla_ao_hist(df_hist: pd.DataFrame):
    data = df_hist.copy()
    data["FECHA"] = pd.to_datetime(data["FECHA"])
    data["MES"]   = data["FECHA"].dt.to_period("M").astype(str)

    pivot = (
        data.groupby(["COD_SUC", "MES"])["T_AO"]
        .sum().unstack("MES").fillna(0).astype(int)
    )
    pivot = pivot[sorted(pivot.columns)]

    st.dataframe(_pivot_with_totals(pivot),
                 use_container_width=True,
                 height=min(600, 60 + 30 * len(pivot) + 60))

tabla_ao_hist(df)

# ────────────────────────────────────────────────────────────────
#  4️⃣  OFERTAS ACEPTADAS – PREDICCIÓN
# ────────────────────────────────────────────────────────────────
st.subheader(f"🔮 Predicción de Ofertas Aceptadas "
             f"(tendencia ± 0.5 σ, {MESES_FUTURO} meses)")

def tabla_ao_pred(df_hist: pd.DataFrame, horizon: int = MESES_FUTURO):
    data = df_hist.copy()
    data["FECHA"] = pd.to_datetime(data["FECHA"])
    data["MES"]   = data["FECHA"].dt.to_period("M")

    serie_men = (
        data.groupby(["COD_SUC", "MES"])["T_AO"]
        .sum().unstack("COD_SUC").sort_index()
    )

    fut_idx = pd.period_range(serie_men.index.max() + 1,
                              periods=horizon, freq="M")
    pred_df = pd.DataFrame(index=fut_idx, columns=serie_men.columns)

    for suc in serie_men.columns:
        pred_df[suc] = _simple_forecast_monthly(serie_men[suc], horizon)

    pivot = _pivot_with_totals(pred_df.T.astype(int))
    pivot.columns = [c.strftime("%Y-%m") if hasattr(c, "strftime") else c
                     for c in pivot.columns]

    st.dataframe(pivot,
                 use_container_width=True,
                 height=min(600, 60 + 30 * len(pivot)))

tabla_ao_pred(df)

# ────────────────────────────────────────────────────────────────
#  GRÁFICOS GLOBAL HISTÓRICO VS PRONÓSTICO
# ────────────────────────────────────────────────────────────────
def _series_total(df_hist: pd.DataFrame, var: str) -> pd.Series:
    data = df_hist.copy(); data["FECHA"] = pd.to_datetime(data["FECHA"])
    return (data.groupby(data["FECHA"].dt.to_period("M"))[var]
            .sum().sort_index())

hist_vis = _series_total(df, "T_VISITAS").tail(12)
hist_ao  = _series_total(df, "T_AO").tail(12)

pred_vis = pd.Series(
    _forecast_visitas_basic(hist_vis, MESES_FUTURO),
    index=pd.period_range(hist_vis.index.max() + 1,
                          periods=MESES_FUTURO, freq="M")
)
pred_ao  = pd.Series(
    _simple_forecast_monthly(hist_ao, MESES_FUTURO),
    index=pd.period_range(hist_ao.index.max() + 1,
                          periods=MESES_FUTURO, freq="M")
)

def _nombre_mes(period_idx):  # Period[M] → 'Enero', 'Febrero',...
    return period_idx.to_timestamp().month_name(locale="es")

def _fig_comparacion(hist: pd.Series, pred: pd.Series, titulo: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[_nombre_mes(m) for m in hist.index], y=hist.values,
        mode="lines+markers", name="Histórico", line=dict(width=2)))
    fig.add_trace(go.Scatter(
        x=[_nombre_mes(m) for m in pred.index], y=pred.values,
        mode="lines+markers", name="Pronóstico",
        line=dict(width=2, dash="dash")))
    fig.update_layout(
        title=titulo, xaxis_title="Mes", yaxis_title="Total",
        template="plotly_dark", hovermode="x unified",
        margin=dict(t=60, l=0, r=0, b=0))
    return fig

st.plotly_chart(_fig_comparacion(hist_vis, pred_vis,
                 "Visitas: histórico vs. pronóstico"),
                 use_container_width=True)
st.plotly_chart(_fig_comparacion(hist_ao, pred_ao,
                 "T_AO: histórico vs. pronóstico"),
                 use_container_width=True)


# Cargar y preparar df_regions
df_regions = pd.read_excel("data/regions.xlsx")
df_regions['COD_SUC'] = df_regions['COD_SUC'].astype(str).str.strip()

# Verificar duplicados
if df_regions['COD_SUC'].duplicated().any():
    st.warning("Advertencia: Hay valores duplicados en COD_SUC de regions.xlsx. Se conservará el primero.")
    df_regions = df_regions.drop_duplicates(subset='COD_SUC', keep='first')

# Merge seguro
df_map = (
    df
    .merge(df_regions, on='COD_SUC', how='left')
    .fillna({'ZONA': 'Desconocida'})
)

# 2) Sumamos visitas, ofertas aceptadas, ofertas aceptadas de venta y calculamos efectividad por zona
df_zona = (
    df_map
    .groupby('ZONA', as_index=False)
    .agg({
        'T_VISITAS':'sum',
        'T_AO':'sum',
        'T_AO_VENTA':'sum'
    })
)
df_zona['EFECTIVIDAD'] = df_zona['T_AO_VENTA'] / df_zona['T_AO']

# 3) Calculamos porcentajes y efectividad
total_vis = df_zona['T_VISITAS'].sum()
total_ao  = df_zona['T_AO'].sum()
total_ao_venta = df_zona['T_AO_VENTA'].sum()
df_zona['pct_vis'] = df_zona['T_VISITAS'] / total_vis
df_zona['pct_ao']  = df_zona['T_AO'] / total_ao
df_zona['pct_ao_venta'] = df_zona['T_AO_VENTA'] / total_ao_venta
df_zona['pct_efectividad'] = df_zona['EFECTIVIDAD']  # Ya está en formato 0-1
df_zona['label_vis'] = df_zona['pct_vis'].apply(lambda x: f"{x:.1%}")
df_zona['label_ao']  = df_zona['pct_ao'].apply(lambda x: f"{x:.1%}")
df_zona['label_ao_venta'] = df_zona['pct_ao_venta'].apply(lambda x: f"{x:.1%}")
df_zona['label_efectividad'] = df_zona['pct_efectividad'].apply(lambda x: f"{x:.1%}")

# 4) Centroides aproximados por zona
centroides = {
    'Norte':      (-20.0, -70.0),
    'Centro':     (-32.5, -71.5),
    'Sur':        (-38.0, -73.0),
    'Santiago':   (-33.45, -70.65),
    'Desconocida':(-33.0, -70.0)
}
df_zona['lat'] = df_zona['ZONA'].map(lambda z: centroides.get(z, centroides['Desconocida'])[0])
df_zona['lon'] = df_zona['ZONA'].map(lambda z: centroides.get(z, centroides['Desconocida'])[1])

# 5) Definimos cuatro capas: visitas, ofertas aceptadas, ofertas aceptadas de venta y efectividad
layer_vis = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color='[200, 30, 30, 160]',
    get_radius='pct_vis * 400000',
    auto_highlight=True,
)

layer_ao = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color='[200, 30, 30, 160]',
    get_radius='pct_ao * 400000',
    auto_highlight=True,
)

layer_ao_venta = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color='[200, 30, 30, 160]',
    get_radius='pct_ao_venta * 400000',
    auto_highlight=True,
)

layer_efectividad = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color='[200, 30, 30, 160]',
    get_radius='pct_efectividad * 10000',
    auto_highlight=True,
)

# 6) Vista centrada en Chile con rotación de 90° hacia la derecha
view_state = pdk.ViewState(
    latitude=-30.5,
    longitude=-70.9,
    zoom=5,
    pitch=0,
    bearing=90
)

# 7) Renderizamos las cuatro capas en un solo mapa con la nueva vista
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=[layer_vis, layer_ao, layer_ao_venta, layer_efectividad],
    tooltip={
        "html": (
            "<b>Zona:</b> {ZONA}<br>"
            "<b>Visitas:</b> {T_VISITAS} ({label_vis})<br>"
            "<b>Ofertas aceptadas:</b> {T_AO} ({label_ao})<br>"
            "<b>Ofertas aceptadas de venta:</b> {T_AO_VENTA} ({label_ao_venta})<br>"
            "<b>Efectividad:</b> {label_efectividad}"
        ),
        "style": {"backgroundColor":"#F0F0F0","color":"#000"}
    }
), use_container_width=True)

st.title("🔍 Predicción de Dotación y Efectividad por Hora")

# --- CONTROL DE HORIZONTE DE PROYECCIÓN ---
days_proj = st.slider(
    "Días a proyectar",
    min_value=1, max_value=HORIZON_DAYS,
    value=30, step=1,
)
sucursales = sorted(df["COD_SUC"].unique())
cod_suc     = st.selectbox("Selecciona una sucursal", sucursales)

# --- FILTRAR Y PREPARAR HISTÓRICO ---
df_suc = df[df["COD_SUC"] == cod_suc].copy()
df_suc = df_suc.sort_values(["FECHA", "HORA"] if "HORA" in df_suc.columns else ["FECHA"]).reset_index(drop=True)

# Asegurar P_EFECTIVIDAD histórica
if "P_EFECTIVIDAD" not in df_suc.columns:
    df_suc["P_EFECTIVIDAD"] = calcular_efectividad(df_suc["T_AO"], df_suc["T_AO_VENTA"])

# Promedio de efectividad con DOTACION=1 (para fallback)
df_dot1 = df_suc[df_suc["DOTACION"] == 1]
avg_eff_dot1 = df_dot1["P_EFECTIVIDAD"].mean() if not df_dot1.empty else np.nan

# --- SLIDER DE EFECTIVIDAD OBJETIVO ---
efectividad_obj = st.slider(
    "Efectividad objetivo (P_EFECTIVIDAD):",
    min_value=0.0, max_value=1.0, value=0.62, step=0.01, format="%.2f"
)

# — CARGA DE MODELOS —
@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(var: str, cod: str):
    """Carga desde disco un modelo LightGBM específico y lo cachea."""
    path = os.path.join(MODEL_DIR, f"predictor_{var}_{cod}.pkl")
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def load_models(cod: str) -> dict[str, lgb.LGBMRegressor | None]:
    """Obtiene los dos modelos requeridos para una sucursal."""
    return {var: load_model(var, cod) for var in ("T_VISITAS", "T_AO")}


models = load_models(cod_suc)
if None in models.values():
    st.error("No se encontraron ambos modelos para esta sucursal en models_lgbm/")
    st.stop()

# --- app.py ---

@st.cache_data(show_spinner="Generando pronóstico…")
def forecast_hourly(df_hist: pd.DataFrame,
                    cod_suc: int,
                    efect_obj: float,
                    days: int,
                    _models: dict[str, lgb.LGBMRegressor]) -> pd.DataFrame:
    """
    Predicción autoregresiva hora a hora sobre `days` días futuros.
    Incluye variables de hora y NO descarta las columnas temporales.
    """

    # --- parámetros de efectividad (sin cambios) ----------------------------
    hist_params = df_hist[["DOTACION", "T_AO", "T_AO_VENTA"]].dropna()
    params_sig = (
        estimar_parametros_efectividad(hist_params)
        if len(hist_params) >= 3
        else {"L": 1.0, "k": 0.5, "x0_base": 5.0, "x0_factor_t_ao_venta": 0.05}
    )

    # --- grid futuro FECHA×HORA --------------------------------------------
    last_date = df_hist["FECHA"].max()
    fut_rows = [
        {"FECHA": last_date + timedelta(days=d), "HORA": h, "COD_SUC": cod_suc}
        for d in range(1, days + 1) for h in HOURS_RANGE
    ]
    df_fut  = assign_turno(pd.DataFrame(fut_rows))
    df_comb = pd.concat([df_hist, df_fut], ignore_index=True, sort=False)

    preds_vis, preds_ao = [], []
    start_idx = len(df_hist)

    # --- bucle autoregresivo hora a hora ------------------------------------
    for step in range(len(df_fut)):
        idx = start_idx + step

        # VISITAS ------------------------------------------------------------
        X_vis, _ = prepare_features(
            df_comb.iloc[: idx + 1].copy(),
            target="T_VISITAS",
            is_prediction=True,
            include_time_features=True      # ← ①
        )
        X_vis = X_vis.iloc[[-1]]           # ya NO se dropea 'HORA'

        # ACEPTA OFERTA ------------------------------------------------------
        X_ao, _ = prepare_features(
            df_comb.iloc[: idx + 1].copy(),
            target="T_AO",
            is_prediction=True,
            include_time_features=True      # ← ①
        )
        X_ao = X_ao.iloc[[-1]]

        # predicciones
        y_vis = _models["T_VISITAS"].predict(X_vis)[0]
        y_ao  = _models["T_AO"].predict(X_ao)[0]

        preds_vis.append(y_vis)
        preds_ao.append(y_ao)

        # inyectar para el siguiente paso
        df_comb.at[idx, "T_VISITAS"] = y_vis
        df_comb.at[idx, "T_AO"]      = y_ao

    # --- ensamblar salida ---------------------------------------------------
    df_out = df_fut.copy().reset_index(drop=True)
    df_out["T_VISITAS_pred"] = preds_vis
    df_out["T_AO_pred"]      = preds_ao

    df_out["T_AO_VENTA_req"]    = df_out["T_AO_pred"] * efect_obj
    df_out["P_EFECTIVIDAD_req"] = calcular_efectividad(
        df_out["T_AO_pred"], df_out["T_AO_VENTA_req"]
    )

    # dotación óptima
    dots, effs = [], []
    for _, r in df_out.iterrows():
        dot, eff = estimar_dotacion_optima(
            [r["T_AO_pred"]], [r["T_AO_VENTA_req"]],
            efect_obj, params_sig
        )
        dots.append(dot); effs.append(eff)

    df_out["DOTACION_req"]      = dots
    df_out["P_EFECTIVIDAD_opt"] = effs
    return df_out



# ---------- LLAMADA ----------
df_pred = forecast_hourly(
    df_suc,              # histórico
    cod_suc,
    efectividad_obj,
    HORIZON_DAYS,
    models               # se pasa como _models, pero no se hashea
)



# ——— TABLA POR HORA ———
st.subheader(f"📈 Predicciones para los próximos {HORIZON_DAYS} días")
st.subheader("Por hora")

# 1) Seleccionamos únicamente las columnas de df_pred que necesitamos
df_hourly = df_pred[[
    "FECHA",
    "HORA",
    "T_VISITAS_pred",
    "T_AO_pred",
    "T_AO_VENTA_req",
    "P_EFECTIVIDAD_req"
]].copy()

# 2) Formateamos FECHA y añadimos día de la semana
df_hourly["Fecha registro"] = df_hourly["FECHA"].dt.strftime("%d-%m-%Y")
df_hourly["Día"]            = df_hourly["FECHA"].dt.day_name(locale="es")

# 3) Renombramos cada métrica de forma explícita
df_hourly = df_hourly.rename(columns={
    "HORA":                  "Hora",
    "T_VISITAS_pred":        "Visitas estimadas",
    "T_AO_pred":             "Ofertas aceptadas estimadas",
    "T_AO_VENTA_req":        "Ventas requeridas",
    "P_EFECTIVIDAD_req":     "% Efectividad requerida",
})

# 4) Redondeamos y transformamos tipos
df_hourly["Visitas estimadas"]           = df_hourly["Visitas estimadas"].round(0).astype(int)
df_hourly["Ofertas aceptadas estimadas"] = df_hourly["Ofertas aceptadas estimadas"].round(0).astype(int)
df_hourly["Ventas requeridas"]           = df_hourly["Ventas requeridas"].round(0).astype(int)
df_hourly["% Efectividad requerida"]     = df_hourly["% Efectividad requerida"].round(2)

# 5) Seleccionamos el orden final de columnas
df_hourly = df_hourly[[
    "Fecha registro", "Día", "Hora",
    "Visitas estimadas", "Ofertas aceptadas estimadas",
    "Ventas requeridas", "% Efectividad requerida"
]]

st.dataframe(df_hourly, use_container_width=True)

# ——— TABLA POR DÍA ———
st.subheader("Por día")

df_daily = (
    df_hourly
    .groupby(["Fecha registro", "Día"], as_index=False)
    .agg({
        "Visitas estimadas":           "sum",
        "Ofertas aceptadas estimadas": "sum",
        "Ventas requeridas":           "sum",
        "% Efectividad requerida":     "mean"
    })
)

# Redondeo final de efectividad
df_daily["% Efectividad requerida"] = df_daily["% Efectividad requerida"].round(2)

# Orden cronológico
df_daily["_dt"] = pd.to_datetime(df_daily["Fecha registro"], format="%d-%m-%Y")
df_daily = df_daily.sort_values("_dt").drop(columns="_dt")

st.dataframe(df_daily, use_container_width=True)

# --- CURVA DE EFECTIVIDAD vs. DOTACIÓN (Teórica) ---
st.subheader("Curva de Efectividad vs. Dotación")

# 1. Estimar parámetros históricos
hist = df_suc[['DOTACION','T_AO','T_AO_VENTA']].dropna()
if len(hist) >= 3:
    params_eff = estimar_parametros_efectividad(hist)
else:
    params_eff = {'L':1.0, 'k':0.5, 'x0_base':5.0, 'x0_factor_t_ao_venta':0.05}

L       = params_eff['L']
k_def   = params_eff['k']
x0_base = params_eff['x0_base']
x0_fac  = params_eff['x0_factor_t_ao_venta']

# 2. Parámetro k personalizable vía entero (se divide internamente entre 100)
# El usuario ingresa, por ejemplo, 50 para k=0.50
k_def_int = int(k_def * 100)
k_int = st.number_input(
    "Coeficiente k = Pendiente de la Curva (T_AO)",
    min_value=0, max_value=2000,
    value=k_def_int,
    step=1
)
k = k_int / 100.0

# 3. Rango de dotación
min_dot   = st.number_input("Dotación mínima", 1, 100, 1, 1)
max_dot   = st.number_input("Dotación máxima", min_dot, 100, 9, 1)
dot_range = np.arange(min_dot, max_dot+1)

# 4. Calcular x0 recalibrado usando promedio de Ventas requeridas
avg_ventas = np.nanmean(df_pred["T_AO_VENTA_req"]) if 'T_AO_VENTA_req' in df_pred else np.nan
x0_theo = x0_base if np.isnan(avg_ventas) or avg_ventas <= 0 else max(1.0, x0_base - x0_fac * avg_ventas)

# 5. Definir funciones con k dinámico
def sigmoid(x, x0):
    return 0.0 if x <= 0 else L / (1 + np.exp(-k * (x - x0)))

def gompertz(x, x0):
    return 0.0 if x <= 0 else L * np.exp(-np.exp(-k * (x - x0)))

# 6. Generar curva teórica
ef_sig = np.array([sigmoid(x, x0_theo) for x in dot_range])
ef_gom = np.array([gompertz(x, x0_theo) for x in dot_range])

df_curve = pd.DataFrame({
    "Dotación":    np.tile(dot_range, 2),
    "Modelo":      ["Sigmoide"] * len(dot_range) + ["Gompertz"] * len(dot_range),
    "Efectividad": np.concatenate([ef_sig, ef_gom])
})

# 7. Graficar con fondo púrpura si aplica
fig = px.line(
    df_curve,
    x="Dotación", y="Efectividad",
    color="Modelo",
    labels={"Dotación":"Dotación","Efectividad":"Efectividad"},
    title=" "
)
fig.update_layout(
    paper_bgcolor="#1a0033",
    plot_bgcolor="#1a0033",
    font_color="#FFFFFF",
    title_font_color="#FFFFFF"
)
st.plotly_chart(fig, use_container_width=True)

df_display = df_pred.copy()

# Formatear la fecha y renombrar columnas como en la tabla horaria
df_display["DÍA"] = df_display["FECHA"].dt.strftime("%d-%m-%Y")
df_display = df_display.rename(columns={
    "T_AO_pred": "Ofertas aceptadas estimadas",
    "HORA": "Hora",
    "T_VISITAS_pred": "Visitas estimadas",
    "T_AO_VENTA_req": "Ventas requeridas",
    "P_EFECTIVIDAD_req": "% Efectividad requerida",
})

# --- ANÁLISIS HISTÓRICO PONDERADO POR DÍA DE LA SEMANA ---


st.header("🔍 Flujo histórico ponderado por día de la semana")

# Mapear nombres de día
dias_map = {
    'Monday':   'Lunes',
    'Tuesday':  'Martes',
    'Wednesday':'Miércoles',
    'Thursday': 'Jueves',
    'Friday':   'Viernes',
    'Saturday': 'Sábado',
    'Sunday':   'Domingo'
}

df = df_suc.copy()
df['DíaSemana'] = df['FECHA'].dt.day_name().map(dias_map)
df['TipoDia']   = np.where(df['FECHA'].dt.weekday < 5, 'Semana', 'Fin de Semana')

# Factor de ponderación: 2 días de fin de semana para cada día de semana, 5 días de semana para cada día de fin de semana
df['Factor'] = np.where(df['TipoDia']=='Semana', 2, 5)

# Agregar sumas y aplicar factor
grouped = (
    df
    .groupby('DíaSemana', observed=True)
    .agg(
        T_VISITAS_raw=('T_VISITAS','sum'),
        T_AO_raw=('T_AO','sum'),
        Factor=('Factor','first')  # mismo factor por grupo
    )
    .reset_index()
)
grouped = (
    df
    .groupby('DíaSemana', observed=True)
    .agg(
        T_VISITAS=('T_VISITAS','sum'),
        T_AO=('T_AO','sum')
    )
    .reset_index()
)

fig = px.bar(
    grouped,
    x='DíaSemana',
    y=['T_VISITAS', 'T_AO'],
    barmode='group',
    labels={
        'value': 'Total registrado',
        'variable': 'Métrica',
        'DíaSemana': 'Día de la semana'
    },
    title='Total de Visitas y Ofertas Aceptadas por Día de la Semana'
)

# Cambiar las etiquetas de la leyenda
fig.for_each_trace(lambda t: t.update(name='Visitas' if t.name == 'T_VISITAS' else 'Acepta Oferta'))

st.plotly_chart(fig, use_container_width=True)


# --- DISTRIBUCIÓN SEMANA vs. FIN DE SEMANA PONDERADA ---
st.header("📊 Semana vs Fin de Semana (ponderado)")

dist = (
    df
    .groupby('TipoDia', observed=True)
    .agg(
        T_VISITAS_raw=('T_VISITAS','sum'),
        T_AO_raw=('T_AO','sum'),
        Factor=('Factor','first')
    )
    .reset_index()
)
dist['T_VISITAS_pond'] = dist['T_VISITAS_raw'] * dist['Factor']
dist['T_AO_pond']      = dist['T_AO_raw']      * dist['Factor']

# Alinear ambos pie charts lado a lado con el mismo tamaño
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        px.pie(
            dist,
            names='TipoDia',
            values='T_VISITAS_pond',
            title='Proporción de Visitas ponderadas: Semana vs Fin de Semana',
            hole=0.4
        ).update_layout(height=600),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        px.pie(
            dist,
            names='TipoDia',
            values='T_AO_pond',
            title='Proporción de Ofertas Aceptadas ponderadas: Semana vs Fin de Semana',
            hole=0.4
        ).update_layout(height=600),
        use_container_width=True
    )



# --- GRÁFICO 1: Ofertas Aceptadas diario ---
st.subheader("📈 Histórico y Predicción de Ofertas Aceptadas (diario)")

# Agrupar histórico por fecha
hist_ao = (
    df_suc
    .groupby('FECHA', observed=True)['T_AO']
    .sum()
    .reset_index()
    .rename(columns={'T_AO':'Valor'})
    .assign(Tipo='Histórico')
)

# Agrupar predicción por fecha
pred_ao = (
    df_display
    .groupby('DÍA', observed=True)['Ofertas aceptadas estimadas']
    .sum()
    .reset_index()
    .rename(columns={'DÍA':'FECHA','Ofertas aceptadas estimadas':'Valor'})
)
pred_ao['FECHA'] = pd.to_datetime(pred_ao['FECHA'], format='%d-%m-%Y')
pred_ao = pred_ao.sort_values('FECHA').head(days_proj).assign(Tipo='Predicción')

# Combinar y pivotar
df_plot_ao = pd.concat([hist_ao, pred_ao], ignore_index=True)
df_pivot_ao = df_plot_ao.pivot_table(
    index='FECHA', columns='Tipo', values='Valor', aggfunc='sum'
)
st.line_chart(df_pivot_ao)

# --- GRÁFICO 2: Ventas Concretadas diario ---
st.subheader("📈 Histórico y Predicción de Ventas Concretadas (diario)")

# Agrupar histórico de ventas
hist_v = (
    df_suc
    .groupby('FECHA', observed=True)['T_AO_VENTA']
    .sum()
    .reset_index()
    .rename(columns={'T_AO_VENTA':'Valor'})
    .assign(Tipo='Histórico')
)

# Agrupar predicción de ventas requeridas
pred_v = (
    df_display
    .groupby('DÍA', observed=True)['Ventas requeridas']
    .sum()
    .reset_index()
    .rename(columns={'DÍA':'FECHA','Ventas requeridas':'Valor'})
)
pred_v['FECHA'] = pd.to_datetime(pred_v['FECHA'], format='%d-%m-%Y')
pred_v = pred_v.sort_values('FECHA').head(days_proj).assign(Tipo='Requerida')

# Combinar y pivotar
df_plot_v = pd.concat([hist_v, pred_v], ignore_index=True)
df_pivot_v = df_plot_v.pivot_table(
    index='FECHA', columns='Tipo', values='Valor', aggfunc='sum'
)
st.line_chart(df_pivot_v)


# --- Agregar selector de rango de días al inicio ---
st.markdown("---")
st.subheader("🔍 Selección de rango de análisis")

# Opciones para el dropdown
opciones_rango = {
    "Últimos 30 días": 30,
    "Últimos 60 días": 60,
    "Últimos 90 días": 90,
    "Toda la data disponible": None
}

# Crear el selector
rango_seleccionado = st.selectbox(
    "Selecciona el rango de días para el análisis:",
    options=list(opciones_rango.keys()),
    index=2  # Por defecto selecciona 90 días
)

# Obtener el valor numérico correspondiente
dias_analisis = opciones_rango[rango_seleccionado]

# Función para filtrar el dataframe según el rango seleccionado
def filtrar_por_rango(df, dias):
    if dias is None:
        return df  # No filtrar si es toda la data
    fecha_corte = df['FECHA'].max() - timedelta(days=dias)
    return df[df['FECHA'] >= fecha_corte]

# Filtrar df_suc según el rango seleccionado
df_suc_filtrado = filtrar_por_rango(df_suc.copy(), dias_analisis)


# ——— Análisis por turno ———
st.markdown("---")
st.subheader("📊 Visitas y Acepta Oferta promedio por turno")

# Generamos la columna 'turno' a partir de df_suc_filtrado
df_turnos = assign_turno(df_suc_filtrado.copy())

# Métricas originales (para la tabla)
metrics = ['T_VISITAS', 'T_AO', 'DOTACION', 'P_EFECTIVIDAD']

# Agrupamos y calculamos medias
res_turno = (
    df_turnos
    .groupby('turno')[metrics]
    .mean()
    .reset_index()
)
res_turno['Turno'] = res_turno['turno'].map({
    0: 'Fuera rango',
    1: '9–11',
    2: '12–14',
    3: '15–17',
    4: '18–21'
})

# — Gráfico: solo T_VISITAS y T_AO, con renombrado de etiquetas —
metrics_graph = ['T_VISITAS', 'T_AO']
fig = px.bar(
    res_turno,
    x='Turno',
    y=metrics_graph,
    barmode='group',
    labels={
        'T_VISITAS': 'Visitas',
        'T_AO': 'Acepta Oferta',
        'value': 'Promedio',
        'variable': 'Métrica'
    },
    title=f"Visitas y Acepta Oferta promedio por franja horaria ({rango_seleccionado})"
)
st.plotly_chart(fig, use_container_width=True)


# ——— KPI de conversión visitas → ofertas aceptadas por turno ———
st.markdown("---")
st.subheader("📈 Conversión de visitas a ofertas aceptadas por turno")

# 1) Calculamos la conversión: sum(T_AO) / sum(T_VISITAS) por turno
conv = (
    df_turnos
    .groupby('turno')
    .agg({'T_VISITAS':'sum', 'T_AO':'sum'})
    .reset_index()
)
conv['conversion'] = conv.apply(
    lambda row: row['T_AO'] / row['T_VISITAS'] if row['T_VISITAS'] > 0 else 0,
    axis=1
)

# 2) Mapeo de etiquetas de turno
etiquetas = {1:'9–11', 2:'12–14', 3:'15–17', 4:'18–21'}
conv['rango_horas'] = conv['turno'].map(etiquetas)

# 3) Mostramos cada turno como tarjeta KPI
cols = st.columns(4)
for i, turno in enumerate([1, 2, 3, 4], start=1):
    pct = conv.loc[conv['turno'] == turno, 'conversion'].iloc[0]
    with cols[i-1]:
        st.metric(
            label=f"Turno {etiquetas[turno]}",
            value=f"{pct:.2%}"
        )

# ——— Heatmap de conversión visitas → ofertas aceptadas por día y turno ———
st.markdown("---")
st.subheader("🌡️ Heatmap de conversión por día de la semana y turno")

# 1) Calcular conversión por día de la semana y turno
df_turnos['DiaSemana'] = df_turnos['FECHA'].dt.day_name(locale="es")
conv_dt = (
    df_turnos
    .groupby(['DiaSemana','turno'], observed=True)
    .agg(T_VISITAS=('T_VISITAS','sum'), T_AO=('T_AO','sum'))
    .reset_index()
)
conv_dt['Conversion'] = conv_dt.apply(
    lambda r: r['T_AO'] / r['T_VISITAS'] if r['T_VISITAS'] > 0 else 0,
    axis=1
)

# 2) Mapeo al español y orden de días y turnos
dias = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
turnos = {1:'9–11',2:'12–14',3:'15–17',4:'18–21'}
conv_dt['DiaSemana'] = pd.Categorical(conv_dt['DiaSemana'], categories=dias, ordered=True)
conv_dt['Turno'] = conv_dt['turno'].map(turnos)

# 3) Pivot para matriz DíaSemana × Turno
pivot_conv = (
    conv_dt
    .pivot(index='DiaSemana', columns='Turno', values='Conversion')
    .loc[dias, list(turnos.values())]
)

# 4) Convertir a % para mostrar
pivot_pct = pivot_conv * 100

# 5) Dibujar heatmap con Plotly Express
fig = px.imshow(
    pivot_pct,
    color_continuous_scale='Greens',
    labels={'x':'Turno','y':'Día de la semana','color':'Conversión (%)'},
    title=f'Conversión visitas → ofertas aceptadas (%) ({rango_seleccionado})',
    aspect='auto'
)
fig.update_xaxes(side='top')
fig.update_layout(
    plot_bgcolor='#1a0033',
    paper_bgcolor='#1a0033',
    font_color='#FFFFFF',
    title_font_color='#FFFFFF'
)

st.plotly_chart(fig, use_container_width=True)

# ——— Serie de efectividad diaria por turno ———
# 1) Calculamos df_ts como antes
df_ts = (
    df_turnos
    .groupby([df_turnos['FECHA'].dt.date.rename('Fecha'), 'turno'])
    .agg({'P_EFECTIVIDAD':'mean'})
    .reset_index()
)
df_ts['Turno'] = df_ts['turno'].map({1:'9–11',2:'12–14',3:'15–17',4:'18–21'})

# 2) Creamos 4 filas, 1 columna, ejes X compartidos
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.25,0.25,0.25,0.25],
    subplot_titles=['Turno 9–11','Turno 12–14','Turno 15–17','Turno 18–21']
)

# 3) Añadimos la línea por cada turno
for i, turno in enumerate(['9–11','12–14','15–17','18–21'], start=1):
    df_sub = df_ts[df_ts['Turno']==turno]
    fig.add_trace(
        go.Scatter(
            x=df_sub['Fecha'],
            y=df_sub['P_EFECTIVIDAD'],
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=4)
        ),
        row=i, col=1
    )
    fig.update_yaxes(row=i, col=1, tickformat='.2f', title_text='Efectividad')

# 4) Solo la última fila muestra etiquetas X
fig.update_xaxes(row=4, col=1, tickangle=-45, title_text='Fecha')

# 5) Diseño general
fig.update_layout(
    height=900,
    showlegend=False,
    title_text=f'Efectividad diaria por turno ({rango_seleccionado})',
    plot_bgcolor='#1a0033',
    paper_bgcolor='#1a0033',
    font_color='#FFFFFF',
    margin=dict(t=80, b=40, l=60, r=40)
)

st.plotly_chart(fig, use_container_width=True)

# ——— Efectividad promedio por día de la semana y turno ———
st.markdown("---")
st.subheader("📊 Efectividad promedio por día de la semana y por turno")

# 1) Preparamos día de la semana y mapeo de turnos
dias_map = {
    'Monday':   'Lunes', 'Tuesday':  'Martes', 'Wednesday':'Miércoles',
    'Thursday': 'Jueves','Friday':   'Viernes','Saturday': 'Sábado','Sunday': 'Domingo'
}
df_turnos['DíaSemana'] = df_turnos['FECHA'].dt.day_name().map(dias_map)
turno_map = {1:'9–11',2:'12–14',3:'15–17',4:'18–21',0:'Fuera rango'}
df_turnos['Turno'] = df_turnos['turno'].map(turno_map)

# 2) Calculamos el promedio
df_dia_turno = (
    df_turnos
    .groupby(['DíaSemana','Turno'], observed=True)['P_EFECTIVIDAD']
    .mean()
    .reset_index()
    .rename(columns={'P_EFECTIVIDAD':'Efectividad'})
)

# 3) Orden de días y turnos
orden_dias = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
orden_turnos = ['9–11','12–14','15–17','18–21']

df_dia_turno['DíaSemana'] = pd.Categorical(df_dia_turno['DíaSemana'], categories=orden_dias, ordered=True)
df_dia_turno['Turno']     = pd.Categorical(df_dia_turno['Turno'],     categories=orden_turnos, ordered=True)

# 4) Redondeo
df_dia_turno['Efectividad'] = df_dia_turno['Efectividad'].round(2)

# 5) Plot
fig = px.bar(
    df_dia_turno,
    x='DíaSemana',
    y='Efectividad',
    color='Turno',
    barmode='group',
    category_orders={'DíaSemana': orden_dias, 'Turno': orden_turnos},
    labels={
        'DíaSemana':'Día de la semana',
        'Efectividad':'Efectividad promedio',
        'Turno':'Franja horaria'
    },
    title=f'Efectividad promedio por día de la semana y por turno ({rango_seleccionado})'
)
fig.update_layout(
    yaxis_tickformat='.2f',
    plot_bgcolor='#1a0033',
    paper_bgcolor='#1a0033',
    font_color='#FFFFFF',
    title_font_color='#FFFFFF',
    legend_title_font_color='#FFFFFF'
)
st.plotly_chart(fig, use_container_width=True)

# ——— Boxplot de productividad (T_AO / DOTACION) por turno ———
st.markdown("---")
st.subheader("📊 Relación de Ofertas Aceptadas vs. Dotación")

# 1) Calculamos la productividad por registro
df_prod = df_turnos.copy()
df_prod['Productividad'] = df_prod.apply(
    lambda r: r['T_AO'] / r['DOTACION'] if r['DOTACION'] > 0 else 0,
    axis=1
)

# 2) Mapear turno numérico a rango horario
turno_map = {1:'9–11', 2:'12–14', 3:'15–17', 4:'18–21', 0:'Fuera rango'}
df_prod['Turno'] = df_prod['turno'].map(turno_map)

# 3) Creamos el boxplot
fig = px.box(
    df_prod,
    x='Turno',
    y='Productividad',
    points='outliers',
    labels={
        'Turno': 'Franja horaria',
        'Productividad': 'T_AO / DOTACION'
    },
    title=' '
)

# 4) Estilo acorde al tema oscuro
fig.update_layout(
    plot_bgcolor='#1a0033',
    paper_bgcolor='#1a0033',
    font_color='#FFFFFF',
    title_font_color='#FFFFFF',
    yaxis_tickformat='.2f'
)

st.plotly_chart(fig, use_container_width=True)

# ——— Recomendaciones automáticas basadas en demanda (T_AO) vs dotación ———
st.markdown("---")
st.subheader("💡 Recomendaciones de dotación según demanda por turno")

# 1) Clonamos y calculamos la carga por agente
df_carga = df_turnos.copy()
df_carga['Carga'] = df_carga.apply(
    lambda r: r['T_AO'] / r['DOTACION'] if r['DOTACION'] > 0 else np.nan,
    axis=1
)

# 2) Mapear turno numérico a rango horario (imprescindible antes de agrupar)
df_carga['Turno'] = df_carga['turno'].map(turno_map)

# 3) Estadísticas globales por turno: mediana e IQR de la carga
stats = (
    df_carga
    .groupby('Turno')['Carga']
    .agg(
        Mediana='median',
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )
    .reset_index()
)
stats['IQR'] = stats['Q3'] - stats['Q1']

# 4) Demanda media por día de la semana y turno
df_carga['DíaSemana'] = df_carga['FECHA'].dt.day_name().map(dias_map)
med_dia_turno = (
    df_carga
    .groupby(['Turno','DíaSemana'], observed=True)['Carga']
    .mean()
    .reset_index()
)

# 5) Generamos recomendaciones
recs = []
for _, row in stats.iterrows():
    turno = row['Turno']
    med   = row['Mediana']
    iqr   = row['IQR']
    sub   = med_dia_turno[med_dia_turno['Turno']==turno]
    peor  = sub.loc[sub['Carga'].idxmax()]
    mejor = sub.loc[sub['Carga'].idxmin()]

    # Lógica de acción
    if med > 1.5:
        accion = "🔴 Aumentar dotación"
    elif med < 0.8:
        accion = "🟢 Reducir dotación"
    else:
        accion = "🟡 Mantener dotación"

    recs.append({
        'Turno': turno,
        'Relación promedio': f"{med:.2f}",
        'IQR carga': f"{iqr:.2f}",
        'Acción': accion,
        'Día con mayor demanda': f"{peor['DíaSemana']} ({peor['Carga']:.2f})",
        'Día con menor demanda': f"{mejor['DíaSemana']} ({mejor['Carga']:.2f})"
    })

df_recs = pd.DataFrame(recs)

# 6) Mostrar recomendaciones como tabla
st.table(df_recs)

# 7) Explicación de cada columna de la tabla de recomendaciones
st.markdown("---")
st.markdown("**🛈 Explicación de parámetros**")
st.markdown("""
- **Turno**: Franja horaria.
- **Relación promedio**: Mediana de la carga histórica, definida como `T_AO / DOTACIÓN` (ofertas aceptadas por agente).
- **IQR carga**: Rango intercuartílico de la carga, que mide la variabilidad entre el cuartil 1 (25%) y el cuartil 3 (75%).
- **Acción**: Recomendación de dotación basada en la mediana de carga:
  - 🔴 Aumentar dotación: mediana > 1.5 (sub-dotación).
  - 🟢 Reducir dotación: mediana < 0.8 (sobre-dotación).
  - 🟡 Mantener dotación: carga equilibrada.
- **Día con mayor demanda**: Día de la semana cuya carga media fue máxima; sugiere cuándo reforzar.
- **Día con menor demanda**: Día de la semana cuya carga media fue mínima; sugiere posible reducción.
""")

# — Dropdown para filtrar por día de la semana —
st.subheader("📋 Resumen por turno")

# Opciones: promedio general + días de la semana en español
opciones = ['Promedio general', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
seleccion = st.selectbox("Selecciona el día de la semana:", opciones, index=0)

# Filtrado según la selección
if seleccion == 'Promedio general':
    df_filtro = df_turnos
else:
    dias_map = {
        'Lunes': 0, 'Martes': 1, 'Miércoles': 2,
        'Jueves': 3, 'Viernes': 4, 'Sábado': 5, 'Domingo': 6
    }
    df_filtro = df_turnos[df_turnos['FECHA'].dt.weekday == dias_map[seleccion]]

# Recalcular resumen por turno
res_turno_sel = (
    df_filtro
    .groupby('turno')[['T_VISITAS', 'T_AO', 'DOTACION', 'P_EFECTIVIDAD']]
    .mean()
    .reset_index()
)
res_turno_sel['Turno'] = res_turno_sel['turno'].map({
    0: 'Fuera rango', 1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21'
})

# Preparar DataFrame para mostrar y redondear a 2 decimales
df_display = (
    res_turno_sel[['Turno', 'T_VISITAS', 'T_AO', 'DOTACION', 'P_EFECTIVIDAD']]
    .rename(columns={
        'T_VISITAS': 'Visitas',
        'T_AO': 'Acepta Oferta',
        'DOTACION': 'Dotación',
        'P_EFECTIVIDAD': 'Efectividad'
    })
)
for col in ['Visitas', 'Acepta Oferta', 'Dotación', 'Efectividad']:
    df_display[col] = df_display[col].round(2)

# Mostrar tabla
st.dataframe(df_display, use_container_width=True)

# ——— Resumen avanzado bajo el heatmap ———
st.subheader(f"🔍 Rendimiento últimos {dias_analisis if dias_analisis else 'todos los'} días")

# 1) Reconstruimos 'rendimiento' igual que en el heatmap
df_turnos['Fecha'] = df_turnos['FECHA'].dt.date
rendimiento = (
    df_turnos
    .groupby(['Fecha', 'turno'])['P_EFECTIVIDAD']
    .mean()
    .reset_index()
)
rendimiento['Turno'] = rendimiento['turno'].map({
    1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21', 0: 'Fuera rango'
})
rendimiento['Efectividad (%)'] = rendimiento['P_EFECTIVIDAD'] * 100

# Filtramos según el rango seleccionado
if dias_analisis:
    fechas_ordenadas = sorted(rendimiento['Fecha'].unique())
    ultimas_n = fechas_ordenadas[-dias_analisis:]
    rendimiento = rendimiento[rendimiento['Fecha'].isin(ultimas_n)]

# 2) Estadísticas básicas por turno
stats = (
    rendimiento
    .groupby('Turno')['Efectividad (%)']
    .agg(['mean','std','min','max'])
    .round(2)
    .reset_index()
)

# 3) Fecha de máximo y mínimo por turno
idx_max = rendimiento.groupby('Turno')['Efectividad (%)'].idxmax()
idx_min = rendimiento.groupby('Turno')['Efectividad (%)'].idxmin()
maximos = rendimiento.loc[idx_max, ['Turno','Fecha','Efectividad (%)']].rename(
    columns={'Fecha':'Fecha_max','Efectividad (%)':'Maximo'}
)
minimos = rendimiento.loc[idx_min, ['Turno','Fecha','Efectividad (%)']].rename(
    columns={'Fecha':'Fecha_min','Efectividad (%)':'Minimo'}
)

# 4) Unimos stats + fechas de pico
resumen = stats.merge(maximos, on='Turno').merge(minimos, on='Turno')

# Aseguramos el orden de los turnos para el print
orden_turnos = ['9–11', '12–14', '15–17', '18–21']

for i, turno_label in enumerate(orden_turnos, start=1):
    fila = resumen[resumen['Turno'] == turno_label]
    if not fila.empty:
        r = fila.iloc[0]
        st.markdown(
            f"- **Turno {i} ({turno_label})**: "
            f"Efectividad promedio de **{r['mean']:.2f}%** (σ={r['std']:.2f}), "
            f"máx **{r['Maximo']:.2f}%** registrado el _{r['Fecha_max']}_ , "
            f"mín **{r['Minimo']:.2f}%** registrado el _{r['Fecha_min']}_."
        )

st.markdown("---")

# --- NUEVO: Ranking de efectividad por turno ---
st.subheader("🏆 Turnos con mejor y peor efectividad")

# 1) Efectividad y dotación promedio por turno
df_perf = (
    df_turnos
    .groupby('turno')
    .agg(
        Efectividad=('P_EFECTIVIDAD', 'mean'),
        Dotacion=('DOTACION', 'mean')
    )
    .reset_index()
)
turno_map = {1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21', 0: 'Fuera rango'}
df_perf['Turno'] = df_perf['turno'].map(turno_map)
df_perf['Efectividad_pct'] = (df_perf['Efectividad'] * 100).round(2)

# Tarjetas de efectividad promedio por turno
df_valid = df_perf[df_perf['turno'] > 0]
if not df_valid.empty:
    cols_avg = st.columns(len(df_valid))
    for i, row in enumerate(df_valid.sort_values('turno').itertuples(), start=0):
        with cols_avg[i]:
            st.metric(f"Turno {row.Turno}", f"{row.Efectividad_pct:.2f}%")

# 2) Identificar mejor y peor turno (ignorando el código 0)
df_valid = df_perf[df_perf['turno'] > 0]
if not df_valid.empty:
    best = df_valid.loc[df_valid['Efectividad'].idxmax()]
    worst = df_valid.loc[df_valid['Efectividad'].idxmin()]

    col_best, col_worst = st.columns(2)
    with col_best:
        st.metric("Mejor turno", f"{best['Turno']} ({best['Efectividad_pct']}%)")
    with col_worst:
        st.metric("Peor turno", f"{worst['Turno']} ({worst['Efectividad_pct']}%)")

# 3) Barra de efectividad promedio por turno
fig_rank = px.bar(
    df_valid.sort_values('Efectividad', ascending=False),
    x='Turno', y='Efectividad_pct',
    labels={'Efectividad_pct': 'Efectividad (%)'},
    title=f'Efectividad promedio por turno ({rango_seleccionado})'
)
fig_rank.update_layout(
    plot_bgcolor='#1a0033',
    paper_bgcolor='#1a0033',
    font_color='#FFFFFF',
    title_font_color='#FFFFFF'
)
st.plotly_chart(fig_rank, use_container_width=True)

# 4) Dispersión Dotación vs Efectividad por turno
fig_scatter = px.scatter(
    df_valid,
    x='Dotacion', y='Efectividad_pct', text='Turno',
    labels={'Dotacion': 'Dotación promedio', 'Efectividad_pct': 'Efectividad (%)'},
    title='Dotación promedio vs Efectividad por turno'
)
fig_scatter.update_traces(textposition='top center')
fig_scatter.update_layout(
    plot_bgcolor='#1a0033',
    paper_bgcolor='#1a0033',
    font_color='#FFFFFF',
    title_font_color='#FFFFFF'
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Distribución de efectividad por turno
df_box = df_turnos[df_turnos['turno'] > 0].copy()
df_box['Turno'] = df_box['turno'].map(turno_map)
fig_box_eff = px.box(
    df_box,
    x='Turno',
    y='P_EFECTIVIDAD',
    points='outliers',
    labels={'Turno':'Turno', 'P_EFECTIVIDAD':'Efectividad'},
    title=f'Distribuci\u00f3n de efectividad por turno ({rango_seleccionado})'
)
fig_box_eff.update_layout(
    plot_bgcolor='#1a0033',
    paper_bgcolor='#1a0033',
    font_color='#FFFFFF',
    title_font_color='#FFFFFF',
    yaxis_tickformat='.2f'
)
st.plotly_chart(fig_box_eff, use_container_width=True)

st.markdown("---")
st.subheader("🧮 Matriz de correlación de variables históricas")

cols_corr = ["T_VISITAS", "T_AO", "T_AO_VENTA", "DOTACION", "P_EFECTIVIDAD"]
df_corr = df_suc[cols_corr].dropna()
if not df_corr.empty:
    corr = df_corr.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Purples",
        title="Correlación entre variables"
    )
    fig.update_layout(
        plot_bgcolor="#1a0033",
        paper_bgcolor="#1a0033",
        font_color="#FFFFFF",
        title_font_color="#FFFFFF"
    )
    st.plotly_chart(fig, use_container_width=True)


