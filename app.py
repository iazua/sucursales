import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from train_models import generate_predictions
import plotly.express as px
from preprocessing import assign_turno
from utils import calcular_efectividad, estimar_dotacion_optima, estimar_parametros_efectividad
import pydeck as pdk
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paleta oficial Banco

PRIMARY_COLOR = "#4F2D7F"  # Minsk
DARK_BG_COLOR = "#361860"  # Scarlet Gum
# Color de fondo para secciones claras y tablas
PRIMARY_BG = "#F8F9FA"
# Usamos el mismo tono oscuro institucional para las tablas
TABLE_BG_COLOR = DARK_BG_COLOR
ACCENT_COLOR = "#F1AC4B"  # Sandy Brown
WHITE = "#FFFFFF"
BLACK = "#000000"
ACCENT_RGBA = "[241, 172, 75, 160]"  # Sandy Brown con opacidad
PRIMARY_RGBA = "[79, 45, 127, 255]"  # Minsk en formato RGBA para resaltar
# Mapeo de colores para series históricas y de predicción
COLOR_DISCRETE_MAP = {
    "Histórico": ACCENT_COLOR,
    "Escenario base": PRIMARY_BG,
    "Escenario alterno": "#FF4B4B",  # rojo para diferenciar el escenario alterno
}
# Colores para series donde se muestran dos categorías
# (e.g. Semana vs Fin de Semana) en gráficas de torta o barras
COLOR_SEQUENCE = [ACCENT_COLOR, PRIMARY_BG]
PIE_COLOR_MAP = {"Semana": ACCENT_COLOR, "Fin de Semana": PRIMARY_BG}
# ---------------------------------------------------------------------------

# --- CONSTANTES ---
# Rango horario estándar para la proyección (9–21)
HOURS_RANGE = list(range(9, 22))
# Fecha límite para las proyecciones automáticas
PREDICTION_END_DATE = pd.Timestamp("2025-12-31")

# Nombre de escenarios disponibles: base (sin ruido) y alterno con ruido
SCENARIO_NAMES = ["Escenario base", "Escenario alterno"]

# Mapeo de días de la semana en inglés a español
DAY_NAME_MAP_ES = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Predicción de Dotación Óptima (Hourly)", layout="wide")
st.markdown(
    f"""
    <style>
    :root {{
      --primary: {PRIMARY_COLOR};
      --dark-bg: {DARK_BG_COLOR};
      --table-bg: {TABLE_BG_COLOR};
      --accent: {ACCENT_COLOR};
      --white: {WHITE};
      --black: {BLACK};
    }}
    /* Aseguramos mismo color de los encabezados en todas las plataformas */
    h1, h2, h3, h4, h5, h6 {{
      color: var(--white);
    }}
    /* Fondo general */
    .stApp, .css-1d391kg {{
      background-color: var(--dark-bg);
    }}
    /* DataFrame: fondo de la tabla y de las celdas */
    .stDataFrame div[role="table"] {{
      background-color: var(--table-bg) !important;
      color: var(--white);
    }}
    /* Para los encabezados de tabla */
    .stDataFrame th {{
      background-color: var(--primary) !important;
      color: var(--white);
    }}
    /* Para las gráficas, el contenedor externo */
    .stPlotlyChart div {{
      background-color: var(--dark-bg) !important;
    }}
    /* Botones primarios */
    .stButton>button {{
      background-color: var(--primary);
      color: var(--white);
      border: none;
      box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }}
    .stButton>button:hover {{
      background-color: var(--accent);
      color: var(--black);
    }}
    .stButton>button:focus {{
      outline: 2px solid var(--accent);
      box-shadow: 0 0 0 3px rgba(241,172,75,0.4);
    }}
    /* Botones secundarios (agregar class "secondary" si se requieren) */
    .stButton.secondary>button {{
      background-color: rgba(255,255,255,0.95);
      color: var(--primary);
      border: 1px solid var(--primary);
    }}
    .stButton.secondary>button:hover {{
      background-color: var(--primary);
      color: var(--white);
    }}
    /* Botones CTA/acento (class "cta") */
    .stButton.cta>button {{
      background-color: var(--accent);
      color: var(--black);
    }}
    .stButton.cta>button:hover {{
      filter: brightness(1.1);
    }}
    /* Slider - estilo moderno */
    div[data-baseweb="slider"] .rc-slider-track {{
      background: linear-gradient(90deg, var(--accent), var(--primary)) !important;
      height: 8px;
    }}
    div[data-baseweb="slider"] .rc-slider-rail {{
      background-color: rgba(255, 255, 255, 0.4) !important;
      height: 8px;
    }}
    div[data-baseweb="slider"] .rc-slider-handle {{
      background-color: var(--accent) !important;
      border: none;
      height: 20px;
      width: 20px;
      margin-top: -6px;
      box-shadow: 0 0 0 2px var(--white);
    }}
    div[data-baseweb="slider"] .rc-slider-handle:active {{
      box-shadow: 0 0 0 4px rgba(241, 172, 75, 0.5);
    }}
    .slider-label {{
      color: var(--white);
      font-weight: 600;
      margin-bottom: 0.25rem;
    }}
    /* Tablas incrustadas en el fondo */
    .stDataFrame, .stTable {{
      background-color: var(--table-bg) !important;
      border: none;
    }}
    /* Ajustes extra para DataFrame en la pestaña Forecast */
    div[data-testid="stDataFrame"] > div {{
      background-color: var(--table-bg) !important;
    }}
    div[data-testid="stDataFrame"] table {{
      background-color: var(--table-bg) !important;
      color: var(--white) !important;
    }}
    div[data-testid="stDataFrame"] th {{
      background-color: var(--primary) !important;
      color: var(--white) !important;
    }}
    .stTable table {{
      background-color: var(--table-bg) !important;
      color: var(--white);
    }}
    .stTable th {{
      background-color: var(--primary) !important;
      color: var(--white);
    }}
    /* Imágenes responsivas */
    img {{
      max-width: 100%;
      height: auto;
    }}
    /* Dropdown integrado */
    .stSelectbox div[data-baseweb="select"] > div {{
      background-color: var(--dark-bg);
      color: var(--white);
      border-color: var(--primary);
    }}
    .stSelectbox div[data-baseweb="select"] > div:hover {{
      border-color: var(--accent);
    }}
    .stSelectbox div[data-baseweb="select"] > div:focus-within {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(241,172,75,0.4);
    }}
    </style>
    """,
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/27/Logo_Ripley_banco_2.png",
        use_container_width=True
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

# Horizonte dinámico hasta fin de 2025
LAST_DATA_DATE = df["FECHA"].max()
HORIZON_DAYS = max((PREDICTION_END_DATE - (LAST_DATA_DATE + timedelta(days=1))).days + 1, 0)

MESES_FUTURO = 12  # horizonte de predicción (puedes cambiarlo)

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
        'T_VISITAS': 'sum',
        'T_AO': 'sum',
        'T_AO_VENTA': 'sum'
    })
)
df_zona['EFECTIVIDAD'] = df_zona['T_AO_VENTA'] / df_zona['T_AO']

# 3) Calculamos porcentajes y efectividad
total_vis = df_zona['T_VISITAS'].sum()
total_ao = df_zona['T_AO'].sum()
total_ao_venta = df_zona['T_AO_VENTA'].sum()
df_zona['pct_vis'] = df_zona['T_VISITAS'] / total_vis
df_zona['pct_ao'] = df_zona['T_AO'] / total_ao
df_zona['pct_ao_venta'] = df_zona['T_AO_VENTA'] / total_ao_venta
df_zona['pct_efectividad'] = df_zona['EFECTIVIDAD']  # Ya está en formato 0-1
df_zona['label_vis'] = df_zona['pct_vis'].apply(lambda x: f"{x:.1%}")
df_zona['label_ao'] = df_zona['pct_ao'].apply(lambda x: f"{x:.1%}")
df_zona['label_ao_venta'] = df_zona['pct_ao_venta'].apply(lambda x: f"{x:.1%}")
df_zona['label_efectividad'] = df_zona['pct_efectividad'].apply(lambda x: f"{x:.1%}")

# 4) Centroides aproximados por zona
centroides = {
    'Norte': (-20.0, -70.0),
    'Centro': (-32.5, -71.5),
    'Sur': (-38.0, -73.0),
    'Santiago': (-33.45, -70.65),
    'Desconocida': (-33.0, -70.0)
}
df_zona['lat'] = df_zona['ZONA'].map(lambda z: centroides.get(z, centroides['Desconocida'])[0])
df_zona['lon'] = df_zona['ZONA'].map(lambda z: centroides.get(z, centroides['Desconocida'])[1])

# 5) Definimos cuatro capas: visitas, ofertas aceptadas, ofertas aceptadas de venta y efectividad
layer_vis = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color=ACCENT_RGBA,
    get_radius='pct_vis * 400000',
    auto_highlight=True,
    highlight_color=PRIMARY_RGBA,
)

layer_ao = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color=ACCENT_RGBA,
    get_radius='pct_ao * 400000',
    auto_highlight=True,
    highlight_color=PRIMARY_RGBA,
)

layer_ao_venta = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color=ACCENT_RGBA,
    get_radius='pct_ao_venta * 400000',
    auto_highlight=True,
    highlight_color=PRIMARY_RGBA,
)

layer_efectividad = pdk.Layer(
    "ScatterplotLayer",
    data=df_zona,
    pickable=True,
    get_position='[lon, lat]',
    get_fill_color=ACCENT_RGBA,
    get_radius='pct_efectividad * 10000',
    auto_highlight=True,
    highlight_color=PRIMARY_RGBA,
)

# 6) Vista centrada en Chile con rotación de 90° hacia la derecha
view_state = pdk.ViewState(
    latitude=-30.5,
    longitude=-70.9,
    zoom=5,
    pitch=0,
    bearing=90
)

# Tabs para organizar la aplicación de forma más clara
tab_mapa, tab_pred, tab_hist, tab_turno = st.tabs(
    [
        "Overview",
        "Forecast",
        "Histórico",
        "Turnos",
    ]
)

# 7) Renderizamos las cuatro capas en un solo mapa con la nueva vista
with tab_mapa:
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
            "style": {"backgroundColor": ACCENT_COLOR, "color": DARK_BG_COLOR}
        }
    ), use_container_width=True)

    # --- Resumen Global de Sucursales ---
    total_vis = int(df['T_VISITAS'].sum())
    total_ao = int(df['T_AO'].sum())
    total_sales = int(df['T_AO_VENTA'].sum())
    global_eff = total_sales / total_ao if total_ao else 0

    cols_global = st.columns(4)
    cols_global[0].metric("Total Visitas", f"{total_vis:,}".replace(",", "."))
    cols_global[1].metric("Total Acepta Oferta", f"{total_ao:,}".replace(",", "."))
    cols_global[2].metric("Ventas", f"{total_sales:,}".replace(",", "."))
    cols_global[3].metric("Efectividad Global", f"{global_eff:.1%}")

    # Tarjetas con la mejor y peor efectividad por sucursal
    eff_branch = (
        df_map.groupby('COD_SUC')
        .agg({'T_AO': 'sum', 'T_AO_VENTA': 'sum'})
    )
    eff_branch['ef'] = eff_branch['T_AO_VENTA'] / eff_branch['T_AO']
    eff_branch = eff_branch.replace([np.inf, -np.inf], np.nan).dropna(subset=['ef'])
    worst_id = eff_branch['ef'].idxmin()
    best_id = eff_branch['ef'].idxmax()
    worst_val = eff_branch.loc[worst_id, 'ef']
    best_val = eff_branch.loc[best_id, 'ef']
    cols_eff = st.columns(2)
    cols_eff[0].metric("Peor efectividad", f"Sucursal {worst_id}", f"{worst_val:.1%}")
    cols_eff[1].metric("Mejor efectividad", f"Sucursal {best_id}", f"{best_val:.1%}")

    st.markdown("---")

    # --- Agrupación por zona ---
    zona_totales = (
        df_map
        .groupby('ZONA', as_index=False)
        .agg({'T_VISITAS': 'sum', 'T_AO_VENTA': 'sum'})
        .sort_values('T_VISITAS', ascending=False)
    )
    fig_zona = px.bar(
        zona_totales,
        x='ZONA',
        y=['T_VISITAS', 'T_AO_VENTA'],
        barmode='group',
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={'value': 'Total', 'variable': 'Métrica', 'ZONA': 'Zona'}
    )
    fig_zona.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE,
        legend_title_font_color=WHITE,
        title='Totales por Zona'
    )
    st.plotly_chart(fig_zona, use_container_width=True)

    st.markdown("### Overview")
    ranking = (
        df_map
        .groupby(['COD_SUC', 'ZONA'], as_index=False)
        .agg({'T_VISITAS': 'sum', 'T_AO': 'sum', 'T_AO_VENTA': 'sum'})
    )
    ranking['Efectividad'] = ranking['T_AO_VENTA'] / ranking['T_AO']
    ranking['COD_SUC'] = ranking['COD_SUC'].astype(str)
    ranking = ranking.sort_values('COD_SUC')
    ranking_display = ranking.rename(columns={
        'COD_SUC': 'Sucursal',
        'ZONA': 'Zona',
        'T_VISITAS': 'Visitas',
        'T_AO': 'Acepta Oferta',
        'T_AO_VENTA': 'Ventas',
        'Efectividad': 'Efectividad'
    })
    for c in ['Visitas', 'Acepta Oferta', 'Ventas']:
        ranking_display[c] = ranking_display[c].astype(int).apply(lambda x: f"{x:,}".replace(',', '.'))
    ranking_display['Efectividad'] = ranking_display['Efectividad'].apply(lambda x: f"{x:.1%}")
    st.dataframe(ranking_display, use_container_width=True, hide_index=True)

    # --- Evolución mensual de efectividad por sucursal ---
    st.markdown("### Evolución mensual de efectividad")

    # Dropdown de sucursal y slider de rango de fechas
    sucursales_hist = sorted(df["COD_SUC"].unique())
    sucursal_hist = st.selectbox(
        "Sucursal",
        sucursales_hist,
        key="eff_branch_sel",
    )

    min_fecha = df["FECHA"].min().date()
    max_fecha = df["FECHA"].max().date()
    fecha_inicio, fecha_fin = st.slider(
        "Rango de fechas",
        min_value=min_fecha,
        max_value=max_fecha,
        value=(min_fecha, max_fecha),
        format="%d/%m/%Y",
    )

    # Filtrar histórico y calcular efectividad mensual
    df_branch = df[(df["COD_SUC"] == sucursal_hist)].copy()
    df_branch = df_branch[(df_branch["FECHA"].dt.date >= fecha_inicio) & (df_branch["FECHA"].dt.date <= fecha_fin)]
    df_branch["Mes"] = df_branch["FECHA"].dt.to_period("M").dt.to_timestamp()
    mensual = (
        df_branch
        .groupby("Mes", as_index=False)
        .agg({"T_AO": "sum", "T_AO_VENTA": "sum"})
    )
    mensual["Efectividad"] = mensual.apply(
        lambda r: r["T_AO_VENTA"] / r["T_AO"] if r["T_AO"] > 0 else 0,
        axis=1,
    )

    fig_eff = px.line(
        mensual,
        x="Mes",
        y="Efectividad",
        markers=True,
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_eff.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE,
        yaxis_tickformat=".1%",
        xaxis_title="Fecha",
        yaxis_title="Efectividad",
    )
    fig_eff.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_eff, use_container_width=True)

with tab_pred:
    st.title("🔍 Predicción de Dotación y Efectividad por Hora")

    method = "Prophet"

    # --- CONTROL DE HORIZONTE DE PROYECCIÓN ---
    # Se fija de forma estática a 306 días
    days_proj = 306
    sucursales = sorted(df["COD_SUC"].unique())
    cod_suc = st.selectbox("Selecciona una sucursal", sucursales)

    # --- FILTRAR Y PREPARAR HISTÓRICO ---
    df_suc = df[df["COD_SUC"] == cod_suc].copy()
    df_suc = df_suc.sort_values(["FECHA", "HORA"] if "HORA" in df_suc.columns else ["FECHA"]).reset_index(drop=True)

    # --- Dotación promedio histórica por día de la semana y hora ---
    if "HORA" in df_suc.columns:
        df_suc["weekday"] = df_suc["FECHA"].dt.dayofweek
        avg_dot_map = (
            df_suc.groupby(["weekday", "HORA"])["DOTACION"].mean().to_dict()
        )
    else:
        avg_dot_map = {}

    # Asegurar P_EFECTIVIDAD histórica
    if "P_EFECTIVIDAD" not in df_suc.columns:
        df_suc["P_EFECTIVIDAD"] = calcular_efectividad(df_suc["T_AO"], df_suc["T_AO_VENTA"])

    # Promedio de efectividad con DOTACION=1 (para fallback)
    df_dot1 = df_suc[df_suc["DOTACION"] == 1]
    avg_eff_dot1 = df_dot1["P_EFECTIVIDAD"].mean() if not df_dot1.empty else np.nan

    # --- SLIDER DE EFECTIVIDAD OBJETIVO ---
    st.markdown(
        "<div class='slider-label'><b>Efectividad objetivo:</b></div>",
        unsafe_allow_html=True,
    )
    efectividad_pct = st.slider(
        label="",
        min_value=0,
        max_value=100,
        value=62,
        step=1,
        format="%d%%",
        label_visibility="collapsed",
    )
    efectividad_obj = efectividad_pct / 100

    st.markdown(
        "<div class='slider-label'><b>Ruido escenario alterno:</b></div>",
        unsafe_allow_html=True,
    )
    noise_slider = st.slider(
        label="",
        min_value=0.0,
        max_value=2.0,
        value=2.0,
        step=0.1,
        format="%.1f",
        label_visibility="collapsed",
    )


    # --- app.py ---

    @st.cache_data(show_spinner="Generando pronóstico…")
    def forecast_fast(df_all: pd.DataFrame,
                      cod_suc: int,
                      efect_obj: float,
                      days: int,
                      noise_scale: float = 0.0) -> pd.DataFrame:
        """Versión rápida usando `generate_predictions` vectorizado."""
        last_date = df_all[df_all["COD_SUC"] == cod_suc]["FECHA"].max()
        start_dt = last_date + timedelta(days=1)
        end_dt = start_dt + timedelta(days=days - 1)
        return generate_predictions(
            df_all,
            branch=cod_suc,
            efectividad_obj=efect_obj,
            start_date=start_dt,
            end_date=end_dt,
            noise_scale=noise_scale,
        )


    # ---------- LLAMADA ----------
    # Generar predicciones para cada escenario definido
    pred_dict = {
        "Escenario base": forecast_fast(
            df,
            cod_suc,
            efectividad_obj,
            days_proj,
            noise_scale=0.0,
        ),
        "Escenario alterno": forecast_fast(
            df,
            cod_suc,
            efectividad_obj,
            days_proj,
            noise_scale=noise_slider,
        ),
    }

    scenario_selected = st.selectbox(
        "Escenario para tablas", SCENARIO_NAMES, index=0
    )

    df_pred = pred_dict[scenario_selected]

    # ——— TABLA POR HORA ———
    st.subheader("Por hora")

    # 1) Seleccionamos únicamente las columnas de df_pred que necesitamos
    df_hourly = df_pred[[
        "FECHA",
        "HORA",
        "T_VISITAS_pred",
        "T_AO_pred",
        "T_AO_VENTA_req",
        "P_EFECTIVIDAD_req",
        "DOTACION_req",
    ]].copy()

    # 2) Formateamos FECHA y añadimos día de la semana
    df_hourly["Fecha registro"] = df_hourly["FECHA"].dt.strftime("%d-%m-%Y")
    df_hourly["Día"] = df_hourly["FECHA"].dt.day_name().map(DAY_NAME_MAP_ES)
    df_hourly["weekday"] = df_hourly["FECHA"].dt.dayofweek

    # 3) Renombramos cada métrica de forma explícita
    df_hourly = df_hourly.rename(columns={
        "HORA": "Hora",
        "T_VISITAS_pred": "Visitas estimadas",
        "T_AO_pred": "Ofertas aceptadas estimadas",
        "T_AO_VENTA_req": "Ventas requeridas",
        "P_EFECTIVIDAD_req": "% Efectividad requerida",
        "DOTACION_req": "Dotación requerida",
    })

    # 4) Redondeamos y transformamos tipos
    df_hourly["Visitas estimadas"] = df_hourly["Visitas estimadas"].round(0).astype(int)
    df_hourly["Ofertas aceptadas estimadas"] = df_hourly["Ofertas aceptadas estimadas"].round(0).astype(int)
    df_hourly["Ventas requeridas"] = df_hourly["Ventas requeridas"].round(0).astype(int)
    df_hourly["% Efectividad requerida"] = df_hourly["% Efectividad requerida"].round(2)
    df_hourly["Dotación requerida"] = df_hourly["Dotación requerida"].round(0).astype(int)

    # 4.b) Dotación histórica y ajuste necesario
    def _avg_hist(row):
        key = (row["weekday"], row["Hora"])
        return round(avg_dot_map.get(key, np.nan), 1)

    df_hourly["Dotación histórica"] = df_hourly.apply(_avg_hist, axis=1)
    df_hourly["Ajuste dotación"] = (df_hourly["Dotación requerida"] - df_hourly["Dotación histórica"]).round(1)

    # Preparamos una copia solo para mostrar, aplicando formato de miles
    df_hourly_display = df_hourly.copy()
    for col in [
        "Visitas estimadas",
        "Ofertas aceptadas estimadas",
        "Ventas requeridas",
        "Dotación requerida",
        "Dotación histórica",
        "Ajuste dotación",
    ]:
        df_hourly_display[col] = df_hourly_display[col].apply(lambda x: f"{x:,}".replace(',', '.'))

    # 5) Seleccionamos el orden final de columnas
    df_hourly = df_hourly[[
        "Fecha registro", "Día", "Hora",
        "Visitas estimadas", "Ofertas aceptadas estimadas",
        "Ventas requeridas", "% Efectividad requerida",
        "Dotación requerida", "Dotación histórica", "Ajuste dotación"
    ]]
    df_hourly_display = df_hourly_display[df_hourly.columns]

    st.dataframe(df_hourly_display, use_container_width=True, hide_index=True)

    # ——— TABLA POR DÍA ———
    st.subheader("Por día")

    df_daily = (
        df_hourly
        .groupby(["Fecha registro", "Día"], as_index=False)
        .agg({
            "Visitas estimadas": "sum",
            "Ofertas aceptadas estimadas": "sum",
            "Ventas requeridas": "sum",
            "% Efectividad requerida": "mean",
            "Dotación requerida": "mean",
            "Dotación histórica": "mean",
            "Ajuste dotación": "mean",
        })
    )

    # Redondeo final de efectividad
    df_daily["% Efectividad requerida"] = df_daily["% Efectividad requerida"].round(2)
    df_daily["Dotación requerida"] = df_daily["Dotación requerida"].round(1)
    df_daily["Dotación histórica"] = df_daily["Dotación histórica"].round(1)
    df_daily["Ajuste dotación"] = df_daily["Ajuste dotación"].round(1)

    # Orden cronológico
    df_daily["_dt"] = pd.to_datetime(df_daily["Fecha registro"], format="%d-%m-%Y")
    df_daily = df_daily.sort_values("_dt").drop(columns="_dt")

    # Preparamos una copia formateada solo para mostrar
    df_daily_display = df_daily.copy()
    for col in [
        "Visitas estimadas",
        "Ofertas aceptadas estimadas",
        "Ventas requeridas",
        "Dotación requerida",
        "Dotación histórica",
        "Ajuste dotación",
    ]:
        df_daily_display[col] = df_daily_display[col].apply(lambda x: f"{round(x,1):,}".replace(',', '.'))

    df_daily = df_daily[[
        "Fecha registro",
        "Día",
        "Visitas estimadas",
        "Ofertas aceptadas estimadas",
        "Ventas requeridas",
        "% Efectividad requerida",
        "Dotación requerida",
        "Dotación histórica",
        "Ajuste dotación",
    ]]
    df_daily_display = df_daily_display[df_daily.columns]

    st.dataframe(df_daily_display, use_container_width=True, hide_index=True)

    # --- CURVA DE EFECTIVIDAD vs. DOTACIÓN (Teórica) ---
    st.subheader("Curva de Efectividad vs. Dotación")

    # 1. Estimar parámetros históricos
    hist = df_suc[['DOTACION', 'T_AO', 'T_AO_VENTA']].dropna()
    if len(hist) >= 3:
        params_eff = estimar_parametros_efectividad(hist)
    else:
        params_eff = {'L': 1.0, 'k': 0.5, 'x0_base': 5.0, 'x0_factor_t_ao_venta': 0.05}

    L = params_eff['L']
    k_def = params_eff['k']
    x0_base = params_eff['x0_base']
    x0_fac = params_eff['x0_factor_t_ao_venta']

    # 2. Coeficiente k estimado para la sucursal
    k = float(k_def)
    st.write(f"Coeficiente k estimado: {k:.2f}")

    # 3. Rango de dotación fijo entre 1 y 12 (enteros)
    dot_range = np.arange(1, 13)

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
        "Dotación": np.tile(dot_range, 2),
        "Modelo": ["Sigmoide"] * len(dot_range) + ["Gompertz"] * len(dot_range),
        "Efectividad": np.concatenate([ef_sig, ef_gom])
    })

    # --- Variación de efectividad al sumar una persona ---
    delta_sig = np.diff(ef_sig)
    df_delta = pd.DataFrame({
        "Dotación": dot_range[1:],
        "Δ Sigmoide": delta_sig,
    })

    # 6.b Calcular dotación óptima usando las predicciones actuales
    dot_opt, _ = estimar_dotacion_optima(
        df_pred["T_AO_pred"],
        df_pred["T_AO_VENTA_req"],
        efectividad_obj,
        params_eff,
    )
    dot_opt = int(np.clip(np.round(dot_opt), dot_range.min(), dot_range.max()))

    # 7. Gráfico combinado de curva e incremento de efectividad
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Curvas teóricas
    for modelo, color in zip(["Sigmoide", "Gompertz"], [ACCENT_COLOR, PRIMARY_BG]):
        df_m = df_curve[df_curve["Modelo"] == modelo]
        fig.add_trace(
            go.Scatter(
                x=df_m["Dotación"],
                y=df_m["Efectividad"],
                mode="lines+markers",
                name=modelo,
                line=dict(color=color),
            ),
            secondary_y=False,
        )

    # Incremento de efectividad al sumar 1 persona (sigmoide)
    fig.add_trace(
        go.Scatter(
            x=df_delta["Dotación"],
            y=df_delta["Δ Sigmoide"],
            mode="lines+markers+text",
            name="Δ Efectividad (Sigmoide)",
            text=[f"{d:.1%}" for d in df_delta["Δ Sigmoide"]],
            textposition="top center",
            line=dict(color=ACCENT_COLOR, dash="dot"),
        ),
        secondary_y=True,
    )

    # Línea vertical en dotación óptima
    fig.add_vline(
        x=dot_opt,
        line_dash="dash",
        line_color=ACCENT_COLOR,
        annotation_text="Dot. Óptima",
        annotation_position="top",
    )

    fig.update_layout(
        paper_bgcolor=DARK_BG_COLOR,
        plot_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title=" ",
        legend_title_text="Modelo",
        barmode="group",
    )
    fig.update_yaxes(title_text="Efectividad", secondary_y=False)
    fig.update_yaxes(title_text="Δ Efectividad", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    df_display_dict = {}
    for name, df_p in pred_dict.items():
        d = df_p.copy()
        d["DÍA"] = d["FECHA"].dt.strftime("%d-%m-%Y")
        d = d.rename(columns={
            "T_AO_pred": "Ofertas aceptadas estimadas",
            "HORA": "Hora",
            "T_VISITAS_pred": "Visitas estimadas",
            "T_AO_VENTA_req": "Ventas requeridas",
            "P_EFECTIVIDAD_req": "% Efectividad requerida",
        })
        df_display_dict[name] = d

    df_display = df_display_dict[scenario_selected]

    # --- GRÁFICO 1: Ofertas Aceptadas diario ---
    st.subheader("📈 Histórico y Predicción de Ofertas Aceptadas")


    # Agrupar histórico por fecha
    hist_ao = (
        df_suc
        .groupby('FECHA', observed=True)['T_AO']
        .sum()
        .reset_index()
        .rename(columns={'T_AO': 'Valor'})
        .assign(Tipo='Histórico')
    )

    # Agrupar predicción por fecha para cada escenario
    pred_ao_list = []
    for name, df_disp in df_display_dict.items():
        tmp = (
            df_disp
            .groupby('DÍA', observed=True)['Ofertas aceptadas estimadas']
            .sum()
            .reset_index()
            .rename(columns={'DÍA': 'FECHA', 'Ofertas aceptadas estimadas': 'Valor'})
        )
        tmp['FECHA'] = pd.to_datetime(tmp['FECHA'], format='%d-%m-%Y')
        tmp = tmp.sort_values('FECHA').head(days_proj).assign(Tipo=name)
        pred_ao_list.append(tmp)

    # Combinar y pivotar
    df_plot_ao = pd.concat([hist_ao] + pred_ao_list, ignore_index=True)
    df_pivot_ao = df_plot_ao.pivot_table(
        index='FECHA', columns='Tipo', values='Valor', aggfunc='sum'
    )
    df_plot_ao_long = (
        df_pivot_ao
        .reset_index()
        .melt(id_vars='FECHA', var_name='Tipo', value_name='Valor')
    )
    fig = px.line(
        df_plot_ao_long,
        x='FECHA',
        y='Valor',
        color='Tipo',
        color_discrete_map=COLOR_DISCRETE_MAP,
    )
    fig.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)



    # --- GRÁFICO 2: Ventas Concretadas diario ---
    st.subheader("📈 Histórico y Predicción de Ventas Concretadas")

    # Agrupar histórico de ventas
    hist_v = (
        df_suc
        .groupby('FECHA', observed=True)['T_AO_VENTA']
        .sum()
        .reset_index()
        .rename(columns={'T_AO_VENTA': 'Valor'})
        .assign(Tipo='Histórico')
    )

    # Agrupar predicción de ventas requeridas por escenario
    pred_v_list = []
    for name, df_disp in df_display_dict.items():
        tmp = (
            df_disp
            .groupby('DÍA', observed=True)['Ventas requeridas']
            .sum()
            .reset_index()
            .rename(columns={'DÍA': 'FECHA', 'Ventas requeridas': 'Valor'})
        )
        tmp['FECHA'] = pd.to_datetime(tmp['FECHA'], format='%d-%m-%Y')
        tmp = tmp.sort_values('FECHA').head(days_proj).assign(Tipo=name)
        pred_v_list.append(tmp)

    # Combinar y pivotar
    df_plot_v = pd.concat([hist_v] + pred_v_list, ignore_index=True)
    df_pivot_v = df_plot_v.pivot_table(
        index='FECHA', columns='Tipo', values='Valor', aggfunc='sum'
    )
    df_plot_v_long = (
        df_pivot_v
        .reset_index()
        .melt(id_vars='FECHA', var_name='Tipo', value_name='Valor')
    )
    fig = px.line(
        df_plot_v_long,
        x='FECHA',
        y='Valor',
        color='Tipo',
        color_discrete_map=COLOR_DISCRETE_MAP,
    )
    fig.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- GRÁFICO 3: Visitas diario ---
    st.subheader("📈 Histórico y Predicción de Visitas")

    # Agrupar histórico de visitas
    hist_vis = (
        df_suc
        .groupby('FECHA', observed=True)['T_VISITAS']
        .sum()
        .reset_index()
        .rename(columns={'T_VISITAS': 'Valor'})
        .assign(Tipo='Histórico')
    )

    # Agrupar predicción de visitas estimadas por escenario
    pred_vis_list = []
    for name, df_disp in df_display_dict.items():
        tmp = (
            df_disp
            .groupby('DÍA', observed=True)['Visitas estimadas']
            .sum()
            .reset_index()
            .rename(columns={'DÍA': 'FECHA', 'Visitas estimadas': 'Valor'})
        )
        tmp['FECHA'] = pd.to_datetime(tmp['FECHA'], format='%d-%m-%Y')
        tmp = tmp.sort_values('FECHA').head(days_proj).assign(Tipo=name)
        pred_vis_list.append(tmp)

    # Combinar y pivotar
    df_plot_vis = pd.concat([hist_vis] + pred_vis_list, ignore_index=True)
    df_pivot_vis = df_plot_vis.pivot_table(
        index='FECHA', columns='Tipo', values='Valor', aggfunc='sum'
    )
    df_plot_vis_long = (
        df_pivot_vis
        .reset_index()
        .melt(id_vars='FECHA', var_name='Tipo', value_name='Valor')
    )
    fig = px.line(
        df_plot_vis_long,
        x='FECHA',
        y='Valor',
        color='Tipo',
        color_discrete_map=COLOR_DISCRETE_MAP,
    )
    fig.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# --- ANÁLISIS HISTÓRICO PONDERADO POR DÍA DE LA SEMANA ---

with tab_hist:
    st.header("🔍 Flujo histórico ponderado por día de la semana")

    # Mapear nombres de día
    dias_map = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }

    df = df_suc.copy()
    df['DíaSemana'] = df['FECHA'].dt.day_name().map(dias_map)
    df['TipoDia'] = np.where(df['FECHA'].dt.weekday < 5, 'Semana', 'Fin de Semana')

    # Factor de ponderación: 2 días de fin de semana para cada día de semana, 5 días de semana para cada día de fin de semana
    df['Factor'] = np.where(df['TipoDia'] == 'Semana', 2, 5)

    # Agregar sumas y aplicar factor
    grouped = (
        df
        .groupby('DíaSemana', observed=True)
        .agg(
            T_VISITAS_raw=('T_VISITAS', 'sum'),
            T_AO_raw=('T_AO', 'sum'),
            Factor=('Factor', 'first')  # mismo factor por grupo
        )
        .reset_index()
    )
    grouped = (
        df
        .groupby('DíaSemana', observed=True)
        .agg(
            T_VISITAS=('T_VISITAS', 'sum'),
            T_AO=('T_AO', 'sum')
        )
        .reset_index()
    )

    fig = px.bar(
        grouped,
        x='DíaSemana',
        y=['T_VISITAS', 'T_AO'],
        barmode='group',
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={
            'value': 'Total registrado',
            'variable': 'Métrica',
            'DíaSemana': 'Día de la semana'
        },
        title='Total de Visitas y Ofertas Aceptadas por Día de la Semana'
    )

    # Cambiar las etiquetas de la leyenda
    fig.for_each_trace(lambda t: t.update(name='Visitas' if t.name == 'T_VISITAS' else 'Acepta Oferta'))
    fig.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE,
        legend_title_font_color=WHITE
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- DISTRIBUCIÓN SEMANA vs. FIN DE SEMANA PONDERADA ---
    st.header("📊 Semana vs Fin de Semana (ponderado)")

    dist = (
        df
        .groupby('TipoDia', observed=True)
        .agg(
            T_VISITAS_raw=('T_VISITAS', 'sum'),
            T_AO_raw=('T_AO', 'sum'),
            Factor=('Factor', 'first')
        )
        .reset_index()
    )
    dist['T_VISITAS_pond'] = dist['T_VISITAS_raw'] * dist['Factor']
    dist['T_AO_pond'] = dist['T_AO_raw'] * dist['Factor']

    # Alinear ambos pie charts lado a lado con el mismo tamaño
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.pie(
                dist,
                names='TipoDia',
                values='T_VISITAS_pond',
                title='Proporción de Visitas ponderadas: Semana vs Fin de Semana',
                hole=0.4,
                color_discrete_sequence=COLOR_SEQUENCE,
                category_orders={'TipoDia': ['Semana', 'Fin de Semana']}
            ).update_layout(
                plot_bgcolor=DARK_BG_COLOR,
                paper_bgcolor=DARK_BG_COLOR,
                font_color=WHITE,
                title_font_color=WHITE
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.pie(
                dist,
                names='TipoDia',
                values='T_AO_pond',
                title='Proporción de Ofertas Aceptadas ponderadas: Semana vs Fin de Semana',
                hole=0.4,
                color_discrete_sequence=COLOR_SEQUENCE,
                category_orders={'TipoDia': ['Semana', 'Fin de Semana']}
            ).update_layout(
                plot_bgcolor=DARK_BG_COLOR,
                paper_bgcolor=DARK_BG_COLOR,
                font_color=WHITE,
                title_font_color=WHITE
            ),
            use_container_width=True
        )

# --- Agregar selector de rango de días al inicio ---
with tab_turno:
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
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={
            'T_VISITAS': 'Visitas',
            'T_AO': 'Acepta Oferta',
            'value': 'Promedio',
            'variable': 'Métrica'
        },
        title=f"Visitas y Acepta Oferta promedio por franja horaria ({rango_seleccionado})"
    )
    fig.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE,
        legend_title_font_color=WHITE
    )
    st.plotly_chart(fig, use_container_width=True)

    # ——— KPI de conversión visitas → ofertas aceptadas por turno ———
    st.markdown("---")
    st.subheader("📈 Conversión de visitas a ofertas aceptadas por turno")

    # 1) Calculamos la conversión: sum(T_AO) / sum(T_VISITAS) por turno
    conv = (
        df_turnos
        .groupby('turno')
        .agg({'T_VISITAS': 'sum', 'T_AO': 'sum'})
        .reset_index()
    )
    conv['conversion'] = conv.apply(
        lambda row: row['T_AO'] / row['T_VISITAS'] if row['T_VISITAS'] > 0 else 0,
        axis=1
    )

    # 2) Mapeo de etiquetas de turno
    etiquetas = {1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21'}
    conv['rango_horas'] = conv['turno'].map(etiquetas)

    # 3) Mostramos cada turno como tarjeta KPI
    cols = st.columns(4)
    for i, turno in enumerate([1, 2, 3, 4], start=1):
        pct = conv.loc[conv['turno'] == turno, 'conversion'].iloc[0]
        with cols[i - 1]:
            st.metric(
                label=f"Turno {etiquetas[turno]}",
                value=f"{pct:.2%}"
            )

    # ——— Heatmap de conversión visitas → ofertas aceptadas por día y turno ———
    st.markdown("---")
    st.subheader("🌡️ Heatmap de conversión por día de la semana y turno")

    # 1) Calcular conversión por día de la semana y turno
    df_turnos['DiaSemana'] = df_turnos['FECHA'].dt.day_name().map(DAY_NAME_MAP_ES)
    conv_dt = (
        df_turnos
        .groupby(['DiaSemana', 'turno'], observed=True)
        .agg(T_VISITAS=('T_VISITAS', 'sum'), T_AO=('T_AO', 'sum'))
        .reset_index()
    )
    conv_dt['Conversion'] = conv_dt.apply(
        lambda r: r['T_AO'] / r['T_VISITAS'] if r['T_VISITAS'] > 0 else 0,
        axis=1
    )

    # 2) Mapeo al español y orden de días y turnos
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    turnos = {1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21'}
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
        color_continuous_scale=[PRIMARY_BG, ACCENT_COLOR],
        labels={'x': 'Turno', 'y': 'Día de la semana', 'color': 'Conversión (%)'},
        title=f'Conversión visitas → ofertas aceptadas (%) ({rango_seleccionado})',
        aspect='auto'
    )
    fig.update_xaxes(side='top')
    fig.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE
    )

    st.plotly_chart(fig, use_container_width=True)

    # ——— Serie de efectividad diaria por turno ———
    # 1) Calculamos df_ts como antes
    df_ts = (
        df_turnos
        .groupby([df_turnos['FECHA'].dt.date.rename('Fecha'), 'turno'])
        .agg({'P_EFECTIVIDAD': 'mean'})
        .reset_index()
    )
    df_ts['Turno'] = df_ts['turno'].map({1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21'})

    # 2) Creamos 4 filas, 1 columna, ejes X compartidos
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=['Turno 9–11', 'Turno 12–14', 'Turno 15–17', 'Turno 18–21']
    )

    # 3) Añadimos la línea por cada turno
    for i, turno in enumerate(['9–11', '12–14', '15–17', '18–21'], start=1):
        df_sub = df_ts[df_ts['Turno'] == turno]
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
    fig.update_xaxes(row=4, col=1, tickangle=-45, title_text='Fecha',
                     rangeslider_visible=True)

    # 5) Diseño general
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f'Efectividad diaria por turno ({rango_seleccionado})',
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        margin=dict(t=80, b=40, l=60, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ——— Efectividad promedio por día de la semana y turno ———
    st.markdown("---")
    st.subheader("📊 Efectividad promedio por día de la semana y por turno")

    # 1) Preparamos día de la semana y mapeo de turnos
    dias_map = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df_turnos['DíaSemana'] = df_turnos['FECHA'].dt.day_name().map(dias_map)
    turno_map = {1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21', 0: 'Fuera rango'}
    df_turnos['Turno'] = df_turnos['turno'].map(turno_map)

    # 2) Calculamos el promedio
    df_dia_turno = (
        df_turnos
        .groupby(['DíaSemana', 'Turno'], observed=True)['P_EFECTIVIDAD']
        .mean()
        .reset_index()
        .rename(columns={'P_EFECTIVIDAD': 'Efectividad'})
    )

    # 3) Orden de días y turnos
    orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    orden_turnos = ['9–11', '12–14', '15–17', '18–21']

    df_dia_turno['DíaSemana'] = pd.Categorical(df_dia_turno['DíaSemana'], categories=orden_dias, ordered=True)
    df_dia_turno['Turno'] = pd.Categorical(df_dia_turno['Turno'], categories=orden_turnos, ordered=True)

    # 4) Redondeo
    df_dia_turno['Efectividad'] = df_dia_turno['Efectividad'].round(2)

    # 5) Plot
    fig = px.bar(
        df_dia_turno,
        x='DíaSemana',
        y='Efectividad',
        color='Turno',
        barmode='group',
        color_discrete_sequence=COLOR_SEQUENCE,
        category_orders={'DíaSemana': orden_dias, 'Turno': orden_turnos},
        labels={
            'DíaSemana': 'Día de la semana',
            'Efectividad': 'Efectividad promedio',
            'Turno': 'Franja horaria'
        },
        title=f'Efectividad promedio por día de la semana y por turno ({rango_seleccionado})'
    )
    fig.update_layout(
        yaxis_tickformat='.2f',
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE,
        legend_title_font_color=WHITE
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
    turno_map = {1: '9–11', 2: '12–14', 3: '15–17', 4: '18–21', 0: 'Fuera rango'}
    df_prod['Turno'] = df_prod['turno'].map(turno_map)

    # 3) Creamos el boxplot
    fig = px.box(
        df_prod,
        x='Turno',
        y='Productividad',
        points='outliers',
        color_discrete_sequence=[ACCENT_COLOR],
        labels={
            'Turno': 'Franja horaria',
            'Productividad': 'T_AO / DOTACION'
        },
        title=' '
    )

    # 4) Estilo acorde al tema oscuro
    fig.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE,
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
        .groupby(['Turno', 'DíaSemana'], observed=True)['Carga']
        .mean()
        .reset_index()
    )

    # 5) Generamos recomendaciones
    recs = []
    for _, row in stats.iterrows():
        turno = row['Turno']
        med = row['Mediana']
        iqr = row['IQR']
        sub = med_dia_turno[med_dia_turno['Turno'] == turno]
        peor = sub.loc[sub['Carga'].idxmax()]
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
    if 'Turno' in df_recs.columns and pd.api.types.is_numeric_dtype(df_recs['Turno']):
        df_recs['Turno'] = df_recs['Turno'].astype(int).apply(lambda x: f"{x:,}".replace(',', '.'))

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
        .agg(['mean', 'std', 'min', 'max'])
        .round(2)
        .reset_index()
    )

    # 3) Fecha de máximo y mínimo por turno
    idx_max = rendimiento.groupby('Turno')['Efectividad (%)'].idxmax()
    idx_min = rendimiento.groupby('Turno')['Efectividad (%)'].idxmin()
    maximos = rendimiento.loc[idx_max, ['Turno', 'Fecha', 'Efectividad (%)']].rename(
        columns={'Fecha': 'Fecha_max', 'Efectividad (%)': 'Maximo'}
    )
    minimos = rendimiento.loc[idx_min, ['Turno', 'Fecha', 'Efectividad (%)']].rename(
        columns={'Fecha': 'Fecha_min', 'Efectividad (%)': 'Minimo'}
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
    st.subheader("🏆 Efectividad por Turno")

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

    
    # Distribución de efectividad por turno
    df_box = df_turnos[df_turnos['turno'] > 0].copy()
    df_box['Turno'] = df_box['turno'].map(turno_map)
    fig_box_eff = px.box(
        df_box,
        x='Turno',
        y='P_EFECTIVIDAD',
        points='outliers',
        color_discrete_sequence=[ACCENT_COLOR],
        labels={'Turno': 'Turno', 'P_EFECTIVIDAD': 'Efectividad'},
        title=f'Distribuci\u00f3n de efectividad por turno ({rango_seleccionado})'
    )
    fig_box_eff.update_layout(
        plot_bgcolor=DARK_BG_COLOR,
        paper_bgcolor=DARK_BG_COLOR,
        font_color=WHITE,
        title_font_color=WHITE,
        yaxis_tickformat='.2f'
    )
    st.plotly_chart(fig_box_eff, use_container_width=True)

    st.markdown("---")

