# Informe Detallado del Proyecto de Predicción de Dotación

Este documento describe en profundidad la estructura del repositorio, la lógica de funcionamiento de cada módulo y las razones por las cuales los resultados que produce son relevantes para la gestión de sucursales. Además, se presentan algunas recomendaciones para sacar el mayor provecho de la herramienta.

## 1. Descripción general

El repositorio contiene scripts de entrenamiento, despliegue y una aplicación web que permiten pronosticar métricas operacionales por sucursal, tales como visitas, aceptación de ofertas y dotación requerida. Según se indica en el archivo `README.md`, la aplicación web está implementada con **Streamlit** y los modelos de series de tiempo se entrenan utilizando **Prophet**.

## 2. Estructura del repositorio

El `README` resume el contenido principal del proyecto:

```text
- app.py – Aplicación Streamlit para visualizar pronósticos y métricas.
- train_models.py – Entrena modelos Prophet y genera pronósticos.
- deploy_prophet.py – Pronósticos Prophet y gráficos para visitas y acepta oferta.
- preprocessing.py y utils.py – Funciones auxiliares para procesamiento y modelamiento.
- data/ – Archivos de datos de ejemplo (DOTACION_EFECTIVIDAD.xlsx, regions.xlsx, etc.).
```

La carpeta `data/` contiene los insumos iniciales para entrenamiento, incluyendo un registro histórico de dotación y un listado de regiones para ubicar cada sucursal.

## 3. Lógica de funcionamiento

### 3.1 Entrenamiento y generación de pronósticos

El script `train_models.py` se encarga de entrenar modelos Prophet por sucursal y variable objetivo. Además, implementa la función `generate_predictions`, que produce pronósticos horarios y cuenta con un mecanismo de respaldo cuando se desean proyecciones más allá de los datos disponibles:

```python
"""Generate hourly predictions for the next ``days`` days.

When forecasting beyond the available history (after March 2025) the
function falls back to a seasonal average based on 2024 data. This allows
projecting the remainder of 2025 even with limited observations. If
``noise_scale`` is greater than 0, Gaussian noise proportional to the
training residuals is added to produce more realistic volatility.
"""
```

Esta lógica aprovecha los promedios históricos de 2024 como base para el periodo abril–diciembre de 2025. Además, permite inyectar ruido gaussiano para simular la volatilidad real, según se explica también en el `README`.

### 3.2 Aplicación web

El archivo `app.py` define la interfaz en Streamlit. Carga la información histórica, la cruza con datos de región y genera un mapa interactivo junto con varias pestañas de análisis. Para calcular rápidamente los pronósticos, se utiliza la función `generate_predictions` de forma vectorizada y con almacenamiento en caché:

```python
@st.cache_data(show_spinner="Generando pronóstico…")
def forecast_fast(df_all: pd.DataFrame,
                  cod_suc: int,
                  efect_obj: float,
                  days: int) -> pd.DataFrame:
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
    )
```

La app incluye gráficas con Plotly, un mapa de ubicación de sucursales mediante PyDeck y distintas tablas de resumen histórico y proyectado.

### 3.3 Modelado de efectividad

En `utils.py` se define una familia de funciones para modelar la probabilidad de efectividad (`P_EFECTIVIDAD`) como una sigmoide dependiente de la dotación y de las ventas concretadas:

```python
def calcular_efectividad(t_ao, t_ao_venta):
    """Calcula la probabilidad de efectividad (P_EFECTIVIDAD) basada en T_AO y T_AO_VENTA."""
    return np.where(t_ao > 0, t_ao_venta / t_ao, 0)

def _modelo_efectividad(dotacion, t_ao_venta, params):
    """Modelo no lineal (sigmoide) para la P_EFECTIVIDAD en función de DOTACION y T_AO_VENTA."""
    if not isinstance(dotacion, (int, float, np.number)) or dotacion <= 0:
        return 0.0
    L = params.get('L', 1.0)
    k = params.get('k', 0.5)
    x0_base = params.get('x0_base', 5.0)
    x0_factor_t_ao_venta = params.get('x0_factor_t_ao_venta', 0.05)
    x0 = x0_base - x0_factor_t_ao_venta * t_ao_venta
    x0 = max(1.0, x0)
    exp_term = -k * (dotacion - x0)
    if exp_term > np.log(np.finfo(float).max / (L if L > 0 else 1.0)):
        return L
    if exp_term < np.log(np.finfo(float).tiny):
        return 0.0
    return L / (1 + np.exp(exp_term))
```

Estos cálculos permiten estimar cuánta dotación se requiere para alcanzar una efectividad deseada, aportando una base cuantitativa para la toma de decisiones.

## 4. Fundamentos de relevancia de los resultados

1. **Modelos de series de tiempo robustos**. Los pronósticos se generan con Prophet, que captura estacionalidades diarias, semanales y anuales. Además, se consideran feriados de Chile para mejorar la precisión.
2. **Vectorización y aprovechamiento del historial**. El `README` destaca que la app utiliza un método vectorizado para acelerar la inferencia en horizontes extensos y que las proyecciones se basan en los promedios de 2024 cuando no hay datos suficientes.
3. **Ajuste mediante ruido gaussiano**. Para reflejar mejor la volatilidad real, `generate_predictions` permite incorporar ruido gaussiano controlado por el usuario.
4. **Parámetros por día de la semana**. Los coeficientes de la curva de efectividad se recalculan según el día, de modo que la dotación requerida refleje la estacionalidad semanal.
5. **Modelado de efectividad**. El empleo de una función sigmoide parametrizada permite estimar la relación entre dotación y ventas concretadas, ofreciendo una visión directa del impacto de aumentar o reducir personal.

## 5. Beneficios de un buen uso de la herramienta

- **Planificación de personal**. Con pronósticos por hora y día, es posible alinear la dotación con la demanda esperada, optimizando recursos y mejorando la atención al cliente.
- **Evaluación de metas**. El análisis de efectividad permite fijar objetivos realistas y monitorear qué tan cerca está cada sucursal de alcanzarlos.
- **Exploración interactiva**. La aplicación Streamlit ofrece gráficos, mapas y tablas que facilitan comprender patrones históricos y proyecciones futuras sin necesidad de conocimientos técnicos profundos.
- **Extensibilidad**. Los scripts de entrenamiento y despliegue pueden adaptarse a nuevos horizontes o variables, permitiendo que el sistema evolucione junto con el negocio.

## 6. Conclusiones

El proyecto presenta una solución integral para pronosticar y analizar la dotación de sucursales. Su arquitectura modular y el soporte para ajustes personalizados (por ejemplo, el ruido gaussiano o la calibración de la sigmoide de efectividad) facilitan que se adapte a distintos escenarios. Usado adecuadamente, constituye una herramienta valiosa para la gestión operativa y la toma de decisiones basadas en datos.

