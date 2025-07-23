# Predicción de Dotación por Sucursal

Este proyecto contiene un conjunto de scripts y una aplicación web para predecir
la dotación y otras métricas operacionales de distintas sucursales. La interfaz
web está implementada con **Streamlit** y utiliza modelos de series de tiempo
entrenados con **SARIMA**.

## Contenido del repositorio

- `app.py` – Aplicación Streamlit para visualizar pronósticos y métricas.
- `train_models.py` – Script de entrenamiento de modelos por sucursal.
- `preprocessing.py` y `utils.py` – Funciones auxiliares para
  procesamiento y modelamiento.
- `models_sarima/` – Modelos ya entrenados en formato `pkl`.
- `data/` – Archivos de datos de ejemplo (`DOTACION_EFECTIVIDAD.xlsx`,
  `regions.xlsx`, etc.).

## Instalación

1. Requiere **Python 3.8** o superior.
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Entrenamiento de modelos

Si deseas generar nuevamente los modelos, ejecuta:

```bash
python train_models.py
```

Los modelos resultantes se guardarán en la carpeta `models_sarima/`.

### Predicciones hasta fin de 2025

El script `train_models.py` permite especificar un rango de fechas para la
generación de pronósticos. Si no se indica otro periodo, los modelos se
entrenan y se generan proyecciones desde el día siguiente al último registro
disponible hasta el **31 de diciembre de 2025** (306 días si la base llega a
marzo de 2025).
El horizonte por defecto corresponde al valor de la constante
`DEFAULT_FORECAST_DAYS` definida en `train_models.py`.

Para este escenario de datos limitados, las proyecciones de 2025 toman como
referencia los valores observados en 2024.

La aplicación utiliza un método de inferencia vectorizado que acelera la
generación de pronósticos para horizontes extensos.

## Ejecución de la aplicación

Para lanzar la aplicación en modo local:

```bash
streamlit run app.py
```

Al abrirse, podrás seleccionar la sucursal y visualizar las predicciones
correspondientes. Para publicar la app en línea puedes usar servicios como
**Streamlit Community Cloud** o cualquier plataforma que ejecute aplicaciones
Python (Heroku, GCP, AWS, etc.).

## Licencia

Este proyecto se distribuye sin garantía y con fines demostrativos.
