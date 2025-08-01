# Predicción de Dotación por Sucursal

Este proyecto contiene un conjunto de scripts y una aplicación web para predecir
la dotación y otras métricas operacionales de distintas sucursales. La interfaz
web está implementada con **Streamlit** y utiliza modelos de series de tiempo
entrenados con **Prophet**.

## Contenido del repositorio

- `app.py` – Aplicación Streamlit para visualizar pronósticos y métricas.
- `train_models.py` – Entrena modelos Prophet y genera pronósticos.
- `deploy_prophet.py` – Pronósticos Prophet y gráficos para visitas y acepta oferta.
- `preprocessing.py` y `utils.py` – Funciones auxiliares para
  procesamiento y modelamiento.
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

Los modelos resultantes se guardarán en la carpeta `models_prophet/`.

`deploy_prophet.py` utiliza esos modelos guardados para generar los pronósticos.
Solo necesitas ejecutarlo después de entrenar para obtener los CSV y gráficos.


Para generar pronósticos con Prophet de *T_VISITAS* y *T_AO* y guardar una
gráfica por variable ejecuta:
```bash
python deploy_prophet.py --horizon_days 90 --changepoint_prior_scale 0.3
```
Los archivos (CSV y PNG) se guardarán en `models_prophet/`.

### Predicciones hasta fin de 2025

El script `train_models.py` permite especificar un rango de fechas para la
generación de pronósticos. Por defecto, el ejemplo generado cubre todos los
días restantes de 2025 a contar del último registro disponible (aproximadamente
**306 días** si la data llega a marzo de 2025).

La aplicación utiliza un método vectorizado que acelera la inferencia para
horizontes extensos y las proyecciones para el periodo abril–diciembre de 2025
se basan en los promedios históricos de 2024.

El módulo `generate_predictions` ahora permite añadir ruido gaussiano opcional
(parámetro `noise_scale`) para simular la volatilidad observada en la data real.
Además, la dotación requerida se calcula optimizando cada día por separado,
de modo que las jornadas con menor actividad no asignen personal de forma
excesiva. Desde esta versión la curva de efectividad se ajusta para cada día de
la semana, mejorando la estimación de dotación según la jornada pronosticada.

## Ejecución de la aplicación

Para lanzar la aplicación en modo local:

```bash
streamlit run app.py
```

Al abrirse, podrás seleccionar la sucursal y visualizar las predicciones
correspondientes. Para publicar la app en línea puedes usar servicios como
**Streamlit Community Cloud** o cualquier plataforma que ejecute aplicaciones
Python (Heroku, GCP, AWS, etc.).

- Sobre la tabla de predicciones podrás descargar los resultados en formato
  Excel mediante el botón **"Descargar tabla en Excel"**.




**Ignacio Azua**
