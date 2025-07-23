import pandas as pd
import numpy as np
from prophet import Prophet


def assign_turno(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a simple shift id based on the hour of day."""
    bins = [8, 11, 14, 17, 21]
    labels = [1, 2, 3, 4]
    df = df.copy()
    df["turno"] = (
        pd.cut(
            df["HORA"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        .cat.add_categories([0])
        .fillna(0)
        .astype(int)
    )
    return df


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing used for training and inference."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["weekday"] = df["FECHA"].dt.weekday
    df["month"] = df["FECHA"].dt.month
    df = assign_turno(df)
    df["COD_SUC"] = df["COD_SUC"].astype("category").cat.codes
    return df


def prepare_features(df: pd.DataFrame, target: str):
    """Return feature matrix X and target y for the given variable."""
    df_proc = basic_preprocess(df)
    features = ["HORA", "weekday", "month", "turno", "COD_SUC"]
    X = df_proc[features].fillna(0)
    y = df_proc[target].fillna(0)
    return X, y


def forecast_dotacion_prophet(
    df_dotacion: pd.DataFrame,
    model: Prophet | None = None,
    horizon_days: int = 30,
    noise_scale: float = 0.0,
    changepoint_prior_scale: float = 0.5,
) -> pd.DataFrame:
    """Forecast dotacion using Facebook Prophet.

    Parameters
    ----------
    df_dotacion : pd.DataFrame
        DataFrame with columns ``['fecha', 'dotacion']``.
    model : Prophet, optional
        Pre-initialized Prophet model. If ``None`` a new model is created.
    horizon_days : int, default 30
        Number of days to forecast into the future.
    noise_scale : float, default 0.0
        Standard deviation of Gaussian noise added to ``yhat``.
    changepoint_prior_scale : float, default 0.5
        Value for Prophet's ``changepoint_prior_scale`` when creating a new model.

    Returns
    -------
    pd.DataFrame
        Forecast with columns ``['ds','yhat','yhat_lower','yhat_upper']`` for the
        forecast horizon.
    """

    df_prophet = (
        df_dotacion.rename(columns={"fecha": "ds", "dotacion": "y"})
        .copy()
    )
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    if model is None:
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        model.add_country_holidays(country_name="CL")
        model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(
        horizon_days
    ).reset_index(drop=True)

    if noise_scale > 0:
        noise = np.random.normal(scale=noise_scale, size=len(result))
        result["yhat"] += noise

    return result


def forecast_target_prophet(
    df_target: pd.DataFrame,
    target: str,
    model: Prophet | None = None,
    horizon_days: int = 30,
    noise_scale: float = 0.0,
    changepoint_prior_scale: float = 0.5,
) -> pd.DataFrame:
    """Forecast a generic target (e.g. T_VISITAS, T_AO) using Prophet."""
    df_prophet = (
        df_target.rename(columns={"FECHA": "ds", target: "y"})
        .copy()
    )
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    if model is None:
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        model.add_country_holidays(country_name="CL")
        model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days).reset_index(drop=True)

    if noise_scale > 0:
        noise = np.random.normal(scale=noise_scale, size=len(result))
        result["yhat"] += noise

    return result
