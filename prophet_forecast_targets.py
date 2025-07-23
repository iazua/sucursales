import os
import argparse
import pandas as pd
from prophet import Prophet

PROPHET_DIR = "models_prophet"
TARGETS = ["T_VISITAS", "T_AO"]


def _forecast_branch(df_branch: pd.DataFrame, target: str, horizon_days: int, cp_scale: float = 0.5) -> pd.DataFrame:
    df_prophet = df_branch[["FECHA", target]].rename(columns={"FECHA": "ds", target: "y"}).copy()
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=cp_scale,
    )
    model.add_country_holidays(country_name="CL")
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)


def main(horizon_days: int = 365, changepoint_prior_scale: float = 0.5) -> None:
    os.makedirs(PROPHET_DIR, exist_ok=True)

    df = pd.read_excel("data/DOTACION_EFECTIVIDAD.xlsx")
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["COD_SUC"] = df["COD_SUC"].astype(str).str.strip()

    for branch in df["COD_SUC"].unique():
        df_branch = df[df["COD_SUC"] == branch]
        for target in TARGETS:
            forecast = _forecast_branch(df_branch, target, horizon_days, changepoint_prior_scale)
            fname = f"{branch}_{target}_forecast.csv"
            forecast.to_csv(os.path.join(PROPHET_DIR, fname), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Prophet forecasts for T_VISITAS and T_AO"
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=365,
        help="Number of days to forecast",
    )
    parser.add_argument(
        "--changepoint_prior_scale",
        type=float,
        default=0.5,
        help="Prophet changepoint prior scale",
    )
    args = parser.parse_args()
    main(
        horizon_days=args.horizon_days,
        changepoint_prior_scale=args.changepoint_prior_scale,
    )
