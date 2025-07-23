import os
import pandas as pd
from preprocessing import forecast_target_prophet

PROPHET_DIR = "models_prophet"
TARGETS = ["T_VISITAS", "T_AO"]


def main(horizon_days: int = 365, changepoint_prior_scale: float = 0.5) -> None:
    os.makedirs(PROPHET_DIR, exist_ok=True)

    df = pd.read_excel("data/DOTACION_EFECTIVIDAD.xlsx")
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["COD_SUC"] = df["COD_SUC"].astype(str).str.strip()

    for branch in df["COD_SUC"].unique():
        df_branch = df[df["COD_SUC"] == branch]
        for target in TARGETS:
            df_t = df_branch[["FECHA", target]]
            forecast = forecast_target_prophet(
                df_t,
                target=target,
                horizon_days=horizon_days,
                changepoint_prior_scale=changepoint_prior_scale,
            )
            fname = f"{branch}_{target}_forecast.csv"
            forecast.to_csv(os.path.join(PROPHET_DIR, fname), index=False)


if __name__ == "__main__":
    main()
