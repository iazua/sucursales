import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import forecast_target_prophet

PROPHET_DIR = "models_prophet"
TARGETS = ["T_VISITAS", "T_AO"]


def load_data(path: str = "data/DOTACION_EFECTIVIDAD.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["COD_SUC"] = df["COD_SUC"].astype(str).str.strip()
    return df


def deploy_forecasts(df: pd.DataFrame, horizon_days: int = 365, changepoint_prior_scale: float = 0.5) -> None:
    os.makedirs(PROPHET_DIR, exist_ok=True)

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
            csv_name = f"{branch}_{target}_forecast.csv"
            forecast.to_csv(os.path.join(PROPHET_DIR, csv_name), index=False)

            plt.figure()
            plt.plot(forecast["ds"], forecast["yhat"], label="forecast")
            plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3)
            plt.title(f"{target} forecast for branch {branch}")
            plt.xlabel("date")
            plt.ylabel(target)
            plt.legend()
            plt.tight_layout()
            plot_name = f"{branch}_{target}_forecast.png"
            plt.savefig(os.path.join(PROPHET_DIR, plot_name))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate and plot Prophet forecasts")
    parser.add_argument("--horizon_days", type=int, default=365, help="Days to forecast")
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.5, help="Prophet changepoint prior scale")
    args = parser.parse_args()

    df = load_data()
    deploy_forecasts(df, horizon_days=args.horizon_days, changepoint_prior_scale=args.changepoint_prior_scale)


if __name__ == "__main__":
    main()
