import os
import argparse
from datetime import timedelta
import joblib
import pandas as pd
import numpy as np
from prophet import Prophet
from preprocessing import forecast_dotacion_prophet, forecast_target_prophet
from utils import estimar_parametros_efectividad, estimar_dotacion_optima

MODEL_DIR = "models_prophet"
PROPHET_DIR = "models_prophet"
TARGETS = ["T_VISITAS", "T_AO"]
HOURS_RANGE = list(range(9, 22))

def load_data(path: str = "data/DOTACION_EFECTIVIDAD.xlsx") -> pd.DataFrame:
    """Load raw data from Excel."""
    return pd.read_excel(path)


def train_models(
    df: pd.DataFrame,
    horizon_days: int = 365,
    noise_scale: float = 0.0,
    changepoint_prior_scale: float = 0.5,
) -> None:
    """Train a Prophet model for each branch and target and save forecasts."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROPHET_DIR, exist_ok=True)

    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["COD_SUC"] = df["COD_SUC"].astype(str).str.strip()


    # Aggregate at daily level
    df_daily = (
        df.groupby(["COD_SUC", "FECHA"])[TARGETS]
        .sum()
        .sort_index()
        .reset_index()
    )

    for branch in df_daily["COD_SUC"].unique():
        df_branch = (
            df_daily[df_daily["COD_SUC"] == branch]
            .set_index("FECHA")
            .asfreq("D")
            .fillna(0)
        )

        for target in TARGETS:
            df_p = df_branch[[target]].reset_index().rename(columns={"FECHA": "ds", target: "y"})
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=changepoint_prior_scale,
            )
            model.add_country_holidays(country_name="CL")
            model.fit(df_p)

            fname = f"prophet_{target}_{branch}.pkl"
            joblib.dump(model, os.path.join(MODEL_DIR, fname))
        # --- Prophet forecast for DOTACION ---
        df_dot = (
            df[df["COD_SUC"] == branch][["FECHA", "DOTACION"]]
            .groupby("FECHA", as_index=False)
            .mean()
            .rename(columns={"FECHA": "fecha", "DOTACION": "dotacion"})
        )
        prophet_forecast = forecast_dotacion_prophet(
            df_dot,
            horizon_days=horizon_days,
            noise_scale=noise_scale,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        forecast_path = os.path.join(PROPHET_DIR, f"{branch}_forecast.csv")
        prophet_forecast.to_csv(forecast_path, index=False)

        # --- Prophet forecasts for operational targets ---
        for tgt in TARGETS:
            df_t = df_branch[[tgt]].reset_index()
            prophet_t = forecast_target_prophet(
                df_t,
                target=tgt,
                horizon_days=horizon_days,
                changepoint_prior_scale=changepoint_prior_scale,
            )
            path_t = os.path.join(PROPHET_DIR, f"{branch}_{tgt}_forecast.csv")
            prophet_t.to_csv(path_t, index=False)


def _load_model(target: str, branch: str):
    fname = f"prophet_{target}_{branch}.pkl"
    path = os.path.join(MODEL_DIR, fname)
    return joblib.load(path)


def generate_predictions(
    df_hist: pd.DataFrame,
    branch: str,
    days: int = 7,
    efectividad_obj: float = 0.6,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    noise_scale: float = 0.0,
) -> pd.DataFrame:
    """Generate hourly predictions for the next ``days`` days.

    When forecasting beyond the available history (after March 2025) the
    function falls back to a seasonal average based on 2024 data. This allows
    projecting the remainder of 2025 even with limited observations. If
    ``noise_scale`` is greater than 0, Gaussian noise proportional to the
    training residuals is added to produce more realistic volatility.
    """

    df_hist = df_hist.copy()
    df_hist.columns = df_hist.columns.str.strip().str.upper()
    df_hist["FECHA"] = pd.to_datetime(df_hist["FECHA"])
    df_hist["COD_SUC"] = df_hist["COD_SUC"].astype(str).str.strip()

    if start_date is None:
        last_date = df_hist[df_hist["COD_SUC"] == branch]["FECHA"].max()
        start_date = pd.to_datetime(last_date) + timedelta(days=1)

    if end_date is None:
        end_date = start_date + timedelta(days=days - 1)

    pred_index = pd.date_range(start_date, end_date)
    horizon = len(pred_index)

    result_rows = [
        {"COD_SUC": branch, "FECHA": fecha, "HORA": h}
        for fecha in pred_index for h in HOURS_RANGE
    ]
    result = pd.DataFrame(result_rows)

    # --- Seasonal baseline from year 2024 ---
    df_branch = df_hist[df_hist["COD_SUC"] == branch].copy()
    df_branch["month"] = df_branch["FECHA"].dt.month
    df_branch["weekday"] = df_branch["FECHA"].dt.weekday
    df_2024 = df_branch[df_branch["FECHA"].dt.year == 2024]
    baseline = (
        df_2024.groupby(["month", "weekday"])[TARGETS]
        .mean()
    )

    def _hourly_distribution(df_hist: pd.DataFrame, branch: str, target: str) -> np.ndarray:
        df_h = df_hist.copy()
        df_h.columns = df_h.columns.str.strip().str.upper()
        df_h["FECHA"] = pd.to_datetime(df_h["FECHA"])
        df_h = df_h[df_h["COD_SUC"].astype(str).str.strip() == branch]
        daily_totals = df_h.groupby("FECHA")[target].transform("sum")
        df_h = df_h[daily_totals > 0]
        df_h["share"] = df_h[target] / daily_totals
        weights = (
            df_h.groupby("HORA")["share"].mean().reindex(HOURS_RANGE, fill_value=1/len(HOURS_RANGE))
        )
        w = weights.values
        return w / w.sum()

    for t in TARGETS:
        try:
            model = _load_model(t, branch)
            future = model.make_future_dataframe(periods=horizon, freq="D")
            forecast = model.predict(future)
            daily_preds = forecast.set_index("ds").reindex(pred_index)["yhat"].fillna(0).values
        except FileNotFoundError:
            daily_preds = []
            for d in pred_index:
                key = (d.month, d.weekday())
                if key in baseline.index:
                    val = baseline.loc[key, t]
                else:
                    val = baseline[t].mean()
                daily_preds.append(val if pd.notna(val) else 0)

        if isinstance(daily_preds, pd.Series):
            daily_preds = daily_preds.values

        if noise_scale > 0:
            resid_std = float(df_branch[t].std())
            noise = np.random.normal(scale=resid_std * noise_scale, size=len(daily_preds))
            daily_preds = np.clip(daily_preds + noise, a_min=0, a_max=None)

        weights = _hourly_distribution(df_hist, branch, t)
        hourly_preds = np.concatenate([
            daily_preds[i] * weights for i in range(horizon)
        ])
        result[f"{t}_pred"] = hourly_preds


    result["T_AO_VENTA_req"] = result["T_AO_pred"] * efectividad_obj
    result["P_EFECTIVIDAD_req"] = np.where(
        result["T_AO_pred"] > 0,
        result["T_AO_VENTA_req"] / result["T_AO_pred"],
        0,
    )
    # --- Required staffing based on desired effectiveness ---
    hist_eff = df_branch[["FECHA", "DOTACION", "T_AO", "T_AO_VENTA"]].dropna()
    if len(hist_eff) >= 3:
        params_eff_global = estimar_parametros_efectividad(hist_eff)
    else:
        params_eff_global = {"L": 1.0, "k": 0.5, "x0_base": 5.0, "x0_factor_t_ao_venta": 0.05}

    # Estimar parámetros por día de la semana para capturar variaciones
    params_by_weekday = {}
    hist_eff["weekday"] = hist_eff["FECHA"].dt.weekday
    for wd in range(7):
        df_wd = hist_eff[hist_eff["weekday"] == wd]
        if len(df_wd) >= 3:
            params_by_weekday[wd] = estimar_parametros_efectividad(df_wd)
        else:
            params_by_weekday[wd] = params_eff_global

    max_dot = float(df_branch["DOTACION"].max()) if "DOTACION" in df_branch.columns else 0

    result["DOTACION_req"] = 0
    for fecha, grp in result.groupby("FECHA"):
        weekday = fecha.weekday()
        params_eff = params_by_weekday.get(weekday, params_eff_global)
        dot_opt, _ = estimar_dotacion_optima(
            grp["T_AO_pred"],
            grp["T_AO_VENTA_req"],
            efectividad_obj,
            params_eff,
        )
        dot_final = int(np.minimum(dot_opt, max_dot))
        result.loc[result["FECHA"] == fecha, "DOTACION_req"] = dot_final


    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Prophet models and forecasts")
    parser.add_argument("--horizon_days", type=int, default=365, help="Days to forecast with Prophet")
    parser.add_argument("--noise_scale", type=float, default=0.0, help="Gaussian noise scale for Prophet forecast")
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.5, help="Prophet changepoint prior scale")
    args = parser.parse_args()

    data = load_data()
    train_models(
        data,
        horizon_days=args.horizon_days,
        noise_scale=args.noise_scale,
        changepoint_prior_scale=args.changepoint_prior_scale,
    )
