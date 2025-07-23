import os
from datetime import timedelta
import joblib
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX





MODEL_DIR = "models_sarima"
TARGETS = ["T_VISITAS", "T_AO"]
HOURS_RANGE = list(range(9, 22))


def load_data(path: str = "data/DOTACION_EFECTIVIDAD.xlsx") -> pd.DataFrame:
    """Load raw data from Excel."""
    return pd.read_excel(path)


def train_models(df: pd.DataFrame) -> None:
    """Train a SARIMA model for each branch and target."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["COD_SUC"] = df["COD_SUC"].astype(str).str.strip()

    # Keep only Monday-Friday records
    df = df[df["FECHA"].dt.weekday < 5]

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
            .asfreq("B")
            .fillna(0)
        )

        exog = pd.get_dummies(df_branch.index.weekday, drop_first=False)
        exog = exog.astype(float)  # Ensure numeric dtype to avoid bool diff issue
        # Ensure exogenous variables use the same index as the target series
        exog.index = df_branch.index

        for target in TARGETS:
            series = df_branch[target]
            model = SARIMAX(
                series,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 5),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            fname = f"sarima_{target}_{branch}.pkl"
            joblib.dump({"model": model, "exog_cols": exog.columns.tolist()}, os.path.join(MODEL_DIR, fname))


def _load_model(target: str, branch: str):
    fname = f"sarima_{target}_{branch}.pkl"
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

    pred_index = pd.bdate_range(start_date, end_date)
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
        df_h = df_h[df_h["FECHA"].dt.weekday < 5]
        daily_totals = df_h.groupby("FECHA")[target].transform("sum")
        df_h = df_h[daily_totals > 0]
        df_h["share"] = df_h[target] / daily_totals
        weights = (
            df_h.groupby("HORA")["share"].mean().reindex(HOURS_RANGE, fill_value=1/len(HOURS_RANGE))
        )
        w = weights.values
        return w / w.sum()

    for t in TARGETS:
        model = None
        try:
            mdl = _load_model(t, branch)
            model = mdl["model"]
            exog_cols = mdl["exog_cols"]
            exog_future = pd.get_dummies(pred_index.weekday, drop_first=False)
            exog_future = exog_future.astype(float).reindex(columns=exog_cols, fill_value=0.0)
            exog_future.index = pred_index
            daily_preds = model.forecast(steps=horizon, exog=exog_future)
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
            if model is not None:
                resid_std = float(np.sqrt(getattr(model, "sigma2", 0.0)))
            else:
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
    result["DOTACION_req"] = (result["T_AO_pred"] / 10).round().astype(int)
    return result


if __name__ == "__main__":
    data = load_data()
    train_models(data)
    branch = str(data["COD_SUC"].iloc[0])
    preds = generate_predictions(data, branch)
    print(preds.head())
