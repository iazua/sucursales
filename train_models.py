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
) -> pd.DataFrame:
    """Generate hourly predictions for the next ``days`` days using SARIMA."""

    models = {t: _load_model(t, branch) for t in TARGETS}

    if start_date is None:
        last_date = df_hist[df_hist["COD_SUC"] == branch]["FECHA"].max()
        start_date = pd.to_datetime(last_date) + timedelta(days=1)

    if end_date is None:
        end_date = start_date + timedelta(days=days - 1)

    pred_index = pd.bdate_range(start_date, periods=days)
    horizon = len(pred_index)

    result_rows = []
    for fecha in pred_index:
        for h in HOURS_RANGE:
            result_rows.append({"COD_SUC": branch, "FECHA": fecha, "HORA": h})
    result = pd.DataFrame(result_rows)

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

    for t, model_info in models.items():
        model = model_info["model"]
        cols = model_info["exog_cols"]
        exog_pred = pd.get_dummies(pred_index.weekday, drop_first=False)
        # Align forecast exogenous data index with prediction horizon
        exog_pred.index = pred_index
        exog_pred = exog_pred.reindex(columns=cols, fill_value=0)
        forecast = model.get_forecast(steps=horizon, exog=exog_pred)
        preds = forecast.predicted_mean.clip(lower=0)
        weights = _hourly_distribution(df_hist, branch, t)
        hourly_preds = np.concatenate([preds.values[i] * weights for i in range(horizon)])
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
