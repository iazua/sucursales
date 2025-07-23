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
            series = df_branch[target]
            model = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            fname = f"sarima_{target}_{branch}.pkl"
            joblib.dump(model, os.path.join(MODEL_DIR, fname))


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

    horizon = (end_date - start_date).days + 1
    pred_index = pd.date_range(start_date, periods=horizon, freq="D")

    result_rows = []
    for fecha in pred_index:
        for h in HOURS_RANGE:
            result_rows.append({"COD_SUC": branch, "FECHA": fecha, "HORA": h})
    result = pd.DataFrame(result_rows)

    for t, model in models.items():
        forecast = model.get_forecast(steps=horizon)
        preds = forecast.predicted_mean.clip(lower=0)
        result[f"{t}_pred"] = np.repeat(preds.values, len(HOURS_RANGE))

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
