import os
from datetime import timedelta
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from preprocessing import basic_preprocess, prepare_features

MODEL_DIR = "models_simple"
TARGETS = ["T_VISITAS", "T_AO"]
HOURS_RANGE = list(range(9, 22))


def load_data(path: str = "data/DOTACION_EFECTIVIDAD.xlsx") -> pd.DataFrame:
    """Load raw data from Excel."""
    return pd.read_excel(path)


def train_models(df: pd.DataFrame) -> None:
    """Train a simple linear model for each target."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df_proc = basic_preprocess(df)
    for target in TARGETS:
        X, y = prepare_features(df_proc, target)
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODEL_DIR, f"predictor_{target}.pkl"))


def _load_model(target: str) -> LinearRegression:
    path = os.path.join(MODEL_DIR, f"predictor_{target}.pkl")
    return joblib.load(path)


def generate_predictions(
    df_hist: pd.DataFrame,
    branch: str,
    days: int = 7,
    efectividad_obj: float = 0.6,
) -> pd.DataFrame:
    """Generate naive hourly predictions for the next ``days`` days."""
    models = {t: _load_model(t) for t in TARGETS}
    last_date = df_hist[df_hist["COD_SUC"] == branch]["FECHA"].max()
    start_dt = pd.to_datetime(last_date) + timedelta(days=1)

    future_rows = []
    for d in range(days):
        fecha = start_dt + timedelta(days=d)
        for h in HOURS_RANGE:
            future_rows.append({"COD_SUC": branch, "FECHA": fecha, "HORA": h})
    df_future = pd.DataFrame(future_rows)
    df_proc = basic_preprocess(df_future)

    result = df_future.copy()
    for t in TARGETS:
        X, _ = prepare_features(df_proc, t)
        result[f"{t}_pred"] = models[t].predict(X).clip(min=0)

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
