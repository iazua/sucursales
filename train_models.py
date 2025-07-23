import os
from datetime import timedelta
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from preprocessing import basic_preprocess, prepare_features





MODEL_DIR = "models_xgb"
TARGETS = ["T_VISITAS", "T_AO"]
HOURS_RANGE = list(range(9, 22))


def load_data(path: str = "data/DOTACION_EFECTIVIDAD.xlsx") -> pd.DataFrame:
    """Load raw data from Excel."""
    return pd.read_excel(path)


def train_models(df: pd.DataFrame) -> None:
    """Train an XGBoost model for each branch and target using time-series cross-validation."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["COD_SUC"] = df["COD_SUC"].astype(str).str.strip()

    # Keep only Monday-Friday records
    df = df[df["FECHA"].dt.weekday < 5]

    for branch in df["COD_SUC"].unique():
        df_branch = df[df["COD_SUC"] == branch].sort_values("FECHA")

        for target in TARGETS:
            X, y = prepare_features(df_branch, target)

            tscv = TimeSeriesSplit(n_splits=3)
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=0,
                n_estimators=500,
                eval_metric="rmse",
            )
            param_grid = {
                "max_depth": [3, 4, 6],
                "learning_rate": [0.1, 0.05],
                "subsample": [0.8, 1.0],
            }
            grid = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            grid.fit(X, y)
            best_model = grid.best_estimator_

            fname = f"xgb_{target}_{branch}.pkl"
            joblib.dump(best_model, os.path.join(MODEL_DIR, fname))


def _load_model(target: str, branch: str):
    fname = f"xgb_{target}_{branch}.pkl"
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
    """Generate hourly predictions for the next ``days`` days using XGBoost models."""

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

    result_rows = [
        {"COD_SUC": branch, "FECHA": fecha, "HORA": h}
        for fecha in pred_index for h in HOURS_RANGE
    ]
    result = pd.DataFrame(result_rows)

    df_pred = basic_preprocess(result)
    X_pred = df_pred[["HORA", "weekday", "month", "turno", "COD_SUC"]].fillna(0)

    for t in TARGETS:
        model = _load_model(t, branch)
        preds = model.predict(X_pred)
        if noise_scale > 0:
            resid_std = float(np.std(preds))
            noise = np.random.normal(scale=resid_std * noise_scale, size=len(preds))
            preds = np.clip(preds + noise, a_min=0, a_max=None)
        result[f"{t}_pred"] = preds

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
