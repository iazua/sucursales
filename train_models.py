import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import prepare_features, assign_turno, add_spike_label
from utils import estimar_dotacion_optima, estimar_parametros_efectividad, calcular_efectividad
from scipy.stats import randint, uniform          # para RandomizedSearchCV
# --- CONSTANTES ---
MODEL_DIR = 'models_lgbm'
os.makedirs(MODEL_DIR, exist_ok=True)

HOURS_RANGE = list(range(9, 22))  # Rango horario 9–21
# Fecha límite para las predicciones automáticas
PREDICTION_END_DATE = pd.Timestamp("2025-12-31")

# --- CONFIGURACIÓN DE MODELOS ---
DEFAULT_PARAMS = {
    "num_leaves": [31, 63, 127],
    "max_depth": [5, 7, 9],
    "learning_rate": [0.03, 0.05, 0.1],   # mantén 3 valores
    "n_estimators": [300, 500],
    "min_child_samples": [20, 50],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5]
}

# --- FUNCIONES PRINCIPALES ---
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Carga y preprocesa los datos manteniendo consistencia con el deploy"""
    df = pd.read_excel(file_path)

    # Limpieza y estandarización de columnas
    df.columns = df.columns.str.strip().str.upper()
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    # Validación de columnas esenciales
    required_cols = ['COD_SUC', 'FECHA', 'HORA', 'T_VISITAS', 'T_AO', 'T_AO_VENTA', 'DOTACION']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")

    # Ordenamiento temporal (crucial para series de tiempo)
    df = df.sort_values(['FECHA', 'HORA', 'COD_SUC']).reset_index(drop=True)

    # Asegurar P_EFECTIVIDAD
    if 'P_EFECTIVIDAD' not in df.columns:
        df['P_EFECTIVIDAD'] = calcular_efectividad(df['T_AO'], df['T_AO_VENTA'])

    # Crear etiqueta de picos de T_AO
    df = add_spike_label(df)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# train_models.py
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import lightgbm as lgb
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

# DEFAULT_PARAMS debe estar definido como antes o importado
# HOURS_RANGE y PREDICTION_END_DATE se mantienen sin cambios


def train_model_for_branch(
        df_branch: pd.DataFrame,
        target: str,
        param_dist: dict | None = None,
        n_iter: int = 25,
        early_stopping_rounds: int = 30
) -> lgb.LGBMRegressor:
    """
    Entrena un modelo LightGBM para una sucursal y un target.
    Mejoras clave:
    • include_time_features=True → hora/turno + flags calendario
    • objective='poisson'       → mejor para conteos con cola pesada
    • sample_weight simétrico   → 10× para picos (≥ p97) y caídas (≤ p3)
    """

    # 1) FEATURES ────────────────────────────────────────────────────────────
    X, y = prepare_features(
        df_branch,
        target,
        is_prediction=False,
        include_time_features=True
    )

    # 2) PESOS: 10× para extremos (p3 / p97) ────────────────────────────────
    p3, p97 = np.percentile(y, [3, 97])
    weights_all = np.where(
        (y >= p97) | (y <= p3), 10.0,  # picos altos O caídas profundas
        1.0                            # resto
    )

    # 3) SPLIT TEMPORAL HOLD-OUT ─────────────────────────────────────────────
    val_size = max(5, int(0.1 * len(X)))
    X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
    w_tr, w_val = weights_all[:-val_size], weights_all[-val_size:]

    # 4) ESPACIO DE HYPERPARAMS ──────────────────────────────────────────────
    if param_dist is None:
        param_dist = {
            "objective": ["poisson"],
            "metric": ["poisson"],
            "learning_rate": uniform(0.02, 0.15),
            "num_leaves": randint(31, 128),
            "min_data_in_leaf": randint(3, 15),
            "feature_fraction": uniform(0.6, 0.4),
            "lambda_l1": uniform(0.0, 0.2)
        }

    base = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

    # Validación temporal para evitar fugas de información
    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    print(f"🔍 Buscando hiperparámetros para {target} …")
    search.fit(X_tr, y_tr, sample_weight=w_tr)

    best = search.best_estimator_
    print(f"🏆  Seleccionado: {search.best_params_}")

    # 5) ENTRENAMIENTO FINAL + EARLY-STOPPING ───────────────────────────────
    best.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(20)
        ]
    )

    return best


def train_spike_classifier(
        df_branch: pd.DataFrame,
        n_estimators: int = 200
) -> lgb.LGBMClassifier:
    """Entrena un clasificador binario para predecir picos de ``T_AO``."""

    X, _ = prepare_features(
        df_branch,
        "T_AO",
        is_prediction=False,
        include_time_features=True,
    )
    valid = df_branch["T_AO"].notna()
    y = df_branch.loc[valid, "is_spike"].fillna(0).astype(int)

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf



def train_all_models(
        df: pd.DataFrame,
        targets: list = ['T_VISITAS', 'T_AO'],
        n_iter: int = 20
) -> None:
    """Entrena modelos para todas las sucursales y targets especificados"""
    branches = df['COD_SUC'].unique()

    for target in targets:
        print(f"\n{'=' * 20}")
        print(f"🚀 ENTRENANDO MODELOS PARA {target}")
        print(f"{'=' * 20}")

        for i, branch in enumerate(branches, 1):
            df_branch = df[df['COD_SUC'] == branch].copy()

            # Verificación de datos mínimos
            if len(df_branch) < 30:
                print(f"⏭️ Sucursal {branch} ({i}/{len(branches)}): Solo {len(df_branch)} registros. Saltando...")
                continue

            print(f"\n🏢 Sucursal {branch} ({i}/{len(branches)}) - {len(df_branch)} registros")

            try:
                model = train_model_for_branch(df_branch, target, n_iter=n_iter)

                # Guardar modelo
                model_path = os.path.join(MODEL_DIR, f"predictor_{target}_{branch}.pkl")
                joblib.dump(model, model_path)
                print(f"💾 Modelo guardado en {model_path}")

                if target == 'T_AO':
                    clf = train_spike_classifier(df_branch)
                    clf_path = os.path.join(MODEL_DIR, f"spike_{branch}.pkl")
                    joblib.dump(clf, clf_path)
                    print(f"💾 Clasificador de spikes guardado en {clf_path}")

            except Exception as e:
                print(f"❌ Error entrenando sucursal {branch}: {str(e)}")
                continue

# --- train_models.py ---

def generate_predictions(
        df: pd.DataFrame,
        branch: int,
        targets: list = ["T_VISITAS", "T_AO"],
        efectividad_obj: float = 0.62,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Genera predicciones para las horas restantes hasta `PREDICTION_END_DATE`
    usando los modelos previamente entrenados.
    """

    # 1) separar histórico de la sucursal ─────────────────────────────────────
    df_hist = df[df["COD_SUC"] == branch].copy()

    # 2) determinar rango futuro personalizado ---------------------------------
    last_date = df_hist["FECHA"].max()
    if start_date is None:
        start_dt = last_date + timedelta(days=1)
    else:
        start_dt = pd.to_datetime(start_date)
        if start_dt <= last_date:
            raise ValueError("start_date debe ser posterior al último dato historico")

    if end_date is None:
        end_dt = PREDICTION_END_DATE
    else:
        end_dt = pd.to_datetime(end_date)
        if end_dt <= start_dt:
            raise ValueError("end_date debe ser posterior a start_date")

    horizon = (end_dt - start_dt).days + 1
    if horizon <= 0:
        return pd.DataFrame()

    fut_rows = [
        {"FECHA": start_dt + timedelta(days=d), "HORA": h, "COD_SUC": branch}
        for d in range(horizon)
        for h in HOURS_RANGE
    ]
    df_fut  = assign_turno(pd.DataFrame(fut_rows))
    df_comb = pd.concat([df_hist, df_fut], ignore_index=True)
    N_fut   = len(df_fut)

    df_out = df_fut.copy()

    # 3) loop por cada target ────────────────────────────────────────────────
    for target in targets:
        model_path = os.path.join(MODEL_DIR, f"predictor_{target}_{branch}.pkl")
        if not os.path.exists(model_path):
            df_out[f"{target}_pred"] = np.nan
            continue

        # 3-a) features SIN drop de hora / turno
        X_all, _ = prepare_features(
            df_comb,
            target,
            is_prediction=True,
            include_time_features=True      # ← ①
        )
        X_fut = X_all.iloc[-N_fut:].reset_index(drop=True)

        # 3-b) inferencia
        model = joblib.load(model_path)
        preds = model.predict(X_fut)
        df_out[f"{target}_pred"] = np.maximum(preds, 0)

        if target == "T_AO":
            spike_path = os.path.join(MODEL_DIR, f"spike_{branch}.pkl")
            if os.path.exists(spike_path):
                clf = joblib.load(spike_path)
                p_spike = clf.predict_proba(X_fut)[:, 1]
                df_out["p_spike"] = p_spike
                df_out["T_AO_pred"] = df_out["T_AO_pred"] * (1 + p_spike)
            else:
                df_out["p_spike"] = np.nan

    # 4) métricas y dotación ─────────────────────────────────────────────────
    df_out["T_AO_VENTA_req"] = df_out["T_AO_pred"] * efectividad_obj
    df_out["P_EFECTIVIDAD_req"] = calcular_efectividad(
        df_out["T_AO_pred"], df_out["T_AO_VENTA_req"]
    )

    # cálculo de dotación óptima (igual que antes) …
    dots, effs = [], []
    for _, r in df_out.iterrows():
        # `estimar_dotacion_optima` espera el parámetro opcional
        # `params_efectividad`. El nombre `params_sig` aquí provocaba
        # un error de tipo por argumento inesperado.
        dot, eff = estimar_dotacion_optima(
            [r["T_AO_pred"]], [r["T_AO_VENTA_req"]],
            efectividad_obj
        )
        dots.append(dot); effs.append(eff)

    df_out["DOTACION_req"]      = dots
    df_out["P_EFECTIVIDAD_opt"] = effs

    # Asegurar que todas las predicciones sean no negativas
    for col in ["T_VISITAS_pred", "T_AO_pred", "T_AO_VENTA_req"]:
        if col in df_out:
            df_out[col] = df_out[col].clip(lower=0)

    return df_out


if __name__ == '__main__':
    # Carga y preprocesamiento de datos
    print("📂 Cargando datos...")
    df = load_and_preprocess_data('data/DOTACION_EFECTIVIDAD.xlsx')
    print(f"✅ Datos cargados. {len(df)} registros de {df['COD_SUC'].nunique()} sucursales")

    # Entrenamiento de modelos
    train_all_models(df, n_iter=20)

    # Generación de predicciones de ejemplo
    example_branch = df['COD_SUC'].iloc[0]
    print(f"\n🔮 Generando predicciones de ejemplo para sucursal {example_branch}...")
    df_pred = generate_predictions(df, example_branch)

    # Mostrar resumen
    print("\n📊 Resumen de predicciones:")
    print(df_pred.groupby(df_pred['FECHA'].dt.date).agg({
        'T_VISITAS_pred': 'sum',
        'T_AO_pred': 'sum',
        'DOTACION_req': 'sum'
    }).head(10))