import pandas as pd


def assign_turno(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a simple shift id based on the hour of day."""
    bins = [8, 11, 14, 17, 21]
    labels = [1, 2, 3, 4]
    df = df.copy()
    df["turno"] = (
        pd.cut(
            df["HORA"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        .cat.add_categories([0])
        .fillna(0)
        .astype(int)
    )
    return df


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing used for training and inference."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["weekday"] = df["FECHA"].dt.weekday
    df["month"] = df["FECHA"].dt.month
    df = assign_turno(df)
    df["COD_SUC"] = df["COD_SUC"].astype("category").cat.codes
    return df


def prepare_features(df: pd.DataFrame, target: str):
    """Return feature matrix X and target y for the given variable."""
    df_proc = basic_preprocess(df)
    features = ["HORA", "weekday", "month", "turno", "COD_SUC"]
    X = df_proc[features].fillna(0)
    y = df_proc[target].fillna(0)
    return X, y
