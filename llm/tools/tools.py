import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..data_context import DATA_CONTEXT


# Data load tool
def load_data(path: str) -> dict:
    ext = Path(path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        fmt = "csv"
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
        fmt = "parquet"
    elif ext in [".tsv"]:
        df = pd.read_csv(path, sep="\t")
        fmt = "tsv"
    elif ext in [".xlsx"]:
        df = pd.read_excel(path)
        fmt = "excel"
    elif ext in [".json", ".jsonl"]:
        df = pd.read_json(path, lines=ext == ".jsonl")
        fmt = "json"
    elif ext in [".pkl"]:
        df = pd.read_pickle(path)
        fmt = "pickle"
    else:
        raise ValueError(f"Unsupported format: {ext}")

    DATA_CONTEXT.df = df
    DATA_CONTEXT.source = path
    DATA_CONTEXT.format = fmt

    return {
        "status": "ok",
        "format": fmt,
        "rows": len(df),
        "columns": len(df.columns)
    }


# Dataset head tool
def dataset_head(n: int = 5) -> list[dict]:
    if not DATA_CONTEXT.is_loaded():
        raise RuntimeError("No dataset loaded")

    return DATA_CONTEXT.df.head(n).to_dict(orient="records")


# Dataset info tool
def dataset_info(max_top_values: int = 5) -> dict:
    if not DATA_CONTEXT.is_loaded():
        raise RuntimeError("No dataset loaded")

    df = DATA_CONTEXT.df

    n_rows = len(df)
    n_cols = len(df.columns)
    missed_values = df.isna().sum().sum()
    missing_pct = round((missed_values / (n_rows * n_cols)) * 100, 2)
    columns_info = {}

    for col in df.columns:
        s = df[col]

        n_missing = int(s.isna().sum())
        col_missing_pct = round(n_missing / n_rows * 100, 2)
        n_unique = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            semantic = "numeric"
            top_values = None
        else:
            semantic = "categorical"
            top_values = (
                s.value_counts(dropna=True)
                 .head(max_top_values)
                 .to_dict()
            )

        columns_info[col] = {
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "col_missing_pct": col_missing_pct,
            "n_unique": n_unique,
            "semantic_type": semantic,
            **({"top_values": top_values} if top_values else {})
        }

    return {
        "rows": n_rows,
        "n_columns": len(df.columns),
        "missing_pct": missing_pct,
        "columns": columns_info
    }


# Correlation matrix and correlation heatmap tools
def correlation_matrix(
    threshold: float = 0.2,
    label: str = None
) -> dict:

    if not DATA_CONTEXT.is_loaded():
        raise RuntimeError("No dataset loaded")

    df = DATA_CONTEXT.df
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()

    pairs = []
    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i+1:]:
            value = corr.loc[col1, col2]
            if abs(value) >= threshold:
                pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(float(value), 3)
                })

    if label and label in df.columns and pd.api.types.is_numeric_dtype(df[label]):
        feature_target_corr = (
            df.select_dtypes(include="number")
              .drop(columns=[label], errors="ignore")
              .corrwith(df[label])
              .abs()
              .sort_values(ascending=False)
              .round(3)
              .to_dict()
        )

        return {
            "threshold": threshold,
            "matrix": corr.round(3).to_dict(),
            "feature_target_abs_corr": feature_target_corr,
            "high_correlation_pairs": pairs
        }
    
    else:
        return {
            "threshold": threshold,
            "matrix": corr.round(3).to_dict(),
            "high_correlation_pairs": pairs
        }


def plot_correlation_heatmap(dataset_name: str) -> dict:
    if not DATA_CONTEXT.is_loaded():
        raise RuntimeError("No dataset loaded")

    df = DATA_CONTEXT.df
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()

    Path("plots").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
    )
    plt.title(f"Correlation heatmap ({dataset_name})")

    path = f"plots/correlation_heatmap{dataset_name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return {
        "type": "heatmap",
        "path": path
    }


TOOLS = {
        "load_data": load_data,
        "dataset_head": dataset_head,
        "dataset_info": dataset_info,
        "correlation_matrix": correlation_matrix,
        "plot_correlation_heatmap": plot_correlation_heatmap
}

