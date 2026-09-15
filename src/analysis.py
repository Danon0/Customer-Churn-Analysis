"""Customer Churn Analysis - reusable analysis pipeline."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn.csv"

def load_data(path=DATA):
    return pd.read_csv(path)

def clean_data(df):
    out = df.copy()
    out = out.drop_duplicates()
    out["age"] = out["age"].fillna(out["age"].median())
    out["monthly_charges"] = out["monthly_charges"].fillna(out["monthly_charges"].median())
    out["total_charges"] = out["total_charges"].fillna(
        out["monthly_charges"] * out["tenure_months"]
    )
    out["payment_method"] = out["payment_method"].fillna("Unknown")
    return out

def add_features(df):
    out = df.copy()
    out["churn_flag"] = (out["churn"] == "Yes").astype(int)
    out["annualized_charges"] = out["monthly_charges"] * 12
    return out

def churn_rate_by(df, column):
    return (df.groupby(column)["churn_flag"]
              .agg(churn_rate="mean", customers="size")
              .assign(churn_rate=lambda x: (x["churn_rate"]*100).round(2))
              .sort_values("churn_rate", ascending=False))

if __name__ == "__main__":
    data = add_features(clean_data(load_data()))
    print(data.shape)
    print(churn_rate_by(data, "contract"))
