
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "telco_churn_cleaned.csv"


def preprocess_data():
    print("Starting preprocessing...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    print("Original dataset shape:", df.shape)

    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if "Churn" not in df.columns:
        raise ValueError("Churn column is missing.")

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Preprocessing completed successfully!")
    print("Cleaned dataset shape:", df.shape)
    print("Saved file:", OUTPUT_FILE)


if __name__ == "__main__":
    preprocess_data()