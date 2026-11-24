"""Data loading, cleaning and shared constants."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "DEFAULT"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS

SEX_MAP = {1: "Male", 2: "Female"}
EDU_MAP = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_MAP = {1: "Married", 2: "Single", 3: "Other"}

CANDIDATE_NAMES = [
    "UCI_Credit_Card.csv",
    "default of credit card clients.xls",
    "default_of_credit_card_clients.xls",
    "default of credit card clients.csv",
    "credit_card_default.csv",
    "sample_credit_data.csv",
]


def find_dataset(data_dir: Path = DATA_DIR):
    """Return the first dataset file found in data/, or None."""
    for name in CANDIDATE_NAMES:
        p = data_dir / name
        if p.exists():
            return p
    for pattern in ("*.csv", "*.xls", "*.xlsx"):
        files = sorted(data_dir.glob(pattern))
        if files:
            return files[0]
    return None


def _read_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    is_excel = path.suffix.lower() in (".xls", ".xlsx")

    def read(header):
        return pd.read_excel(path, header=header) if is_excel else pd.read_csv(path, header=header)

    df = read(0)
    # The original UCI .xls has a throw-away "X1, X2, ..." row above the real header.
    if "LIMIT_BAL" not in [str(c).strip() for c in df.columns]:
        df = read(1)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c.lower().replace(".", " ").replace("_", " ") == "default payment next month":
            rename[c] = TARGET
    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        rename["PAY_1"] = "PAY_0"
    df = df.rename(columns=rename)
    if "ID" in df.columns:
        df = df.drop(columns="ID")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented category codes in the UCI dataset and drop duplicates."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df.drop_duplicates().reset_index(drop=True)


def load_raw(path=None) -> pd.DataFrame:
    """Dataset exactly as read from disk (column names standardised, ID dropped, NOT cleaned)."""
    path = Path(path) if path else find_dataset()
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "No dataset found in the data/ folder.\n"
            "Download 'UCI_Credit_Card.csv' (Default of Credit Card Clients) and place it in data/,\n"
            "or run `python -m src.generate_sample_data` to create a synthetic demo file."
        )
    df = standardize_columns(_read_any(path))
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df[FEATURES + [TARGET]]


def load_dataset(path=None) -> pd.DataFrame:
    return clean(load_raw(path))
