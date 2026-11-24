#!/usr/bin/env bash
# One-click setup + launch (macOS / Linux). Usage: ./run.sh
set -e
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

if ! ls data/*.csv data/*.xls data/*.xlsx >/dev/null 2>&1; then
  echo "No dataset in data/ - generating SYNTHETIC demo data (put UCI_Credit_Card.csv in data/ for real results)."
  python -m src.generate_sample_data
fi
[ -f models/logistic_regression.joblib ] || { python -m src.eda; python -m src.train; }
streamlit run app.py
