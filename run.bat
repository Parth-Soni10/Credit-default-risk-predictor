@echo off
REM One-click setup + launch (Windows). Usage: double-click or run.bat
cd /d "%~dp0"
if not exist venv python -m venv venv
call venv\Scripts\activate
pip install -q -r requirements.txt

dir /b data\*.csv data\*.xls data\*.xlsx >nul 2>&1
if errorlevel 1 (
  echo No dataset in data\ - generating SYNTHETIC demo data. Put UCI_Credit_Card.csv in data\ for real results.
  python -m src.generate_sample_data
)
if not exist models\logistic_regression.joblib (
  python -m src.eda
  python -m src.train
)
streamlit run app.py
