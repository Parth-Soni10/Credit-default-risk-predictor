# 💳 Credit Card Default Risk Predictor

Classical supervised ML (**Logistic Regression** and **Decision Tree**) that predicts whether a credit card
customer will default next month, using the *Default of Credit Card Clients* dataset (UCI / Kaggle).
Includes EDA, feature scaling, grid-searched models, evaluation with **Precision / Recall / F1 / ROC-AUC**,
and a Streamlit web app that runs on **localhost**.

## Project structure
```
credit-default-risk-predictor/
├── app.py                    # Streamlit web app (localhost:8501)
├── requirements.txt
├── run.sh / run.bat          # one-click setup, train and launch
├── data/                     # put the dataset here
├── models/                   # trained models (.joblib) - created by training
├── reports/                  # metrics.json + evaluation charts + eda/ charts
└── src/
    ├── data_utils.py         # loading, column normalisation, cleaning
    ├── eda.py                # exploratory data analysis charts
    ├── train.py              # preprocessing, training, tuning, evaluation
    └── generate_sample_data.py   # SYNTHETIC data for smoke-testing only
```

## 1. Get the dataset
Download **UCI_Credit_Card.csv** from either source and place it in `data/`:
- Kaggle: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- UCI: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (the `.xls` file also works)

The loader auto-detects the file and handles both naming styles (`PAY_0`/`PAY_1`,
`default.payment.next.month` / `default payment next month`).

> No dataset yet? `run.sh`/`run.bat` generate a **synthetic** file so you can try the app.
> Metrics from synthetic data are not meaningful.

## 2. Run on localhost

**Quick way** - macOS/Linux: `./run.sh`  |  Windows: `run.bat`

**Manual way**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.eda               # EDA charts  -> reports/eda/
python -m src.train             # train + evaluate -> models/, reports/
streamlit run app.py            # open http://localhost:8501
```
Python 3.9+ recommended.

### Training options
```bash
python -m src.train --data data/UCI_Credit_Card.csv --test-size 0.2 --seed 42
python -m src.train --no-balance      # disable class_weight='balanced'
```

## What the pipeline does
1. **Cleaning** - drops `ID`; maps undocumented codes (EDUCATION 0/5/6 → Other, MARRIAGE 0 → Other); removes duplicates.
2. **Preprocessing** - `StandardScaler` on numeric features (LIMIT_BAL, AGE, PAY_*, BILL_AMT*, PAY_AMT*),
   one-hot encoding for SEX / EDUCATION / MARRIAGE, all inside an sklearn `Pipeline` (no train/test leakage).
3. **Split** - stratified 80/20 train/test.
4. **Models** - Logistic Regression (tuned `C`) and Decision Tree (tuned `max_depth`, `min_samples_leaf`),
   5-fold stratified CV grid search optimising **F1**, with `class_weight='balanced'` because only ~22% of
   customers default.
5. **Evaluation** - accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves,
   LR coefficients, tree feature importances and a tree diagram.

## Using the app
- **Predict** - enter a customer profile; see each model's default probability, risk band and decision.
  The sidebar threshold slider trades precision for recall (lower threshold = fewer missed defaults, more false alarms).
- **Model performance** - metrics table and charts.
- **EDA** - all exploratory charts.

## Typical results on the real dataset
Roughly: accuracy ≈ 0.68-0.78, ROC-AUC ≈ 0.72-0.77, F1 (default class) ≈ 0.45-0.53. `PAY_0`
(most recent repayment status) is by far the strongest predictor.

## Troubleshooting
- `No dataset found` → place the CSV/XLS in `data/`.
- `No trained models found` in the app → run `python -m src.train`.
- Port busy → `streamlit run app.py --server.port 8502`.
- `.xls` read error → `pip install xlrd openpyxl`.

*Educational project - not for production credit decisions.*
