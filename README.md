# Credit Risk Prediction API

A FastAPI service that serves a random forest credit risk model. It is the backend for the [Streamlit credit risk app](https://github.com/naufalarkan2004-spec/Streamlit).

Live: https://ml-deployment34.vercel.app

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/get_prediction` | Predict risk for a list of applicants |

### Request

A JSON list of records, each with these 17 fields:

`CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, DAYS_BIRTH, DAYS_EMPLOYED, FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL, OCCUPATION_TYPE, CNT_FAM_MEMBERS`

### Response

```json
{ "class": [0], "prob": [0.034] }
```

`prob` is the probability of the risky class. `class` uses a 0.5 cut-off; because the data is imbalanced, clients usually apply their own threshold to `prob`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Interactive docs are then available at http://127.0.0.1:8000/docs.

## Files

`app.py` (API), `model_rf_v2.sav` (trained model), `requirements.txt`.

## Tech stack

Python, FastAPI, scikit-learn, pandas, imbalanced-learn, Vercel.
