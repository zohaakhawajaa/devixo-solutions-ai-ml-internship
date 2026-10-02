# Used Car Price Prediction

A Streamlit dashboard that estimates a used vehicle's price with a trained Random Forest regression pipeline. The project includes the trained model, evaluation tables, feature-importance data, and the notebook/report used to document the work.

## Dashboard

- **Home / Prediction** — enter vehicle details and get an estimated price.
- **Model Performance** — review evaluation metrics and model comparisons.
- **Feature Importance** — explore the trained model's most influential features.
- **Data Visualization** — inspect price, mileage, year, fuel, and condition patterns.

## Run locally

Use Python 3.10 or newer. From this directory, create an environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

On macOS or Linux, activate the environment with `source .venv/bin/activate` after creating it with `python3 -m venv .venv`.

## Dataset

The raw `vehicles.csv` dataset is intentionally excluded from Git because it is about 1.45 GB. The prediction page, model-performance page, and feature-importance page use the committed model and artifacts. To enable the data-visualization page and populate dropdowns from the source data, place the dataset at `vehicles.csv` in this directory.

The training notebook expects the dataset at that same path. Run Jupyter with this project directory as its working directory before opening `notebooks/used_car_price_prediction.ipynb`.

## Project files

```text
used-car-price-app/
├── app.py
├── requirements.txt
├── artifacts/
│   ├── comparison.csv
│   ├── feature_importance.csv
│   ├── reference_year.csv
│   └── results.csv
├── models/
│   └── used_car_price_model.joblib
├── notebooks/
│   └── used_car_price_prediction.ipynb
└── reports/
    └── used_car_price_prediction_report.pdf
```

## Evaluation snapshot

Metrics below come from the saved held-out evaluation results. Lower MAE and RMSE indicate smaller errors; higher R² indicates more variance explained.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 4,226.73 | 6,610.67 | 0.7344 |
| Tuned Random Forest | 4,378.88 | 6,740.34 | 0.7239 |

The saved tuning run did not improve on the baseline Random Forest, so the comparison is shown as measured rather than presented as a gain.

## Notes

- Predictions are estimates and are not guaranteed market valuations.
- The model is serialized with Joblib and depends on compatible Python and scikit-learn versions. If loading it raises a version-compatibility error, retrain and save the model using the environment that will run the app.
- The notebook and report are included for reproducibility and review; the Streamlit app does not need Jupyter to run.
