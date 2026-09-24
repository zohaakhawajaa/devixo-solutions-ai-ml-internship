# Diabetes Prediction Project

**Author:** Zoha Khawaja  
**Date of submission:** 27th September 2026

## Overview

This project develops a machine-learning system for predicting diabetes from patient health and lifestyle information. It includes data preparation, feature engineering, model comparison, hyperparameter tuning, model saving, project documentation, and a Streamlit prediction interface.

## Dataset

The project uses `diabetes_prediction_dataset.csv`.

The target variable is `diabetes`. The main input features are:

- Gender
- Age
- Hypertension
- Heart disease
- Smoking history
- BMI
- HbA1c level
- Blood glucose level

The dataset is checked for missing values and duplicate records. Duplicate records are removed before training.

## Workflow

1. Load and inspect the dataset.
2. Check missing values and duplicates.
3. Remove duplicate rows.
4. Create derived features including `cardio_risk` and `age_bmi`.
5. Split the data into training and testing sets.
6. Standardize numerical features.
7. One-hot encode categorical features.
8. Train and compare five classification models.
9. Tune hyperparameters with five-fold cross-validation.
10. Select the best model using F1 score.
11. Save the model together with its preprocessing transformer.
12. Run predictions through the Streamlit application.

## Models Compared

- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine
- Gradient Boosting

The main selection metric is F1 score because it balances precision and recall for the diabetes class. Accuracy, precision, recall, F1 score, and ROC-AUC are also reported.

## Result

In the current run, Gradient Boosting was selected as the best model based on the F1-score comparison. The saved artifact is stored in `best_diabetes_model.pkl` and contains both the trained model and the fitted preprocessing transformer.

## Project Files

- `Diabetes_prediction.ipynb`: Complete analysis and model development notebook.
- `diabetes_prediction_dataset.csv`: Dataset used for training and evaluation.
- `best_diabetes_model.pkl`: Saved model and preprocessing transformer.
- `app.py`: Standalone Streamlit prediction application.
- `generate_project_documentation.py`: Script used to generate the PDF report.
- `Diabetes_Project_Documentation.pdf`: Project documentation report.
- `streamlit.png`: Application screenshot.

## Run the Streamlit Application

Open PowerShell in the `Task-03` folder and run:

```powershell
python -m pip install pandas scikit-learn xgboost joblib streamlit matplotlib
streamlit run app.py
```

The application will open in a browser. Enter the patient information and select **Predict**.

## Limitations

- The dataset may not represent every population or clinical setting.
- A single train-test split can produce results that vary with the random split.
- Duplicate removal does not correct measurement errors or biased labels.
- Predictions are estimates and must not replace professional medical diagnosis.
- The current interface does not provide calibrated probabilities or individual explanations.

## Future Improvements

- Use repeated stratified cross-validation and report confidence intervals.
- Investigate class weighting or resampling for class imbalance.
- Calibrate prediction probabilities.
- Add explainability with feature importance or SHAP.
- Validate the model on an independent external dataset.
- Add input validation, model versioning, monitoring, and retraining procedures.
