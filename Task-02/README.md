# Task 02 - Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to leave the bank. It was completed for the Devixo Solutions AI/ML Internship Program by Zoha Khawaja.

## Contents

- `notebook/Bank_Customer_Churn_Task02.ipynb` - complete analysis, preprocessing, model comparison, tuning, and evaluation.
- `dataset/train.csv` - training data with the `Exited` target column.
- `dataset/test.csv` - held-out test data used for final evaluation.
- `submission.csv` - prediction output generated for the task.
- `report/Task_02_Report.pdf` - project report.
- `report/Task_02_Report.md` - editable report source.
- `images/` - space for exported notebook figures and screenshots.

## Objective

Build and compare classification models that identify customers who may churn, with attention to recall, F1-score, and ROC-AUC.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## How to Run

1. Open the notebook from the `notebook` directory in Jupyter or VS Code.
2. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. Run the cells from top to bottom. The notebook reads the datasets from `../dataset/`.

## Main Result

The standard XGBoost model achieved the highest validation F1-score (0.639) and validation ROC-AUC (0.889) among the compared models. The tuned XGBoost candidate had a slightly higher validation ROC-AUC (0.890), while the final held-out test ROC-AUC was 0.888.

## Notes

The data contains customer attributes such as credit score, geography, age, balance, number of products, and activity status. The `Surname` and identifier columns are excluded from model features. The notebook checks duplicates and missing values before training.