# Bank Customer Churn Prediction

## Title Page

**Student Name:** Zoha Khawaja  
**Internship Domain:** Artificial Intelligence and Machine Learning  
**Task Number:** Task 02  
**Date of Submission:** 20 September 2026  

## Objective

The objective of this project was to develop a machine learning model that predicts whether a bank customer will churn. I also compared different classification algorithms and selected a model using validation metrics.

## Problem Statement

Customer churn can reduce a bank's revenue and increase the cost of acquiring new customers. A predictive model can help the bank identify customers who may leave and support earlier retention actions.

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, and Jupyter Notebook.

## Implementation Steps

1. Loaded the train and test CSV files and inspected their structure.
2. Checked duplicate records and missing-value percentages.
3. Used `Exited` as the binary target variable.
4. Removed identifier-style columns from the model features.
5. Split the training data into training and validation sets.
6. Applied one-hot encoding to categorical features and standard scaling to numerical features through a Scikit-learn pipeline.
7. Trained Logistic Regression, Decision Tree, Random Forest, and XGBoost models.
8. Compared accuracy, precision, recall, F1-score, ROC-AUC, and training time.
9. Tuned XGBoost with randomized cross-validation.
10. Evaluated the final tuned candidate on the held-out test split and displayed a confusion matrix.

## Dataset Summary

The training data has 165,034 rows and 14 columns. The test data has 110,023 rows and 13 columns because the test file does not include the target column. The training data contained no duplicate rows. The target is imbalanced, so recall and F1-score were considered along with accuracy.

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Training time (seconds) |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.867 | 0.749 | 0.558 | 0.639 | 0.889 | 1.302 |
| Decision Tree | 0.862 | 0.732 | 0.547 | 0.626 | 0.881 | 0.510 |
| Random Forest | 0.858 | 0.723 | 0.535 | 0.615 | 0.873 | 3.831 |
| Logistic Regression | 0.835 | 0.701 | 0.385 | 0.497 | 0.819 | 0.181 |

The tuned XGBoost model reached validation accuracy of 0.867, precision of 0.753, recall of 0.555, F1-score of 0.639, and ROC-AUC of 0.890. The final held-out test ROC-AUC was 0.888.

## Code and Screenshots

The complete source code is in `notebook/Bank_Customer_Churn_Task02.ipynb`. The notebook includes screenshots generated from the analysis cells, including the target distribution, numerical feature plots, correlation matrix, model comparison, and confusion matrices. These outputs provide a visual explanation of the implementation and results.

## Output and Result Explanation

The standard XGBoost model produced the best validation F1-score, so it was selected as the best validation model. The tuned XGBoost model was very close and produced the strongest validation ROC-AUC. On the held-out test data, its ROC-AUC remained close to the validation result, which suggests that the model generalised reasonably well.

## Challenges Faced

- The target classes were not balanced, so accuracy alone was not enough to compare the models.
- The dataset included both numerical and categorical columns, requiring separate preprocessing steps.
- XGBoost produced good results but required more parameters and took longer to train than Logistic Regression.
- It was important to keep the test split separate until the final evaluation to reduce data leakage.

## Conclusion

This project successfully built a bank customer churn prediction workflow and compared four classification models. XGBoost gave the strongest overall validation performance, especially for F1-score and ROC-AUC. The final test ROC-AUC of approximately 0.888 shows that the selected approach can distinguish likely churners from non-churners reasonably well. In a future version, I would investigate class-weighting or threshold tuning to improve churn recall further.