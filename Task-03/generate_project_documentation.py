from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "diabetes_prediction_dataset.csv"
MODEL_PATH = PROJECT_DIR / "best_diabetes_model.pkl"
PDF_PATH = PROJECT_DIR / "Diabetes_Project_Documentation.pdf"


def add_page(pdf, title, sections):
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.95, title, fontsize=20, weight="bold", color="#16324F")
    y_position = 0.89

    for heading, paragraphs in sections:
        figure.text(0.08, y_position, heading, fontsize=13, weight="bold", color="#2A6F97")
        y_position -= 0.035

        for paragraph in paragraphs:
            lines = []
            words = paragraph.split()
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if len(candidate) > 92:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            if line:
                lines.append(line)

            for line in lines:
                figure.text(0.08, y_position, line, fontsize=10, va="top")
                y_position -= 0.023
            y_position -= 0.012

    figure.text(0.08, 0.035, "Diabetes Prediction Project", fontsize=8, color="gray")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def evaluate_saved_model():
    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    data["cardio_risk"] = data["hypertension"] + data["heart_disease"]
    data["age_bmi"] = data["age"] * data["bmi"]

    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    X_test_processed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    return model.__class__.__name__, {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }


def main():
    model_name, metrics = evaluate_saved_model()
    metric_text = [f"{name}: {value:.4f}" for name, value in metrics.items()]

    with PdfPages(PDF_PATH) as pdf:
        add_page(
            pdf,
            "Diabetes Prediction Project",
            [
                (
                    "Submission Details",
                    [
                        "Name: Zoha Khawaja",
                        "Date of submission: 27th September 2026",
                    ],
                ),
                (
                    "Project Overview",
                    [
                        "This project develops a machine-learning classifier for predicting diabetes from patient health and lifestyle information.",
                        "The final model is stored with its fitted preprocessing transformer and is used by the Streamlit prediction interface.",
                    ],
                ),
                (
                    "Dataset",
                    [
                        "The dataset is diabetes_prediction_dataset.csv. The target column is diabetes.",
                        "Input variables include gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, and blood glucose level.",
                        "The dataset was checked for missing values and duplicate rows. Duplicate records were removed before modeling.",
                    ],
                ),
            ],
        )

        add_page(
            pdf,
            "Dataset Workflow",
            [
                (
                    "Preparation",
                    [
                        "The data is loaded with pandas, duplicate rows are removed, and the index is reset.",
                        "Two derived variables are created: cardio_risk, based on hypertension and heart disease, and age_bmi, based on age multiplied by BMI.",
                    ],
                ),
                (
                    "Preprocessing",
                    [
                        "The data is split into training and testing subsets using an 80/20 split with random_state=42.",
                        "Numerical features are standardized with StandardScaler. Categorical features are converted using OneHotEncoder with unknown categories ignored.",
                        "A ColumnTransformer keeps preprocessing consistent during training and prediction.",
                    ],
                ),
                (
                    "Training Workflow",
                    [
                        "Five models are trained and evaluated. GridSearchCV uses five-fold cross-validation and F1 scoring for hyperparameter tuning.",
                        "The selected model and the fitted preprocessor are saved together in best_diabetes_model.pkl.",
                    ],
                ),
            ],
        )

        add_page(
            pdf,
            "Model Selection and Results",
            [
                (
                    "Models Compared",
                    [
                        "Decision Tree, Random Forest, XGBoost, Support Vector Machine, and Gradient Boosting were compared.",
                        "F1 score was used as the primary selection metric because it balances precision and recall for the positive diabetes class.",
                    ],
                ),
                (
                    "Selected Model",
                    [
                        f"The saved model is {model_name}. It was selected as the best model according to the F1-score comparison.",
                    ],
                ),
                (
                    "Evaluation Results",
                    ["The saved model was evaluated on the held-out test set:", *metric_text],
                ),
            ],
        )

        add_page(
            pdf,
            "Limitations and Future Improvements",
            [
                (
                    "Limitations",
                    [
                        "The dataset may not represent every population or clinical setting.",
                        "A single train-test split can produce results that vary with the selected random split.",
                        "Duplicate removal does not correct measurement errors, missing context, or biased labels.",
                        "The prediction is a machine-learning estimate and must not replace professional medical diagnosis.",
                        "The current interface does not provide calibrated risk probabilities or explanations for individual predictions.",
                    ],
                ),
                (
                    "Future Improvements",
                    [
                        "Use repeated stratified cross-validation and report confidence intervals.",
                        "Investigate class weighting or resampling if class imbalance affects recall.",
                        "Calibrate probability estimates and add explainability with feature importance or SHAP.",
                        "Validate the model on an independent external dataset.",
                        "Add stronger input validation, model versioning, monitoring, and retraining procedures.",
                    ],
                ),
            ],
        )

    print(f"Created: {PDF_PATH}")


if __name__ == "__main__":
    main()