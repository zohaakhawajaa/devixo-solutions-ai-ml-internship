from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


output_path = "Task_02_Report.pdf"
document = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=24, leading=30, spaceAfter=20))
styles.add(ParagraphStyle(name="CoverText", parent=styles["Normal"], alignment=TA_CENTER, fontSize=12, leading=18))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#244b5a"), spaceBefore=12, spaceAfter=6))
styles["BodyText"].leading = 15


def p(text, style="BodyText"):
    return Paragraph(text, styles[style])


story = [
    Spacer(1, 4 * cm),
    p("Bank Customer Churn Prediction", "CoverTitle"),
    p("Devixo Solutions AI/ML Internship", "CoverText"),
    Spacer(1, 2 * cm),
    p("Student Name: Zoha Khawaja", "CoverText"),
    p("Internship Domain: Artificial Intelligence and Machine Learning", "CoverText"),
    p("Task Number: Task 02", "CoverText"),
    p("Date of Submission: 20 September 2026", "CoverText"),
    PageBreak(),
    p("Objective", "Section"),
    p("The objective of this project was to develop a machine learning model that predicts whether a bank customer will churn. I also compared different classification algorithms and selected a model using validation metrics."),
    p("Problem Statement", "Section"),
    p("Customer churn can reduce a bank's revenue and increase the cost of acquiring new customers. A predictive model can help the bank identify customers who may leave and support earlier retention actions."),
    p("Technologies Used", "Section"),
    p("Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, and Jupyter Notebook."),
    p("Implementation Steps", "Section"),
    p("1. Loaded and inspected the train and test CSV files.<br/>2. Checked duplicates and missing-value percentages.<br/>3. Used Exited as the binary target.<br/>4. Applied one-hot encoding and standard scaling through a Scikit-learn pipeline.<br/>5. Trained Logistic Regression, Decision Tree, Random Forest, and XGBoost models.<br/>6. Compared classification metrics and training time.<br/>7. Tuned XGBoost with randomized cross-validation.<br/>8. Evaluated the final candidate on held-out data and displayed confusion matrices."),
    p("Dataset Summary", "Section"),
    p("The training data contains 165,034 rows and 14 columns. The test data contains 110,023 rows and 13 columns. No duplicate training rows were found. The target is imbalanced, so recall and F1-score were considered along with accuracy."),
    p("Model Comparison", "Section"),
]

data = [
    ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Time (s)"],
    ["XGBoost", "0.867", "0.749", "0.558", "0.639", "0.889", "1.302"],
    ["Decision Tree", "0.862", "0.732", "0.547", "0.626", "0.881", "0.510"],
    ["Random Forest", "0.858", "0.723", "0.535", "0.615", "0.873", "3.831"],
    ["Logistic Regression", "0.835", "0.701", "0.385", "0.497", "0.819", "0.181"],
]
table = Table(data, colWidths=[4.1 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm, 1.4 * cm, 1.8 * cm, 1.8 * cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244b5a")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aaeb5")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf3f4")]),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(table)
story.extend([
    p("The tuned XGBoost candidate reached validation accuracy of 0.867, precision of 0.753, recall of 0.555, F1-score of 0.639, and ROC-AUC of 0.890."),
    p("Code and Screenshots", "Section"),
    p("The complete source code is provided in the notebook. Its outputs include the target distribution, feature plots, correlation matrix, model comparison, and confusion matrices. These notebook outputs serve as the code and result screenshots for the submission."),
    p("Challenges Faced", "Section"),
    p("The main challenges were handling the imbalanced target, preprocessing mixed numerical and categorical columns, choosing metrics beyond accuracy, and keeping the held-out test split separate until final evaluation."),
    p("Conclusion", "Section"),
    p("This project successfully built a bank customer churn prediction workflow and compared four classification models. XGBoost gave the strongest overall validation performance. The final test ROC-AUC of approximately 0.888 shows reasonable generalisation. Future work could use threshold tuning or class weighting to improve churn recall."),
])
document.build(story)
print(output_path)