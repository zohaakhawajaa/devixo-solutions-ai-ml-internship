import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Used Car Price Prediction — Professional Streamlit Dashboard
# ============================================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "used_car_price_model.joblib"
DATA_PATH = APP_DIR / "vehicles.csv"
ARTIFACTS_DIR = APP_DIR / "artifacts"
# Notebook-generated artifacts currently live in the project-level folder.
# Prefer app-local artifact files, then use the project-level artifacts folder.
artifact_filenames = (
    "reference_year.csv",
    "comparison.csv",
    "results.csv",
    "feature_importance.csv",
)
if not any((ARTIFACTS_DIR / name).is_file() for name in artifact_filenames):
    ARTIFACTS_DIR = APP_DIR.parent / "artifacts"

REFERENCE_YEAR_PATH = ARTIFACTS_DIR / "reference_year.csv"
COMPARISON_PATH = ARTIFACTS_DIR / "comparison.csv"
RESULTS_PATH = ARTIFACTS_DIR / "results.csv"
IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"


# ============================================================
# Custom UI
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 0;
    }

    .section-card {
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .prediction-card {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border: 1px solid #bbf7d0;
        text-align: center;
        margin-top: 1.5rem;
    }

    .prediction-label {
        font-size: 1rem;
        color: #166534;
        margin-bottom: 0.3rem;
    }

    .prediction-price {
        font-size: 2.8rem;
        font-weight: 800;
        color: #14532d;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 14px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model at '{MODEL_PATH}': {e}")
    st.stop()


# ============================================================
# Load dataset
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()
except Exception:
    df = None


# ============================================================
# Reference year
# ============================================================

@st.cache_data
def load_reference_year():
    if REFERENCE_YEAR_PATH.exists():
        try:
            ref = pd.read_csv(REFERENCE_YEAR_PATH)
            return int(ref["reference_year"].iloc[0])
        except Exception:
            pass
    return 2026


REFERENCE_YEAR = load_reference_year()


# ============================================================
# Helper functions
# ============================================================

def options(column, fallback):
    if df is not None and column in df.columns:
        values = (
            df[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(100)
            .index
            .tolist()
        )
        return values if values else fallback
    return fallback


def money(value):
    return f"${value:,.0f}"


def load_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🚗 CarPrice AI")
    st.caption("Used Vehicle Price Prediction")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home / Prediction",
            "📊 Model Performance",
            "🔍 Feature Importance",
            "📈 Data Visualization"
        ]
    )

    st.divider()

    st.markdown("### About")
    st.caption(
        "A machine learning application for estimating used vehicle prices "
        "using a trained Random Forest regression pipeline."
    )

    st.caption(f"Model reference year: {REFERENCE_YEAR}")


# ============================================================
# 1. HOME / PREDICTION
# ============================================================

if page == "🏠 Home / Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>🚗 Used Car Price Predictor</h1>
            <p>
                Enter vehicle details below and let the trained machine
                learning model estimate its market price.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    manufacturers = options(
        "manufacturer",
        ["ford", "chevrolet", "toyota", "honda", "bmw", "mercedes-benz"]
    )

    conditions = options(
        "condition",
        ["excellent", "good", "like new", "fair", "new", "salvage"]
    )

    cylinders_options = options(
        "cylinders",
        ["4 cylinders", "6 cylinders", "8 cylinders"]
    )

    fuels = options(
        "fuel",
        ["gas", "diesel", "hybrid", "electric", "other"]
    )

    title_statuses = options(
        "title_status",
        ["clean", "rebuilt", "salvage", "lien", "missing", "parts only"]
    )

    transmissions = options(
        "transmission",
        ["automatic", "manual", "other"]
    )

    drives = options(
        "drive",
        ["4wd", "fwd", "rwd"]
    )

    sizes = options(
        "size",
        ["full-size", "mid-size", "compact", "sub-compact"]
    )

    types = options(
        "type",
        ["sedan", "SUV", "truck", "coupe", "hatchback", "wagon", "van"]
    )

    paint_colors = options(
        "paint_color",
        ["black", "white", "silver", "red", "blue", "grey"]
    )

    states = options(
        "state",
        ["ca", "tx", "fl", "ny", "pa", "az"]
    )

    st.markdown("### 🚘 Vehicle Details")

    col1, col2 = st.columns(2)

    with col1:
        manufacturer = st.selectbox("Manufacturer", manufacturers)
        model_name = st.text_input(
            "Vehicle Model",
            placeholder="e.g. camry, f-150, civic"
        )

        year = st.number_input(
            "Manufacturing Year",
            min_value=1980,
            max_value=REFERENCE_YEAR,
            value=2018,
            step=1
        )

        odometer = st.number_input(
            "Mileage / Odometer",
            min_value=0.0,
            max_value=1_000_000.0,
            value=80_000.0,
            step=1_000.0
        )

        condition = st.selectbox("Condition", conditions)
        cylinders = st.selectbox("Cylinders", cylinders_options)

    with col2:
        fuel = st.selectbox("Fuel Type", fuels)
        title_status = st.selectbox("Title Status", title_statuses)
        transmission = st.selectbox("Transmission", transmissions)
        drive = st.selectbox("Drive", drives)
        size = st.selectbox("Vehicle Size", sizes)
        vehicle_type = st.selectbox("Vehicle Type", types)
        paint_color = st.selectbox("Paint Color", paint_colors)
        state = st.selectbox("State", states)

    st.divider()

    predict_clicked = st.button(
        "🔮 Predict Vehicle Price",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:

        if not model_name.strip():
            st.warning("Please enter the vehicle model.")
            st.stop()

        vehicle_age = max(
            REFERENCE_YEAR - int(year),
            0
        )

        input_data = pd.DataFrame([{
            "manufacturer": manufacturer,
            "model": model_name.strip(),
            "condition": condition,
            "cylinders": cylinders,
            "fuel": fuel,
            "odometer": odometer,
            "title_status": title_status,
            "transmission": transmission,
            "drive": drive,
            "size": size,
            "type": vehicle_type,
            "paint_color": paint_color,
            "state": state,
            "vehicle_age": vehicle_age
        }])

        try:
            prediction = float(model.predict(input_data)[0])

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Estimated Market Price
                    </div>
                    <div class="prediction-price">
                        {money(prediction)}
                    </div>
                    <div class="small-note">
                        Estimated using the trained Random Forest regression model
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📋 Vehicle Summary")

            s1, s2, s3, s4 = st.columns(4)

            s1.metric("Manufacturer", manufacturer.title())
            s2.metric("Model", model_name.strip().title())
            s3.metric("Year", int(year))
            s4.metric("Mileage", f"{odometer:,.0f}")

            st.info(
                "This is a machine-learning estimate and should be treated "
                "as an estimated value rather than a guaranteed market price."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ============================================================
# 2. MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Model Performance</h1>
            <p>
                Evaluation metrics and model comparison from the training pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    comparison = load_csv(COMPARISON_PATH)
    results = load_csv(RESULTS_PATH)

    if comparison is not None and len(comparison) > 0:

        st.subheader("Before vs After Tuning")

        tuned_row = comparison[
            comparison["Model"].astype(str).str.contains(
                r"Tuned|After Tuning",
                case=False,
                regex=True,
                na=False
            )
        ]

        if len(tuned_row) > 0:

            row = tuned_row.iloc[0]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("MAE", f"{row['MAE']:,.2f}")
            c2.metric("MSE", f"{row['MSE']:,.2f}")
            c3.metric("RMSE", f"{row['RMSE']:,.2f}")
            c4.metric("R²", f"{row['R2']:.4f}")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "Performance metrics have not been exported yet. "
            "Run the notebook artifact-saving cell."
        )

    if results is not None and len(results) > 0:

        st.subheader("Model Comparison")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        chart_data = results.set_index("Model")["RMSE"]

        st.bar_chart(chart_data)

        st.caption(
            "Lower RMSE indicates smaller prediction errors on the evaluation set."
        )


# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.markdown(
        """
        <div class="hero">
            <h1>🔍 Feature Importance</h1>
            <p>
                Features contributing to predictions made by the tuned
                Random Forest model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    importance_df = load_csv(IMPORTANCE_PATH)

    if importance_df is not None and len(importance_df) > 0:

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=30,
            value=20
        )

        top = (
            importance_df
            .head(top_n)
            .sort_values("Importance")
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.barh(
            top["Feature"],
            top["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_n} Feature Importances")

        plt.tight_layout()

        st.pyplot(fig)

        st.subheader("Feature Importance Table")

        st.dataframe(
            importance_df.head(top_n),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "Feature importance data has not been exported yet. "
            "Run the notebook artifact-saving cell."
        )


# ============================================================
# 4. DATA VISUALIZATION
# ============================================================

elif page == "📈 Data Visualization":

    st.markdown(
        """
        <div class="hero">
            <h1>📈 Data Visualization</h1>
            <p>
                Explore relationships and patterns in the used vehicle dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if df is None:
        st.error(
            "vehicles.csv was not found. Put the dataset in the same "
            "folder as app.py."
        )
        st.stop()

    plot_df = df.copy()
    plot_df = plot_df[plot_df["price"] > 0].copy()

    price_upper = plot_df["price"].quantile(0.99)

    chart_df = plot_df[
        plot_df["price"] <= price_upper
    ].copy()

    # --------------------------------------------------------
    # Price Distribution
    # --------------------------------------------------------

    st.subheader("💰 Price Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        chart_df["price"],
        bins=50
    )

    ax.set_xlabel("Price")
    ax.set_ylabel("Number of Vehicles")
    ax.set_title("Used Car Price Distribution")

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Mileage vs Price
    # --------------------------------------------------------

    st.subheader("🛣️ Mileage vs Price")

    mileage_df = chart_df.dropna(
        subset=["odometer", "price"]
    )

    if len(mileage_df) > 10_000:
        mileage_df = mileage_df.sample(
            10_000,
            random_state=42
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        mileage_df["odometer"],
        mileage_df["price"],
        alpha=0.35
    )

    ax.set_xlabel("Mileage / Odometer")
    ax.set_ylabel("Price")
    ax.set_title("Mileage vs Price")

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Year vs Price
    # --------------------------------------------------------

    st.subheader("📅 Vehicle Year vs Price")

    year_df = chart_df.dropna(
        subset=["year", "price"]
    )

    if len(year_df) > 10_000:
        year_df = year_df.sample(
            10_000,
            random_state=42
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        year_df["year"],
        year_df["price"],
        alpha=0.35
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    ax.set_title("Vehicle Year vs Price")

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Average Price by Fuel
    # --------------------------------------------------------

    st.subheader("⛽ Average Price by Fuel Type")

    if "fuel" in chart_df.columns:

        fuel_price = (
            chart_df
            .dropna(subset=["fuel"])
            .groupby("fuel")["price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(fuel_price)

    # --------------------------------------------------------
    # Average Price by Condition
    # --------------------------------------------------------

    st.subheader("✨ Average Price by Condition")

    if "condition" in chart_df.columns:

        condition_price = (
            chart_df
            .dropna(subset=["condition"])
            .groupby("condition")["price"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(condition_price)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Used Car Price Prediction • Machine Learning Regression Application"
)
