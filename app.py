"""
Automobile Fuel Efficiency (MPG) Predictor
==========================================
End-to-end ML application built with Streamlit.

Pipeline
--------
1. Data extraction  : load auto-mpg.csv, convert '?' placeholders to NaN
2. Preprocessing    : (inside a scikit-learn Pipeline / ColumnTransformer)
                      - missing value treatment  -> SimpleImputer (median / most frequent)
                      - feature scaling          -> StandardScaler
                      - categorical encoding     -> OneHotEncoder (origin)
3. Modelling        : Linear Regression, Ridge, Random Forest, Gradient Boosting
                      compared with 5-fold CV + hold-out test set; best model is deployed
4. Application      : user enters vehicle specs -> predicted MPG
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "auto-mpg.csv"
RANDOM_STATE = 42
TARGET = "mpg"

NUMERIC_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]
CATEGORICAL_FEATURES = ["origin"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}
ORIGIN_REVERSE = {v: k for k, v in ORIGIN_MAP.items()}


# --------------------------------------------------------------------------
# 1. Data extraction
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and clean column names / placeholder values."""
    df = pd.read_csv(path)
    df = df.rename(columns={"model year": "model_year", "car name": "car_name"})

    # 'horsepower' contains '?' for unknown values -> convert to real NaN
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

    # Map numeric origin codes to readable categories (encoded later in the pipeline)
    df["origin"] = df["origin"].map(ORIGIN_MAP)
    return df


# --------------------------------------------------------------------------
# 2. Preprocessing pipeline
# --------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numeric columns, imputation + one-hot for categorical."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),   # missing value treatment
            ("scaler", StandardScaler()),                     # feature scaling
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),  # categorical encoding
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }


# --------------------------------------------------------------------------
# 3. Training, comparison and model selection
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models... (first run only)")
def train_and_select():
    """
    Train every candidate model inside a full Pipeline (preprocessor + regressor),
    evaluate with cross-validation and a hold-out test set, and return the best one.
    """
    df = load_data()
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}

    for name, model in get_candidate_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])

        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        results.append(
            {
                "Model": name,
                "CV R2": round(cv_r2, 4),
                "Test R2": round(r2_score(y_test, pred), 4),
                "Test MAE": round(mean_absolute_error(y_test, pred), 3),
                "Test RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
            }
        )
        fitted[name] = pipe

    results_df = (
        pd.DataFrame(results).sort_values("CV R2", ascending=False).reset_index(drop=True)
    )
    best_name = results_df.loc[0, "Model"]

    # Refit the winning pipeline on ALL data for deployment
    best_pipe = Pipeline(
        [("preprocess", build_preprocessor()), ("model", get_candidate_models()[best_name])]
    )
    best_pipe.fit(X, y)

    return best_pipe, best_name, results_df


# --------------------------------------------------------------------------
# 4. Streamlit application
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="centered")
    st.title("🚗 Automobile Fuel Efficiency Predictor")
    st.caption("Predict a vehicle's Miles Per Gallon (MPG) from its technical specifications.")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH.name}`. Place auto-mpg.csv next to app.py.")
        st.stop()

    df = load_data()
    model, best_name, results_df = train_and_select()

    tab_predict, tab_models, tab_data = st.tabs(
        ["🔮 Predict MPG", "📊 Model comparison", "🗂️ Dataset"]
    )

    # ---------------- Prediction tab ----------------
    with tab_predict:
        st.subheader("Enter vehicle details")
        c1, c2 = st.columns(2)

        with c1:
            cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
            displacement = st.number_input(
                "Displacement (cu. inches)", min_value=50.0, max_value=500.0,
                value=150.0, step=1.0,
            )
            weight = st.number_input(
                "Weight (lbs)", min_value=1000, max_value=6000, value=2800, step=10
            )
            origin_label = st.selectbox("Origin", list(ORIGIN_REVERSE.keys()))

        with c2:
            hp_known = st.checkbox("I know the horsepower", value=True)
            horsepower = st.number_input(
                "Horsepower", min_value=40.0, max_value=300.0, value=95.0, step=1.0,
                disabled=not hp_known,
            )
            acceleration = st.number_input(
                "Acceleration (0-60 mph, seconds)", min_value=5.0, max_value=30.0,
                value=15.0, step=0.1,
            )
            model_year = st.slider("Model year (19xx)", min_value=70, max_value=82, value=76)

        if not hp_known:
            st.info("Horsepower unknown: it will be filled in automatically by the pipeline's imputer.")

        if st.button("Predict MPG", type="primary", use_container_width=True):
            input_df = pd.DataFrame(
                [
                    {
                        "cylinders": cylinders,
                        "displacement": displacement,
                        "horsepower": horsepower if hp_known else np.nan,
                        "weight": weight,
                        "acceleration": acceleration,
                        "model_year": model_year,
                        "origin": origin_label,
                    }
                ]
            )
            prediction = float(model.predict(input_df)[0])

            st.success(f"### Predicted fuel efficiency: **{prediction:.1f} MPG**")
            st.caption(
                f"≈ {prediction * 0.4251:.1f} km/l  |  ≈ {235.215 / max(prediction, 0.1):.1f} L/100 km"
                f"  •  Model used: {best_name}"
            )

    # ---------------- Model comparison tab ----------------
    with tab_models:
        st.subheader("Algorithm comparison")
        st.write(
            "Each algorithm is wrapped in the same preprocessing pipeline "
            "(imputation → scaling → encoding) and evaluated with 5-fold cross-validation "
            "on the training split plus a 20% hold-out test set."
        )
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.success(f"Best model (highest CV R²) selected for deployment: **{best_name}**")

    # ---------------- Dataset tab ----------------
    with tab_data:
        st.subheader("Dataset overview")
        st.write(f"Rows: **{df.shape[0]}**  |  Columns: **{df.shape[1]}**")
        st.dataframe(df.head(20), use_container_width=True)
        st.write("**Missing values per column (after cleaning '?' to NaN):**")
        st.dataframe(
            df.isna().sum().rename("missing").to_frame(), use_container_width=True
        )
        st.write("**Summary statistics:**")
        st.dataframe(df.describe(), use_container_width=True)


if __name__ == "__main__":
    main()
