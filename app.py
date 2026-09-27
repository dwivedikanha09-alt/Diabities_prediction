"""
Diabetes Prediction — Streamlit App
Built from the Logistic Regression notebook workflow:
Load data -> Encode categoricals -> Train/Test split -> Scale -> Logistic Regression -> Evaluate -> Predict
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

DATA_FILENAME = "diabetes_prediction_dataset.csv"
FEATURE_COLS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
TARGET_COL = "diabetes"


@st.cache_resource(show_spinner="Training the logistic regression model...")
def train_model(df: pd.DataFrame):
    df = df.copy()

    # Encode categorical columns exactly as in the notebook
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return clf, scaler, encoders, metrics


def load_data():
    if os.path.exists(DATA_FILENAME):
        return pd.read_csv(DATA_FILENAME)

    st.warning(
        f"Couldn't find **{DATA_FILENAME}** next to app.py. "
        "Upload the dataset CSV to train the model."
    )
    uploaded = st.file_uploader("Upload diabetes_prediction_dataset.csv", type="csv")
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return None


def main():
    st.title("🩺 Diabetes Prediction — Logistic Regression")
    st.caption(
        "A Streamlit UI around the Logistic Regression workflow from the notebook: "
        "label encoding, train/test split, feature scaling, and prediction."
    )

    df = load_data()
    if df is None:
        st.stop()

    clf, scaler, encoders, metrics = train_model(df)

    tab_predict, tab_performance, tab_data = st.tabs(
        ["🔮 Predict", "📊 Model Performance", "🗂️ Data Overview"]
    )

    # ---------------- Predict tab ----------------
    with tab_predict:
        st.subheader("Enter patient details")

        gender_options = list(encoders["gender"].classes_)
        smoking_options = list(encoders["smoking_history"].classes_)

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", gender_options)
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            smoking_history = st.selectbox("Smoking history", smoking_options)
            bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
            hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
            glucose = st.number_input("Blood glucose level", min_value=50, max_value=400, value=100, step=1)

        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame(
                [
                    {
                        "gender": encoders["gender"].transform([gender])[0],
                        "age": age,
                        "hypertension": hypertension,
                        "heart_disease": heart_disease,
                        "smoking_history": encoders["smoking_history"].transform([smoking_history])[0],
                        "bmi": bmi,
                        "HbA1c_level": hba1c,
                        "blood_glucose_level": glucose,
                    }
                ]
            )[FEATURE_COLS]

            input_scaled = scaler.transform(input_df)
            prediction = clf.predict(input_scaled)[0]
            probability = clf.predict_proba(input_scaled)[0]

            st.divider()
            if prediction == 1:
                st.error(f"⚠️ Likely **diabetic** — probability {probability[1]*100:.1f}%")
            else:
                st.success(f"✅ Likely **not diabetic** — probability {probability[0]*100:.1f}%")

            st.progress(float(probability[1]), text=f"Diabetes risk: {probability[1]*100:.1f}%")

    # ---------------- Performance tab ----------------
    with tab_performance:
        st.subheader("Test-set performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1 Score", f"{metrics['f1']:.2%}")

        st.subheader("Confusion matrix")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No diabetes", "Actual: Diabetes"],
            columns=["Predicted: No diabetes", "Predicted: Diabetes"],
        )
        st.table(cm_df)

    # ---------------- Data tab ----------------
    with tab_data:
        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        st.subheader("Target distribution")
        st.bar_chart(df[TARGET_COL].value_counts())

        st.subheader("Summary statistics")
        st.dataframe(df.describe(), use_container_width=True)


if __name__ == "__main__":
    main()
