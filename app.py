import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("iris_model.pkl")

# Page settings
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# App heading
st.title("🌸 Iris Flower Species Predictor")

st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species."
)

st.divider()

# Input section
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=0.1, value=5.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)", min_value=0.1, value=1.4
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=0.1, value=3.5
    )

    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.1, value=0.2
    )

st.divider()

# Prediction
if st.button("Predict Flower Species", use_container_width=True):

    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    st.success(
        f"Predicted Flower Species: {prediction.capitalize()}"
    )

    st.subheader("Prediction Confidence")

    for species, probability in zip(model.classes_, probabilities):

        percentage = probability * 100

        st.write(f"{species.capitalize()}: {percentage:.2f}%")

        st.progress(float(probability))

st.caption("Powered by a trained Logistic Regression model.")