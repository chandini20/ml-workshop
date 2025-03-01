import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.title("Interactive Model Deployment without PyCaret")

# Upload a model file
uploaded_model = st.file_uploader("Upload your model (.joblib)", type=["joblib"])

if uploaded_model:
    # Save and load the uploaded model
    with open("best_model.joblib", "wb") as f:
        f.write(uploaded_model.getbuffer())

    model = joblib.load("best_model.joblib")
    st.success("Model uploaded and loaded successfully!")

    # Dynamically ask for feature inputs
    st.subheader("Enter Feature Values")

    # Example feature names (these should match your model's features)
    feature_names = ['feature1', 'feature2', 'feature3']
    user_inputs = {}

    for feature in feature_names:
        user_inputs[feature] = st.text_input(f"Enter {feature}", "")

    # Convert input to DataFrame
    if st.button("Predict"):
        # Convert input values to correct data types
        input_df = pd.DataFrame([user_inputs])
        
        # Predict using the model
        predictions = model.predict(input_df)
        st.success(f"Prediction: {predictions[0]}")
