import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

st.title("Nano Indentation Prediction")

# Inputs
duration = st.number_input("Laser pulse duration (ns)", value=0.0)
energy = st.number_input("Laser energy (mJ)", value=0.0)
loding_rate = st.number_input("Loding rate (µN/s)", value=0.0)
load = st.number_input("Load (µN)", value=0.0)
time = st.number_input("Time (s)", value=0.0)

# Load model once
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Use model's feature names exactly for input dataframe columns
input_df = pd.DataFrame(
    [[duration, energy, loding_rate, load, time]],
    columns=model.feature_names_
)

# Show the input dataframe and model features (for debug)
st.write("✅ Input DataFrame:")
st.write(input_df)

st.write("ℹ️ Model feature names:")
st.write(model.feature_names_)

# List of categorical features (as per your training)
categorical = ['laser pulse duration (ns)', 'laser energy (mJ)', 'loding rate (µN/s)']

# Convert categorical columns to string as model expects
for col in categorical:
    input_df[col] = input_df[col].astype(str)

# Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Output: {prediction}")
