import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle


# Load dataset
df = pd.read_csv('CAR DETAILS.csv')

# Load the pipeline
with open('best_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

best_model = 'best_model.pkl'
# Predict function
def predict_price(year, km_driven, fuel, seller_type, transmission, brand, owner):
    # Create a dictionary with the input features
    input_data = {
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "brand": brand,
        "owner": owner
    }

    # Convert input data to DataFrame and preprocess
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_data(input_df)

    # Make prediction
    prediction = best_model.predict(input_df)

    return prediction


# Streamlit UI
st.title('Car Price Prediction')
st.write('This app predicts the selling price of a used car.')

# Input fields
year = st.number_input('Year', min_value=1990, max_value=2023, step=1)
km_driven = st.number_input('Kilometers Driven', min_value=0, step=1000)
fuel = st.selectbox('Fuel', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
brand = st.text_input('Car Brand', '')
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

# Predict button
if st.button('Predict'):
    prediction = predict_price(year, km_driven, fuel, seller_type, transmission, brand, owner)
    st.success(f'Predicted Selling Price: Rs. {prediction:.2f}')


