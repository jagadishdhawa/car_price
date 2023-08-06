import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved machine learning model
model = pickle.load('best_model.pkl')

# Function to predict car price
def predict_price(year, km_driven, fuel, seller_type, transmission, brand, owner):
    input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'brand': [brand],
        'owner': [owner]
    })

    prediction = model.predict(input_data)
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

if __name__ == '__main__':
    main()
