import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing

import pickle

model=pickle.load(open('best_model.pkl','rb'))

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
    le_seller = preprocessing.LabelEncoder()
    le_seller.fit(input_data["seller_type"])
    input_data["seller_type"] = le_seller.transform(input_data["seller_type"])

    le_trans = preprocessing.LabelEncoder()
    le_trans.fit(input_data["transmission"])
    input_data["transmission"] = le_trans.transform(input_data["transmission"])

    le_fuel = preprocessing.LabelEncoder()
    le_fuel.fit(input_data["fuel"])
    input_data["fuel"] = le_fuel.transform(input_data["fuel"])

    le_owner = preprocessing.LabelEncoder()
    le_owner.fit(input_data["owner"])
    input_data["owner"] = le_owner.transform(input_data["owner"])


    le_brand = preprocessing.LabelEncoder()
    le_brand.fit(input_data["brand"])
    input_data["brand"] = le_brand.transform(input_data["brand"])

    prediction = model.predict(input_data)[0]
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

