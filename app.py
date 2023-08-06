import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
import pickle



def preprocess_data(df):
    le_seller = preprocessing.LabelEncoder()
    le_seller.fit(df["seller_type"])
    df["seller_type"] = le_seller.transform(df["seller_type"])

    le_trans = preprocessing.LabelEncoder()
    le_trans.fit(df["transmission"])
    df["transmission"] = le_trans.transform(df["transmission"])

    le_fuel = preprocessing.LabelEncoder()
    le_fuel.fit(df["fuel"])
    df["fuel"] = le_fuel.transform(df["fuel"])

    le_owner = preprocessing.LabelEncoder()
    le_owner.fit(df["owner"])
    df["owner"] = le_owner.transform(df["owner"])


    le_brand = preprocessing.LabelEncoder()
    le_brand.fit(df["brand"])
    df["brand"] = le_brand.transform(df["brand"])


def predict_price(year, km_driven, fuel, seller_type, transmission, brand, owner):
    # Create a dictionary with the input features
    input_data = {
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "brand": [brand],
        "owner": [owner]
    }
    input_df = pd.DataFrame(input_data)
    input_df = preprocess_data(input_data)
    best_model = joblib.load('best_model.pkl')

    prediction = best_model.predict(input_data)
    return prediction[0]



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

