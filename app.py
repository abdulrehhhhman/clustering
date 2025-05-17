import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('myModel.pkl', 'rb') as f:
    scaler, kmeans = pickle.load(f)

st.title("Customer Segmentation App")
st.write("Enter customer details to find out their segment.")

# Input fields
age = st.slider('Age', 18, 70, 30)
income = st.slider('Annual Income (k$)', 15, 150, 60)
score = st.slider('Spending Score (1-100)', 1, 100, 50)

# Predict cluster
if st.button('Predict Segment'):
    data = np.array([[age, income, score]])
    data_scaled = scaler.transform(data)
    cluster = kmeans.predict(data_scaled)
    st.success(f'The customer belongs to segment {cluster[0]}')
