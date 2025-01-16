import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

def run_streamlit_app():
    """Create and run the Streamlit web application"""
    st.title('Customer Churn Prediction')

    # Load the saved model
    with open('C:\\Users\\HP\\Downloads\\random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Create LabelEncoders for categorical features
    encoders = {
        'Surname': LabelEncoder().fit(data['Surname']),
        'Geography': LabelEncoder().fit(data['Geography']),
        'Gender': LabelEncoder().fit(data['Gender']),
    }

    # Input fields for features
    st.header('Enter Feature Values')
    surname = st.selectbox('Surname', data['Surname'].unique())
    CreditScore = st.number_input('CreditScore', min_value=0.0)
    Geography = st.selectbox('Geography', data['Geography'].unique())
    Gender = st.selectbox('Gender', data['Gender'].unique())
    Age = st.number_input('Age', min_value=0.0)
    Tenure = st.number_input('Tenure', min_value=0.0)
    Balance = st.number_input('Balance', min_value=0.0)
    NumOfProducts = st.number_input('NumOfProducts', min_value=0.0)
    HasCrCard = st.number_input('HasCrCard', min_value=0.0)
    IsActiveMember = st.number_input('IsActiveMember', min_value=0.0)
    EstimatedSalary = st.number_input('EstimatedSalary', min_value=0.0)
    Exited = st.number_input('Exited', min_value=0.0)

    if st.button('Predict'):
        # Encode the categorical inputs
        surname_encoded = encoders['Surname'].transform([surname])[0]
        geography_encoded = encoders['Geography'].transform([Geography])[0]
        gender_encoded = encoders['Gender'].transform([Gender])[0]

        # Construct input features array
        input_features = np.array([[
            surname_encoded, CreditScore, geography_encoded, gender_encoded,
            Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
            EstimatedSalary, Exited
        ]])
           

        # Make prediction
        prediction = model.predict(input_features)

        # Display results
        st.header('Prediction Results')
        if prediction[0] == 1:
            st.error('⚠️ **Warning!** The customer is likely to churn. '
        'We recommend taking proactive retention actions such as offering discounts, '
        'targeted offers, or personalized engagement to retain the customer.')
        else:
            st.success('✅ **Good news!** The customer is not likely to churn. '
        'You can focus on nurturing and strengthening the relationship to keep the customer satisfied.'
    )

# Run the Streamlit app
if __name__ == '__main__':
    run_streamlit_app()
