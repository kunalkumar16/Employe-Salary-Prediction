# app.py
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model
model = joblib.load('salary_predictor.pkl')

# App title
st.title('🏢 Employee Salary Predictor')
st.write("""
Predict employee salaries based on:
- Age
- Gender
- Education Level
- Job Title
- Years of Experience
""")

# Sidebar
st.sidebar.header('About')
st.sidebar.info("""
This app predicts employee salaries using a Random Forest Regressor model 
trained on historical salary data.
""")

# Input fields
st.header('Employee Information')
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 20, 65, 30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    education = st.selectbox('Education Level', [
        "Bachelor's", 
        "Master's", 
        "PhD"
    ])

with col2:
    experience = st.slider('Years of Experience', 0, 30, 5)
    job_title = st.selectbox('Job Title', [
        'Software Engineer',
        'Data Scientist',
        'Data Analyst',
        'Product Manager',
        'Marketing Analyst',
        'Sales Manager',
        'HR Manager',
        'Financial Analyst',
        'Senior Manager',
        'Director'
    ])

# Prediction button
if st.button('Predict Salary'):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [experience]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f'Predicted Salary: ${prediction:,.2f}')
    
    # Show salary distribution
    st.subheader('Salary Distribution for Similar Profiles')
    st.write("""
    The chart below shows how the predicted salary compares to typical ranges 
    for similar education levels and experience.
    """)
    
    # Dummy data for visualization (in a real app, you'd use actual distribution data)
    salary_ranges = pd.DataFrame({
        'Percentile': ['25th', '50th', '75th'],
        'Salary': [
            prediction * 0.8,
            prediction,
            prediction * 1.2
        ]
    })
    
    st.bar_chart(salary_ranges.set_index('Percentile'))

# Footer
st.markdown("---")
st.markdown("© 2023 Employee Salary Prediction System made by Anurag Tiwari")