import streamlit as st
import pandas as pd
import joblib

# Load your saved preprocessor and model (update file paths as needed)
preprocessor = joblib.load('preprocessor.pkl')
stacking_regressor = joblib.load('stacking_regressor.pkl')

# Categorical columns as used in training
categorical_cols = ['Gender', 'Occupation', 'BMI Category']

# Feature columns in exact order and names as training data
feature_columns = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration',
    'Physical Activity Level', 'Stress Level', 'BMI Category',
    'Blood Pressure', 'Heart Rate', 'Daily Steps'
]

# Dtype mapping from training data
dtype_mapping = {
    'Gender': 'object',
    'Age': 'int64',
    'Occupation': 'object',
    'Sleep Duration': 'float64',
    'Physical Activity Level': 'int64',
    'Stress Level': 'float64',
    'BMI Category': 'object',
    'Blood Pressure': 'object',  # Important: Blood Pressure as object/string
    'Heart Rate': 'int64',
    'Daily Steps': 'int64'
}

# Options for categorical inputs
gender_options = ['Male', 'Female']
occupation_options = [
    'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse',
    'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager',
    'Unemployed', 'Student', 'Self-employed'
]
bmi_options = ['Underweight', 'Normal', 'Overweight', 'Obese']

st.title("Sleep Quality Prediction")

# Collect user inputs
gender = st.selectbox("Gender", gender_options)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
occupation = st.selectbox("Occupation", occupation_options)
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.5)
physical_activity = st.number_input("Physical Activity Level", min_value=0, max_value=10, value=3)
stress_level = st.number_input("Stress Level", min_value=0, max_value=10, value=2)
bmi_category = st.selectbox("BMI Category", bmi_options)
blood_pressure = st.text_input("Blood Pressure", value="120")  # Input as string
heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, value=70)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=100000, value=8000)

if st.button("Predict Sleep Quality"):
    # Prepare input dictionary
    new_sample = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,  # already string
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps
    }

    # Create DataFrame with exact columns and order
    input_df = pd.DataFrame([new_sample], columns=feature_columns)

    # Cast columns to correct dtypes
    for col, dtype in dtype_mapping.items():
        try:
            input_df[col] = input_df[col].astype(dtype)
        except Exception as e:
            st.error(f"Error casting column {col} to {dtype}: {e}")
            st.stop()

    # Check for missing values
    if input_df.isnull().any().any():
        st.error("Please fill in all fields before predicting.")
        st.stop()

    # Preprocess input
    try:
        input_processed = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    # Predict
    try:
        pred_score = stacking_regressor.predict(input_processed)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Convert to percentage (assuming max score 10)
    pred_percentage = (pred_score / 10) * 100

    # Classification bins and labels
    bins = [0, 4, 6, 8, 10]
    labels = ['Poor', 'Fair', 'Good', 'Excellent']
    category = pd.cut([pred_score], bins=bins, labels=labels, right=False)[0]

    # Show results
    st.success(f"Sleep Quality Percentage: {pred_percentage:.2f}%")
    st.success(f"Your sleep quality was {category}")
