import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load pre-trained components
preprocessor = joblib.load('preprocessor.pkl')
stacking_regressor = joblib.load('stacking_regressor.pkl')

# Constants
categorical_cols = ['Gender', 'Occupation', 'BMI Category']
feature_columns = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration',
    'Physical Activity Level', 'Stress Level', 'BMI Category',
    'Blood Pressure', 'Heart Rate', 'Daily Steps'
]
dtype_mapping = {
    'Gender': 'object', 'Age': 'int64', 'Occupation': 'object',
    'Sleep Duration': 'float64', 'Physical Activity Level': 'int64',
    'Stress Level': 'float64', 'BMI Category': 'object',
    'Blood Pressure': 'object', 'Heart Rate': 'int64', 'Daily Steps': 'int64'
}
gender_options = ['Male', 'Female']
occupation_options = [
    'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse',
    'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager',
    'Unemployed', 'Student', 'Self-employed'
]
bmi_options = ['Underweight', 'Normal', 'Overweight', 'Obese']

# UI Setup
st.set_page_config(page_title="Sleep Quality Analyser", layout="centered")
st.title("Sleep Quality Analyser")
st.markdown("Enter your information below to receive an estimated sleep quality score.")

# --- Input Section ---
st.header("Input Your Details")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", gender_options)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    occupation = st.selectbox("Occupation", occupation_options)
    sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 24.0, value=7.5, step=0.1)
    stress_level = st.slider("Stress Level (1–10)", 1, 10, value=5)
with col2:
    physical_activity = st.number_input("Physical Activity Level (0–10)", 0, 10, value=5)
    bmi_category = st.selectbox("BMI Category", bmi_options)
    blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)", value="120/80")
    heart_rate = st.number_input("Heart Rate (bpm)", 0, 200, value=70)
    daily_steps = st.number_input("Daily Steps", 0, 50000, value=8000)

st.markdown("---")
if st.button("Predict Sleep Quality"):
    input_data = {
        'Gender': gender, 'Age': age, 'Occupation': occupation,
        'Sleep Duration': sleep_duration, 'Physical Activity Level': physical_activity,
        'Stress Level': float(stress_level), 'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure, 'Heart Rate': heart_rate, 'Daily Steps': daily_steps
    }
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Validate and preprocess
    for col, dtype in dtype_mapping.items():
        try:
            input_df[col] = input_df[col].astype(dtype)
        except Exception as e:
            st.error(f"Column '{col}' could not be converted to {dtype}: {e}")
            st.stop()

    if input_df.isnull().any().any():
        st.error("Please complete all fields before predicting.")
        st.stop()

    try:
        input_processed = preprocessor.transform(input_df)
        pred_score = stacking_regressor.predict(input_processed)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    pred_score = max(0, min(10, pred_score))
    pred_percentage = (pred_score / 10) * 100
    bins = [0, 4, 6, 8, 10.1]
    labels = ['Poor', 'Fair', 'Good', 'Excellent']
    category = pd.cut([pred_score], bins=bins, labels=labels, right=False)[0]

    # --- Results ---
    st.subheader("Results")
    st.markdown(f"- **Sleep Quality Score:** {pred_score:.2f} / 10")
    st.markdown(f"- **Sleep Quality Percentage:** {pred_percentage:.2f}%")
    st.markdown(f"- **Assessment:** {category}")

    # --- 📊 Visualization: Gauge Meter ---
    st.plotly_chart(go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_score,
        title={'text': "Sleep Quality Score"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 4], 'color': "red"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 8], 'color': "lightgreen"},
                {'range': [8, 10], 'color': "green"},
            ],
        }
    )), use_container_width=True)

    # --- 📊 Visualization: Lifestyle Bar Chart ---
    st.subheader("Lifestyle Factor Overview")
    fig, ax = plt.subplots()
    factors = ['Sleep Duration', 'Physical Activity', 'Stress Level']
    values = [sleep_duration, physical_activity, stress_level]
    ax.bar(factors, values, color=['skyblue', 'green', 'salmon'])
    ax.set_ylim(0, 10)
    ax.set_ylabel("Scale (0–10)")
    st.pyplot(fig)

    # --- Recommendations ---
    st.markdown("---")
    st.subheader("Recommendations")

    if category == 'Poor':
        st.warning("Your sleep quality is low and may require intervention.")
        with st.expander("Suggested Actions"):
            st.markdown("""
            - Maintain a consistent sleep-wake schedule.
            - Limit caffeine, alcohol, and screens before bed.
            - Ensure a comfortable, quiet sleep environment.
            - Consider speaking with a sleep specialist.
            """)

    elif category == 'Fair':
        st.info("Your sleep quality is moderate.")
        with st.expander("Suggestions for Improvement"):
            st.markdown("""
            - Improve sleep hygiene (light, noise, timing).
            - Add light exercise during the day.
            - Avoid screens and stressors before bed.
            """)

    elif category == 'Good':
        st.success("You're doing well with your sleep.")
        with st.expander("Tips to Maintain"):
            st.markdown("""
            - Stick to your routine.
            - Avoid late-night stimulants.
            - Continue physical activity.
            """)

    elif category == 'Excellent':
        st.success("Excellent sleep quality detected.")
        with st.expander("Maintain Healthy Sleep Habits"):
            st.markdown("""
            - Keep up with current habits.
            - Stay alert to stress or routine changes.
            - Maintain your physical health.
            """)

    st.markdown("---")
    st.caption("Note: This tool provides general health recommendations and does not replace medical advice.")
