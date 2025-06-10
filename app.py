import streamlit as st
import pandas as pd
import joblib

# Load your saved preprocessor and model (update file paths as needed)
# Ensure 'preprocessor.pkl' and 'stacking_regressor.pkl' are in the same directory
# as your Streamlit app, or provide the full path.
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
    'Physical Activity Level': 'int64', # Stays int64 as it's a whole number on a 0-10 scale
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

# Set page configuration for better aesthetics
st.set_page_config(page_title="Sleep Quality Predictor", layout="centered")

st.title("😴 Sleep Quality Prediction")
st.markdown("Enter your details below to get an estimated sleep quality score and personalized recommendations.")

# --- User Inputs ---
st.header("Your Information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", gender_options)
    age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Your age in years.")
    occupation = st.selectbox("Occupation", occupation_options)
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.5, step=0.1, help="Average hours of sleep per night.")
    # Stress Level input from 1 to 10
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5, help="How stressed do you feel on a scale of 1 to 10?")
with col2:
    # UPDATED: Physical Activity Level on a scale of 0 to 10
    physical_activity = st.number_input(
        "Physical Activity Level",
        min_value=0,
        max_value=10,
        value=5,
        help="Your average physical activity level on a scale of 0 to 10 (0 = sedentary, 10 = extremely active)."
    )
    bmi_category = st.selectbox("BMI Category", bmi_options)
    blood_pressure = st.text_input("Blood Pressure (Systolic/Diastolic)", value="120/80", help="e.g., 120/80 or 130/85")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=200, value=70, help="Your average resting heart rate.")
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=8000, help="Average number of steps you take daily.")

# --- Prediction Button ---
st.markdown("---")
if st.button("✨ Predict Sleep Quality"):
    # Prepare input dictionary
    new_sample = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Physical Activity Level': physical_activity,
        'Stress Level': float(stress_level), # Ensure float as per dtype_mapping if needed, slider gives int
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,
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
            st.error(f"Error casting column **{col}** to **{dtype}**: {e}. Please check your input format.")
            st.stop()

    # Check for missing values (though Streamlit widgets usually prevent this for required inputs)
    if input_df.isnull().any().any():
        st.error("It looks like some fields are empty. Please fill in all the details before predicting.")
        st.stop()

    # Preprocess input
    try:
        input_processed = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"An error occurred during preprocessing your input: {e}. Please ensure inputs are valid.")
        st.stop()

    # Predict
    try:
        pred_score = stacking_regressor.predict(input_processed)[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}. The model might be having issues.")
        st.stop()

    # Convert to percentage (assuming max score 10)
    # Clamp the prediction to be within [0, 10] before converting to percentage
    pred_score = max(0, min(10, pred_score))
    pred_percentage = (pred_score / 10) * 100

    # Classification bins and labels
    # Adjusted upper bound to slightly above 10 to ensure 10 itself falls into 'Excellent'
    bins = [0, 4, 6, 8, 10.1]
    labels = ['Poor', 'Fair', 'Good', 'Excellent']
    category = pd.cut([pred_score], bins=bins, labels=labels, right=False)[0]

    # --- Show Results ---
    st.subheader("🎉 Your Sleep Quality Results:")
    st.markdown(f"**Predicted Sleep Quality Score:** {pred_score:.2f} (out of 10)")
    st.markdown(f"**Sleep Quality Percentage:** {pred_percentage:.2f}%")
    st.markdown(f"**Overall Assessment:** Your sleep quality is considered **{category}**.")

    # --- Recommendations ---
    st.markdown("---")
    st.subheader("💡 Recommendations for Improving Sleep Quality:")

    if category == 'Poor':
        st.error("### 🔴 Your Sleep Quality is Poor.")
        st.markdown("""
        **Urgent attention is recommended.** Poor sleep quality can significantly impact your health, mood, and cognitive function.
        """)
        st.markdown("""
        **Considerations for Sleep Disorder Risk:**
        * **High Risk:** There's a higher likelihood of an underlying sleep disorder (e.g., insomnia, sleep apnea, restless legs syndrome).
        * **Action:** It is **highly recommended** that you consult a healthcare professional, ideally a sleep specialist. They can help diagnose any underlying issues and recommend appropriate treatments.
        """)
        st.markdown("""
        **Immediate Steps You Can Take:**
        * **Establish a Strict Sleep Schedule:** Go to bed and wake up at the same time every day, even on weekends.
        * **Optimize Your Sleep Environment:** Ensure your bedroom is dark, quiet, and cool.
        * **Avoid Stimulants:** Limit caffeine and nicotine, especially in the afternoon and evening.
        * **Limit Screen Time Before Bed:** Avoid electronic devices (phones, tablets, computers, TVs) at least an hour before sleep.
        * **Gentle Relaxation:** Try reading a book, taking a warm bath, or practicing light stretching before bed.
        """)

    elif category == 'Fair':
        st.warning("### 🟠 Your Sleep Quality is Fair.")
        st.markdown("""
        You're getting some rest, but there's significant room for improvement. Consistent fair sleep can still lead to negative health outcomes over time.
        """)
        st.markdown("""
        **Considerations for Sleep Disorder Risk:**
        * **Moderate Risk:** While not immediately alarming, persistent fair sleep could indicate developing sleep issues or lifestyle habits that hinder restorative sleep.
        * **Action:** If these issues persist, consider discussing your sleep patterns with your doctor.
        """)
        st.markdown("""
        **Tips for Improvement:**
        * **Review Your Sleep Hygiene:** Re-evaluate your daily habits related to sleep. Are you consistent with your sleep schedule? Is your bedroom conducive to sleep?
        * **Incorporate Relaxation Techniques:** Practice mindfulness, meditation, or deep breathing exercises during the day or before bed to manage stress.
        * **Regular Physical Activity:** Engage in moderate exercise most days of the week, but avoid strenuous workouts too close to bedtime.
        * **Watch Your Diet:** Avoid heavy meals, excessive sugar, and alcohol close to bedtime.
        * **Consider a Sleep Diary:** Track your sleep patterns, habits, and how you feel daily. This can help identify triggers or patterns.
        """)

    elif category == 'Good':
        st.info("### 🟢 Your Sleep Quality is Good.")
        st.markdown("""
        You're generally getting sufficient and restorative sleep. This is a positive indicator for your overall health and well-being.
        """)
        st.markdown("""
        **Considerations for Sleep Disorder Risk:**
        * **Low Risk:** You are likely not experiencing a major sleep disorder.
        * **Action:** Continue with your healthy sleep habits. Be mindful of any changes in your routine or stress levels that could affect sleep.
        """)
        st.markdown("""
        **Maintain and Enhance:**
        * **Consistency is Key:** Continue to prioritize a consistent sleep schedule.
        * **Healthy Lifestyle:** Maintain your balanced diet, regular exercise, and stress management practices.
        * **Listen to Your Body:** Pay attention to how you feel. If you notice any dips in energy or changes in your sleep, revisit your habits.
        * **Stay Hydrated:** Ensure adequate water intake throughout the day, but reduce fluids before bed to avoid nighttime awakenings.
        """)

    elif category == 'Excellent':
        st.balloons() # Add a fun animation for excellent sleep!
        st.success("### 💎 Your Sleep Quality is Excellent!")
        st.markdown("""
        Congratulations! You are achieving highly restorative and consistent sleep, which is fundamental for optimal physical and mental health.
        """)
        st.markdown("""
        **Considerations for Sleep Disorder Risk:**
        * **Very Low Risk:** Your current patterns suggest excellent sleep health.
        * **Action:** Continue to maintain your exemplary sleep hygiene and healthy lifestyle choices.
        """)
        st.markdown("""
        **Keep Up the Great Work!**
        * **Be a Role Model:** Share your healthy sleep habits with others!
        * **Monitor for Changes:** While excellent, life circumstances can change. Be aware of stress, travel, or other factors that might temporarily impact your sleep.
        * **Continuous Learning:** Stay informed about new research on sleep and well-being to continually optimize your health.
        """)

    st.markdown("---")
    st.info("""
    *Disclaimer: This prediction and the recommendations are based on a machine learning model and general health guidelines. They are not a substitute for professional medical advice. If you have serious concerns about your sleep or health, please consult a qualified healthcare professional.*
    """)