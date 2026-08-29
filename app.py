import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load pre-trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'Cardiff'
]

st.set_page_config(page_title="Cricket Score Predictor", page_icon="🏏", layout="centered")
st.title("🏏 T20 Cricket Score Predictor")
st.markdown("Predict the projected final innings total based on live match dynamics.")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

city = st.selectbox('Select City / Venue', sorted(cities))

col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score', min_value=0, step=1, value=50)
with col4:
    overs = st.number_input('Overs Completed (must be >= 5)', min_value=5.0, max_value=20.0, step=0.1, value=6.0)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=9, step=1, value=1)

last_five = st.number_input('Runs Scored in Last 5 Overs', min_value=0, step=1, value=40)

if st.button('Predict Final Score'):
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same!")
    else:
        # Calculate derived features
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_five': [last_five]
        })

        result = pipe.predict(input_df)
        predicted_score = int(result[0])
        
        st.success(f"### 🎯 Predicted Final Score: **{predicted_score} runs**")
