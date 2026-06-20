import streamlit as st

st.title("🏏 Cricket Score Predictor")

runs = st.number_input("Current Runs", min_value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0)

if st.button("Predict Score"):
    predicted_score = runs + (20 - overs) * 8
    st.success(f"Predicted Final Score: {int(predicted_score)}")
