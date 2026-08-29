from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Model loading
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except Exception:
    pipe = None

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
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts',
    'Cardiff', 'Christchurch', 'Trinidad'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        current_score = int(request.form['current_score'])
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets'])
        last_five = int(request.form['last_five'])

        balls_bowled = int(overs) * 6 + int(round((overs % 1) * 10))
        balls_left = 120 - balls_bowled
        wickets_left = 10 - wickets
        crr = current_score / overs if overs > 0 else 0

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

        if pipe:
            result = pipe.predict(input_df)
            prediction_text = int(round(result[0]))
        else:
            prediction_text = "Model file (pipe.pkl) not found. Upload pipe.pkl file."

    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
