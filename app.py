"""
app.py
------
Flask web application for the T20 Cricket Score Prediction project.

Loads the trained pipeline from models/pipe.pkl, serves a simple form
where a user can enter live match details, and displays the predicted
final score.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipe.pkl")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained pipeline once, at startup.
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    pipe = pickle.load(f)

# Pull the exact team / city vocabulary straight out of the fitted
# OneHotEncoder so the dropdowns always match what the model was trained on
# (works whether you use the bundled model or retrain your own with
# src/train.py).
_encoder = pipe.named_steps["step1"].transformers_[0][1]
BATTING_TEAMS = sorted(_encoder.categories_[0].tolist())
BOWLING_TEAMS = sorted(_encoder.categories_[1].tolist())
CITIES = sorted(_encoder.categories_[2].tolist())


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        batting_teams=BATTING_TEAMS,
        bowling_teams=BOWLING_TEAMS,
        cities=CITIES,
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    error = None
    prediction = None
    form_data = request.form

    try:
        batting_team = form_data["batting_team"]
        bowling_team = form_data["bowling_team"]
        city = form_data["city"]
        current_score = float(form_data["current_score"])
        overs_done = int(form_data["overs_done"])
        balls_done = int(form_data["balls_done"])
        wickets_fallen = int(form_data["wickets_fallen"])
        last_five = float(form_data["last_five"])

        if batting_team == bowling_team:
            error = "Batting team and Bowling team must be different."
        elif not (0 <= overs_done <= 20):
            error = "Overs completed must be between 0 and 20."
        elif not (0 <= balls_done <= 5):
            error = "Balls into the current over must be between 0 and 5."
        elif not (0 <= wickets_fallen <= 10):
            error = "Wickets fallen must be between 0 and 10."
        else:
            balls_bowled = overs_done * 6 + balls_done
            balls_left = 120 - balls_bowled
            wickets_left = 10 - wickets_fallen
            crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0.0

            if balls_left <= 0:
                error = "No balls left in the innings."
            else:
                input_df = pd.DataFrame(
                    {
                        "batting_team": [batting_team],
                        "bowling_team": [bowling_team],
                        "city": [city],
                        "current_score": [current_score],
                        "balls_left": [balls_left],
                        "wickets_left": [wickets_left],
                        "crr": [crr],
                        "last_five": [last_five],
                    }
                )
                result = pipe.predict(input_df)[0]
                prediction = int(round(result))

    except (KeyError, ValueError):
        error = "Please fill in every field with a valid value."

    return render_template(
        "index.html",
        batting_teams=BATTING_TEAMS,
        bowling_teams=BOWLING_TEAMS,
        cities=CITIES,
        prediction=prediction,
        error=error,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
