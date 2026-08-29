# 🏏 T20 Cricket Score Predictor

Predict the final score of a T20 International cricket team **during a live match**, using current match conditions — runs, overs, wickets, run rate, and more.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-black)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

In a live cricket match, it's difficult to estimate the final score based only on the current score. Many factors — overs completed, wickets lost, run rate, venue, and the two teams involved — all affect the final outcome. This project predicts the **final score** of the batting team using the match state at any point in the innings.

## 🎯 Objective

Build a Machine Learning model that estimates a team's final score from current match statistics, helping analysts, commentators, and fans understand the expected outcome mid-match.

## 🧠 Approach

A **data-driven regression model** trained on historical T20 International match data. The model learns scoring patterns from thousands of past innings and applies them to the current match state to project the final total.

---

## ✨ Features Used

| Feature | Description |
|---|---|
| **Batting Team** | The team currently batting |
| **Bowling Team** | The team currently bowling |
| **City** | Venue of the match (affects pitch & scoring conditions) |
| **Current Score** | Runs scored so far in the innings |
| **Overs Completed** | Overs (and balls) played so far |
| **Wickets Fallen** | Wickets lost by the batting team |
| **Last 5 Overs Runs** | Runs scored in the last 5 overs (recent momentum) |

From these, the model derives `balls_left`, `wickets_left`, and `crr` (current run rate) as engineered features.

---

## 📂 Dataset

- **Source:** [Kaggle](https://www.kaggle.com/) — T20 International historical match data
- **Format:** CSV, ball-by-ball / over-level match statistics
- **Coverage:** ~40,000 rows spanning matches from approximately 2005–2025
- **Columns include:** teams, city, overs, runs, wickets, run rate, and final score

The raw dataset is included at [`data/t20I_cricket_dataset.csv`](data/t20I_cricket_dataset.csv).

---

## 🧹 Data Preprocessing

1. **Removed unnecessary columns** — kept only features relevant to prediction.
2. **Filtered rare cities/teams** — venues and teams with very few matches were dropped to keep the training data consistent.
3. **One-Hot Encoding** — `batting_team`, `bowling_team`, and `city` converted to numeric form.
4. **Train-Test Split (80/20)** — 80% of data used for training, 20% held out for evaluation.

See [`src/train.py`](src/train.py) for the full, reproducible pipeline.

---

## 🤖 Model

**XGBoost Regressor**, wrapped in a scikit-learn `Pipeline` alongside the `ColumnTransformer` used for encoding — so raw match details go in, and a predicted score comes out, in one step.

- Handles non-linear relationships between match variables well
- Fast to train, strong out-of-the-box accuracy on tabular data

### Evaluation

| Metric | What it measures |
|---|---|
| **R² Score** | How well predictions match actual final scores |
| **MAE (Mean Absolute Error)** | Average prediction error, in runs |

On the held-out test set, the model's average prediction error is roughly **±10 runs**.

---

## 🌐 Web App (Flask)

A simple web interface lets you enter live match details and get an instant predicted final score:

1. User enters batting/bowling team, city, current score, overs, wickets, and recent run rate.
2. The trained pipeline processes the input.
3. The predicted final score is displayed on the page.

Runs locally with Flask's built-in dev server.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/t20-score-predictor.git
cd t20-score-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

A pre-trained model is already included at `models/pipe.pkl`, so you can run the app directly:

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### 5. (Optional) Retrain the model yourself

```bash
python src/train.py
```

This regenerates `models/pipe.pkl` from `data/t20I_cricket_dataset.csv` and prints the R² / MAE evaluation.

---

## 📁 Project Structure

```
t20-score-predictor/
├── app.py                  # Flask web application
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── t20I_cricket_dataset.csv
├── models/
│   └── pipe.pkl             # Trained pipeline (encoder + XGBoost model)
├── src/
│   └── train.py             # Reproducible training script
├── templates/
│   └── index.html           # Web UI
└── static/
    └── style.css
```

---

## 🌍 Real-World Applications

- **Live match score prediction** — estimate the final total during a live game
- **Sports analytics platforms** — power insights for commentators and analysts
- **Fantasy cricket** — inform player-selection decisions
- **Match strategy planning** — help coaches decide required run rate & approach

---

## ⚠️ Challenges Faced

- **Unknown categories** — city/team names appearing at prediction time but not in training data required careful encoding (`OneHotEncoder(handle_unknown=...)` considerations).
- **Filtering trade-off** — removing rare cities/teams improved consistency but reduced available training data.
- **Prediction variance** — cricket is inherently unpredictable; conditions can shift quickly mid-match, so predictions are estimates, not certainties.
- **Retraining** — the model needs periodic retraining as new match data becomes available.

---

## 🔮 Future Scope

- ☁️ Deploy to a cloud platform (Render, Railway, AWS, etc.) for public access
- 🔌 Integrate a real-time API to pull live match data automatically
- 📊 Add a live win-probability model alongside the score predictor

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).
