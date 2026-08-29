# 🏏 T20 Cricket Score Predictor (End-to-End ML System)

An end-to-end Machine Learning web application designed to forecast the final innings score in live T20 International matches using historical match momentum and in-game statistics.

---

## 📌 Project Overview
Estimating final scores in modern T20 cricket is challenging due to dynamic variables such as wickets lost, pitch dynamics, and recent momentum. This project implements an **XGBoost Regression** pipeline trained on ball-by-ball T20 International data (2005–2025) to provide accurate real-time score estimations.

### Key Use Cases
* **Sports Analytics & Broadcasts:** Generating real-time expected score projections.
* **Fantasy Cricket Strategy:** Assessing team totals and scoring momentum for better player selection.
* **Team Strategy:** Assisting coaches and analysts in pacing run chases and target defense.

---

## 📊 Dataset & Feature Engineering

* **Source:** Kaggle (Historical T20 International ball-by-ball & over-level records from ~2005 to 2025).
* **Format:** CSV processed via `pandas` and `numpy`.

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Batting Team` | Categorical | Team currently batting (encodes batting strength) |
| `Bowling Team` | Categorical | Team currently bowling (encodes bowling attack) |
| `City` | Categorical | Host venue/city (accounts for pitch conditions) |
| `Current Score` | Numerical | Total runs scored up to current ball |
| `Overs Completed` | Numerical | Overs delivered (measures remaining balls) |
| `Wickets Fallen` | Numerical | Total wickets lost |
| `Last 5 Overs Runs`| Numerical | Runs scored in the previous 30 balls (momentum metric) |

### Preprocessing Pipeline
1. **Noise Reduction:** Dropped non-informative metadata and unneeded match identifiers.
2. **Threshold Filtering:** Filtered out obscure venues with sparse historical match frequencies.
3. **Categorical Encoding:** Applied One-Hot Encoding to categorical variables (`batting_team`, `bowling_team`, `city`).
4. **Data Splitting:** 80-20 train-test split for unbiased evaluation.

---

## ⚙️ Model Architecture & Performance

The core prediction engine uses an **XGBoost Regressor** to model complex, non-linear relationships across cricket dynamics.

* **Algorithm:** Extreme Gradient Boosting (XGBoost Regressor)
* **Evaluation Metrics:**
  * **$R^2$ Score:** High variance explanation on test data.
  * **Mean Absolute Error (MAE):** $\approx \pm 10\text{ runs}$ across unseen match conditions.

---

## 📁 Repository Structure

```text
├── templates/
│   └── index.html               # Prediction form UI
├── app.py                       # Flask web server & prediction routes
├── pipe.pkl                     # Serialized XGBoost pipeline / model
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
