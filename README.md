# 🏏 Cricket Player Performance Predictor

A machine learning project that predicts a batter's expected runs in their next IPL match, built from historical ball-by-ball data (2008–2023), with a custom web interface for interactive predictions.

## Overview

This project takes raw ball-by-ball IPL data and transforms it into player-level performance features, trains a Gradient Boosting Regression model to predict batting performance, and serves predictions through a Flask-powered web app styled around IPL matchday nostalgia.

## Features

- **Data pipeline**: aggregates raw ball-by-ball deliveries into per-match batting and bowling statistics
- **Leakage-safe feature engineering**: career-to-date averages, recent form, and opponent/venue-specific stats — all computed using only matches *prior* to the one being predicted
- **Trained ML model**: Gradient Boosting Regressor (scikit-learn), evaluated using a chronological (not random) train/test split
- **Web interface**: a Flask app with a custom-designed, matchday-themed UI — player selector, live stat preview, and an animated scoreboard-style prediction reveal

## Tech Stack

- **Language**: Python 3.14
- **Data processing**: pandas
- **Machine learning**: scikit-learn (GradientBoostingRegressor)
- **Model persistence**: joblib
- **Backend**: Flask
- **Frontend**: HTML, CSS, vanilla JavaScript
- **Data source**: Kaggle (`jamiewelsh2/ball-by-ball-ipl`)
- **Version control**: Git & GitHub

## Project Structure
player-performance-prediction-model/
├── dataset/ # Raw dataset (gitignored — download separately)
├── outputs/ # Generated feature files and trained model
│ ├── batting_stats.csv
│ ├── bowling_stats.csv
│ ├── batting_features_v2.csv
│ └── player_performance_model.pkl
├── src/
│ ├── build_features.py # Data aggregation, feature engineering, model training
│ └── predict.py # Loads trained model, runs predictions
├── templates/
│ └── index.html # Web UI markup
├── static/
│ ├── style.css # Web UI styling
│ └── script.js # Web UI interactivity
├── server.py # Flask app entry point
├── main.py # CLI entry point for predictions
├── requirements.txt
└── README.md

## Setup & Installation

1. **Clone the repository**
```bash
   git clone https://github.com/harshafencer12/player-performance-prediction-model.git
   cd player-performance-prediction-model
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Download the dataset**
   This project uses the [Ball-by-Ball IPL dataset](https://www.kaggle.com/datasets/jamiewelsh2/ball-by-ball-ipl) from Kaggle.
```bash
   kaggle datasets download -d jamiewelsh2/ball-by-ball-ipl -p dataset/
```
   Unzip it into the `dataset/` folder. (Requires a Kaggle API token — see [Kaggle API docs](https://www.kaggle.com/docs/api).)

4. **Build features and train the model**
```bash
   python src/build_features.py
```
   This generates the feature files and trains/saves the model to `outputs/player_performance_model.pkl`.

## Usage

**Command-line prediction:**
```bash
python main.py
```

**Web interface:**
```bash
python server.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Methodology

1. **Data aggregation** — raw ball-by-ball rows are grouped by match and player to compute batting stats (runs, strike rate, boundaries) and bowling stats (wickets, economy).
2. **Feature engineering** — for each player, historical performance is summarized as of *before* each match: career-to-date average runs, career-to-date strike rate, recent form (last 5 innings), and opponent/venue-specific averages. A `shift(1)` operation before all rolling/expanding calculations ensures no feature uses information from the match being predicted (data leakage prevention).
3. **Model training** — a Gradient Boosting Regressor is trained on these features to predict runs scored, using a chronological train/test split (training on earlier matches, testing on later ones) to simulate real-world forecasting conditions.
4. **Evaluation** — model performance is measured using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Results

| Metric | Baseline (career avg only) | With recent form + opponent/venue features |
|---|---|---|
| MAE | 15.52 runs | 15.48 runs |
| RMSE | 20.71 runs | 20.67 runs |

Adding recent-form and contextual (opponent/venue) features produced only marginal improvement over the baseline. This suggests that single-innings batting performance in T20 cricket carries high inherent variance that historical aggregate statistics alone cannot fully explain — a result consistent with known difficulty in single-match sports outcome prediction. `career_avg_runs` remained the dominant predictive feature throughout (~70–80% relative importance).

## Future Improvements

- Reframe the prediction target (e.g., predicting performance over a rolling next-3-match window, or classifying above/below-average performance) to reduce single-match noise
- Incorporate bowling-side matchup data (specific bowler vs. batter history)
- Add match-situation context (powerplay/death-overs role, batting position)
- Extend the model to bowling performance prediction

## Dataset Credit

Data sourced from [Ball-by-Ball IPL dataset](https://www.kaggle.com/datasets/jamiewelsh2/ball-by-ball-ipl) on Kaggle, covering IPL matches from 2008–2023.

## Author

P Venkata Harshavardhan