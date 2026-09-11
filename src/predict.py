import pandas as pd
import joblib

# Load the trained model
model = joblib.load("outputs/player_performance_model.pkl")

# Load the feature dataset to pull a player's latest known stats
data = pd.read_csv("outputs/batting_features_v2.csv")
data["Date"] = pd.to_datetime(data["Date"])

def predict_player_performance(player_name: str) -> float:
    """
    Predicts a player's next-match runs based on their most recent
    available stats (career average, form, opponent/venue history).
    """
    player_data = data[data["Batter"] == player_name].sort_values("Date")

    if player_data.empty:
        raise ValueError(f"No data found for player: {player_name}")

    # Take the player's most recent row — their latest known stats
    latest = player_data.iloc[-1]

    features = [
        "career_avg_runs",
        "career_avg_strike_rate",
        "matches_played_so_far",
        "recent_form_runs",
        "career_avg_vs_opponent",
        "career_avg_at_venue",
    ]

    X = latest[features].values.reshape(1, -1)
    prediction = model.predict(X)[0]

    return round(prediction, 1)


if __name__ == "__main__":
    player = "RD Gaikwad"  # change this to test other players
    predicted_runs = predict_player_performance(player)
    print(f"Predicted runs for {player}: {predicted_runs}")