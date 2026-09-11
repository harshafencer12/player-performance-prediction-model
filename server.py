from flask import Flask, render_template, request, jsonify
from src.predict import predict_player_performance, data

app = Flask(__name__)

@app.route("/")
def home():
    players = sorted(data["Batter"].unique())
    return render_template("index.html", players=players)

@app.route("/player_stats", methods=["POST"])
def player_stats():
    player_name = request.json.get("player_name")
    player_history = data[data["Batter"] == player_name].sort_values("Date")
    if player_history.empty:
        return jsonify({"error": "No data found for this player"}), 400
    latest = player_history.iloc[-1]
    recent = player_history[["Date", "runs_scored"]].tail(10).sort_values("Date", ascending=False)
    return jsonify({
        "career_avg": round(latest["career_avg_runs"], 1),
        "recent_form": round(latest["recent_form_runs"], 1),
        "matches_played": int(latest["matches_played_so_far"]),
        "history": [
            {"date": row["Date"].strftime("%d %b %Y"), "runs": int(row["runs_scored"])}
            for _, row in recent.iterrows()
        ]
    })

@app.route("/predict", methods=["POST"])
def predict():
    player_name = request.json.get("player_name")
    try:
        predicted_runs = predict_player_performance(player_name)
        return jsonify({"predicted_runs": predicted_runs})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)