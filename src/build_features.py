import pandas as pd

# Load data
df = pd.read_csv("dataset/ball_by_ball_ipl.csv")

# ---------- BATTING STATS ----------
valid = df[df["Valid Ball"] == 1]

batting = valid.groupby(["Match ID", "Batter"]).agg(
    runs_scored=("Batter Runs", "sum"),
    balls_faced=("Batter Runs", "count"),
    fours=("Batter Runs", lambda x: (x == 4).sum()),
    sixes=("Batter Runs", lambda x: (x == 6).sum()),
).reset_index()

batting["strike_rate"] = (batting["runs_scored"] / batting["balls_faced"] * 100).round(2)

# Was the batter dismissed in this match?
dismissals = df[df["Wicket"] == 1][["Match ID", "Player Out"]].rename(
    columns={"Player Out": "Batter"}
)
dismissals["was_dismissed"] = 1

batting = batting.merge(dismissals.drop_duplicates(), on=["Match ID", "Batter"], how="left")
batting["was_dismissed"] = batting["was_dismissed"].fillna(0).astype(int)


# ---------- CAREER-TO-DATE FORM (BATTING) ----------
# Need match date to sort chronologically
match_dates = df[["Match ID", "Date"]].drop_duplicates()
batting = batting.merge(match_dates, on="Match ID", how="left")
batting["Date"] = pd.to_datetime(batting["Date"])
batting = batting.sort_values(["Batter", "Date"])

# Expanding mean = average of all PRIOR matches only (shift(1) excludes current match)
batting["career_avg_runs"] = (
    batting.groupby("Batter")["runs_scored"].transform(lambda x: x.shift(1).expanding().mean())
)
batting["career_avg_strike_rate"] = (
    batting.groupby("Batter")["strike_rate"].transform(lambda x: x.shift(1).expanding().mean())
)
batting["matches_played_so_far"] = (
    batting.groupby("Batter").cumcount()
)

# Drop rows with no prior history (first match for each player — no valid feature yet)
model_ready_batting = batting[batting["matches_played_so_far"] > 0].copy()

model_ready_batting.to_csv("outputs/batting_features.csv", index=False)
print("Model-ready batting features sample:")
print(model_ready_batting[["Match ID","Batter","Date","runs_scored","career_avg_runs","career_avg_strike_rate","matches_played_so_far"]].head(10))


# ---------- TRAIN / TEST SPLIT (TIME-BASED) ----------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

model_ready_batting = model_ready_batting.sort_values("Date")

features = ["career_avg_runs", "career_avg_strike_rate", "matches_played_so_far"]
target = "runs_scored"

# Time-based split: train on earlier 80%, test on later 20% — NOT random
split_idx = int(len(model_ready_batting) * 0.8)
train = model_ready_batting.iloc[:split_idx]
test = model_ready_batting.iloc[split_idx:]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5

print(f"\nMAE: {mae:.2f} runs")
print(f"RMSE: {rmse:.2f} runs")

# Feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)
# ---------- BOWLING STATS ----------
bowling = valid.groupby(["Match ID", "Bowler"]).agg(
    runs_conceded=("Bowler Runs Conceded", "sum"),
    balls_bowled=("Bowler Runs Conceded", "count"),
    wickets=("Wicket", "sum"),
).reset_index()

bowling["overs_bowled"] = (bowling["balls_bowled"] // 6) + (bowling["balls_bowled"] % 6) / 6
bowling["economy"] = (bowling["runs_conceded"] / bowling["overs_bowled"]).round(2)

# ---------- SAVE ----------
batting.to_csv("outputs/batting_stats.csv", index=False)
bowling.to_csv("outputs/bowling_stats.csv", index=False)

print("Batting stats sample:")
print(batting.head())
print("\nBowling stats sample:")
print(bowling.head())