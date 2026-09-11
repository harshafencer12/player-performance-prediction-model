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
match_dates = df[["Match ID", "Date"]].drop_duplicates()
batting = batting.merge(match_dates, on="Match ID", how="left")
batting["Date"] = pd.to_datetime(batting["Date"])
batting = batting.sort_values(["Batter", "Date"])

batting["career_avg_runs"] = (
    batting.groupby("Batter")["runs_scored"].transform(lambda x: x.shift(1).expanding().mean())
)
batting["career_avg_strike_rate"] = (
    batting.groupby("Batter")["strike_rate"].transform(lambda x: x.shift(1).expanding().mean())
)
batting["matches_played_so_far"] = batting.groupby("Batter").cumcount()

# ---------- DETERMINE BATTING TEAM & OPPONENT (fast, vectorized) ----------
match_teams = df[["Match ID", "Bat First", "Bat Second"]].drop_duplicates()
batter_innings = df[["Match ID", "Batter", "Innings"]].drop_duplicates(subset=["Match ID", "Batter"])

batting = batting.merge(batter_innings, on=["Match ID", "Batter"], how="left")
batting = batting.merge(match_teams, on="Match ID", how="left")

batting["batting_team"] = batting["Bat First"].where(batting["Innings"] == 1, batting["Bat Second"])
batting["opponent_team"] = batting["Bat Second"].where(batting["Innings"] == 1, batting["Bat First"])

# Venue
match_venue = df[["Match ID", "Venue"]].drop_duplicates()
batting = batting.merge(match_venue, on="Match ID", how="left")

# ---------- ROLLING LAST-5 FORM ----------
batting = batting.sort_values(["Batter", "Date"])
batting["recent_form_runs"] = (
    batting.groupby("Batter")["runs_scored"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

# ---------- OPPONENT-SPECIFIC CAREER AVERAGE ----------
batting["career_avg_vs_opponent"] = (
    batting.groupby(["Batter", "opponent_team"])["runs_scored"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
batting["career_avg_vs_opponent"] = batting["career_avg_vs_opponent"].fillna(batting["career_avg_runs"])

# ---------- VENUE-SPECIFIC CAREER AVERAGE ----------
batting["career_avg_at_venue"] = (
    batting.groupby(["Batter", "Venue"])["runs_scored"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
batting["career_avg_at_venue"] = batting["career_avg_at_venue"].fillna(batting["career_avg_runs"])

# ---------- FINALIZE MODEL-READY DATA ----------
model_ready_batting = batting[batting["matches_played_so_far"] > 0].copy()
model_ready_batting["recent_form_runs"] = model_ready_batting["recent_form_runs"].fillna(model_ready_batting["career_avg_runs"])

model_ready_batting.to_csv("outputs/batting_features_v2.csv", index=False)
print("Enhanced features sample:")
print(model_ready_batting[[
    "Match ID","Batter","Date","runs_scored","career_avg_runs",
    "recent_form_runs","career_avg_vs_opponent","career_avg_at_venue"
]].head(10))

# ---------- TRAIN / TEST SPLIT (TIME-BASED) ----------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

model_ready_batting = model_ready_batting.sort_values("Date")

features = [
    "career_avg_runs",
    "career_avg_strike_rate",
    "matches_played_so_far",
    "recent_form_runs",
    "career_avg_vs_opponent",
    "career_avg_at_venue",
]
target = "runs_scored"

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

importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

# ---------- SAVE THE TRAINED MODEL ----------
import joblib

joblib.dump(model, "outputs/player_performance_model.pkl")
print("\nModel saved to outputs/player_performance_model.pkl")

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

print("\nBatting stats sample:")
print(batting.head())
print("\nBowling stats sample:")
print(bowling.head())