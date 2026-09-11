import streamlit as st
import pandas as pd
from src.predict import predict_player_performance, data

st.set_page_config(page_title="Cricket Performance Predictor", page_icon="🏏", layout="centered")

st.title("🏏 Cricket Player Performance Predictor")
st.write("Predict a player's expected runs in their next match, based on historical IPL performance data (2008–2023).")

st.divider()

player_list = sorted(data["Batter"].unique())
player_name = st.selectbox("Select a player:", player_list)

if player_name:
    player_history = data[data["Batter"] == player_name].sort_values("Date")
    latest = player_history.iloc[-1]

    # Show key stats before prediction
    col1, col2, col3 = st.columns(3)
    col1.metric("Career Avg Runs", f"{latest['career_avg_runs']:.1f}")
    col2.metric("Recent Form (last 5)", f"{latest['recent_form_runs']:.1f}")
    col3.metric("Matches Played", int(latest['matches_played_so_far']))

    st.divider()

    if st.button("🔮 Predict Next Match Performance", use_container_width=True):
        predicted_runs = predict_player_performance(player_name)
        st.success(f"### Predicted runs: {predicted_runs}")

        # Simple visual comparison
        st.write("**How this compares to their career average:**")
        chart_data = pd.DataFrame({
            "Metric": ["Career Average", "Recent Form", "Predicted"],
            "Runs": [latest["career_avg_runs"], latest["recent_form_runs"], predicted_runs]
        })
        st.bar_chart(chart_data.set_index("Metric"))

        # Show recent match history
        st.write("**Recent match history:**")
        st.dataframe(
            player_history[["Date", "runs_scored"]].tail(10).sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )