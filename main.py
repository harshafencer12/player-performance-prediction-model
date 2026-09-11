from src.predict import predict_player_performance

def main():
    print("=== Cricket Player Performance Predictor ===")
    player_name = input("Enter player name: ").strip()

    try:
        predicted_runs = predict_player_performance(player_name)
        print(f"\nPredicted runs for {player_name}: {predicted_runs}")
    except ValueError as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()