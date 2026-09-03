from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project26.inference import InferencePredictor


def main():
    # Synthetic values demonstrate the pipeline only; they do not describe a real person.
    example_features = {
        "Depression_Score": 4.0,
        "Anxiety_Score": 5.0,
        "Stress_Score": 6.0,
        "Sleep_Quality": 7.0,
        "Social_Engagement": 6.0,
        "Daily_App_Usage_Min": 180.0,
        "Typing_Speed_WPM": 52.0,
        "Session_Frequency": 4.0,
        "Idle_Time_Min": 35.0,
        "Facial_Emotion_Variance": 0.4,
        "Eye_Blink_Rate": 18.0,
        "Smile_Intensity": 0.6,
        "Head_Motion_Index": 0.3,
        "MFCC_Mean": -12.0,
        "MFCC_Variance": 8.0,
        "Pitch_Mean": 190.0,
        "Speech_Rate": 4.5,
        "Heart_Rate_BPM": 72.0,
        "HRV_Index": 48.0,
        "Skin_Temperature": 33.0,
        "GSR_Level": 2.0,
    }

    result = InferencePredictor(root=ROOT).predict(example_features)
    print("\nPrediction")
    print("----------")
    print("Predicted state:", result["predicted_state"])
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Entropy uncertainty: {result['entropy_uncertainty']:.2f}")
    print(f"Normalized uncertainty: {result['normalized_uncertainty']:.2f}")
    print("\nClass probabilities:")
    for name, probability in result["class_probabilities"].items():
        print(f"{name}: {probability:.4f}")


if __name__ == "__main__":
    main()