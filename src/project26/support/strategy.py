def classify_uncertainty(confidence, uncertainty):
    if confidence >= 0.80 and uncertainty < 0.25:
        return "Low Uncertainty"
    elif confidence >= 0.60 and uncertainty < 0.50:
        return "Medium Uncertainty"
    return "High Uncertainty"


def select_support_strategy(predicted_state, confidence, uncertainty):
    if confidence < 0.60 or uncertainty >= 0.50:
        return "Clarification and cautious support"
    if predicted_state == "Healthy":
        return "Positive reinforcement"
    elif predicted_state == "Mild Stress":
        return "Emotional support and relaxation guidance"
    elif predicted_state == "Moderate Stress":
        return "Coping strategies and practical guidance"
    elif predicted_state == "Severe Stress":
        return "Strong emotional support and safety-oriented guidance"
    return "General supportive response"
