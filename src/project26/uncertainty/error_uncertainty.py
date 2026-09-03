def extract_errors(uncertainty_results):
    return uncertainty_results[uncertainty_results["true_state"] != uncertainty_results["predicted_state"]].copy()
