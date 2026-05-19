import pathlib
import pandas as pd
import models.model_training
import pickle

def predict_success_prob_for_reef_visits(best_model_path: pathlib.Path, reef_data: pd.DataFrame):
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)

    X = reef_data[models.model_training.MODEL_FEATURES]
    X_unique = X.drop_duplicates()

    predicted_base_probabilities = best_model.predict_proba_base(X_unique)
    predicted_probabilities = best_model.proba_to_success_prob(predicted_base_probabilities)

    base_prob_map = dict(zip(map(tuple, X_unique.values), predicted_base_probabilities[:,1]))
    predicted_prob_map = dict(zip(map(tuple, X_unique.values), predicted_probabilities))
    predicted_base_probabilities = X.apply(lambda row: base_prob_map[tuple(row.values)], axis=1)
    predicted_probs = X.apply(lambda row: predicted_prob_map[tuple(row.values)], axis=1)

    reef_data['predicted_dataset_probability'] = predicted_base_probabilities
    reef_data['predicted_success_prob'] = predicted_probs

    return reef_data