# Trains and evaluates classification models (RandomForest, LogisticRegression, SVM, XGBoost)
# using temporal cross-validation with SMOTE oversampling or class-weighting to predict
# reef survey workability from wave height and wind component features.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from models.model_training import ModelAndCalibrationCurve, AbsWind
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
import ml_insights as mli
import xgboost as xgb
import pathlib
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
    
MODEL_FEATURES = ['wave_height', 'u_wind', 'v_wind', 'wind_magnitude', 'month']

def train_and_evaluate_probability_models(combined_df, model_save_path: pathlib.Path):
    # We aim to train calibrated probability models, then evaluate them by their Brier Skill Scores.
    train_data, test_data = train_test_split(combined_df, test_size=0.25, stratify=combined_df["was_successful"], random_state=42)

    SUCCESSFUL_DIVE_CLASS = 1
    FAILED_DIVE_CLASS = 0

    X_train = train_data[MODEL_FEATURES]
    y_train = (train_data['was_successful']).astype(int)
    X_test = test_data[MODEL_FEATURES]
    y_test = (test_data['was_successful']).astype(int)

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        print("Error: Insufficient training data")
        return None

    train_class_counts = pd.Series(y_train).value_counts()
    test_class_counts = pd.Series(y_test).value_counts()

    print(f"\nTraining set class distribution:")
    print(f"Failed visits ({FAILED_DIVE_CLASS}): {train_class_counts.get(FAILED_DIVE_CLASS, 0)} ({train_class_counts.get(FAILED_DIVE_CLASS, 0) / len(y_train) * 100:.1f}%)")
    print(f"Successful visits ({SUCCESSFUL_DIVE_CLASS}): {train_class_counts.get(SUCCESSFUL_DIVE_CLASS, 0)} ({train_class_counts.get(SUCCESSFUL_DIVE_CLASS, 0) / len(y_train) * 100:.1f}%)")

    print(f"\nTest set class distribution:")
    print(f"Failed visits ({FAILED_DIVE_CLASS}): {test_class_counts.get(FAILED_DIVE_CLASS, 0)} ({test_class_counts.get(FAILED_DIVE_CLASS, 0) / len(y_test) * 100:.1f}%)")
    print(f"Successful visits ({SUCCESSFUL_DIVE_CLASS}): {test_class_counts.get(SUCCESSFUL_DIVE_CLASS, 0)} ({test_class_counts.get(SUCCESSFUL_DIVE_CLASS, 0) / len(y_test) * 100:.1f}%)")

    model_params = {
        "RandomForest": {
            'classifier__n_estimators': [1000],     # There's no need to try a small number of trees, since random forest performance should only ever improve with additional trees.
            'classifier__max_depth': [None, 5],
            'classifier__min_samples_split': [2, 3, 5],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        "XGBoost": {
            'classifier__n_estimators': [10, 20, 50, 100, 200, 500],    # XGBoost can overfit with too many trees.
            'classifier__max_depth': [2,3,4,5],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0]
        }
    }

    models = {
        #'RandomForest': RandomForestClassifier(random_state=42, monotonic_cst={"wave_height": -1, "wind_magnitude": -1, "west_wind": -1, "east_wind": -1, "south_wind": -1, "north_wind": -1}),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', monotone_constraints={"wave_height": -1, "wind_magnitude": -1, "west_wind": -1, "east_wind": -1, "south_wind": -1, "north_wind": -1})
    }

    print(f"\n{'=' * 80}")

    def model_to_pipeline(model_name):        
        return Pipeline([
            ('directional_winds', AbsWind()),
            ('classifier', models[model_name])
        ])
    
    def train_model_with_name(model_name, data_X, data_Y):
        print(f"\nTraining {model_name} on {len(data_X)} samples with {data_Y.sum()} positive cases")
        pipeline = model_to_pipeline(model_name)

        rs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_params[model_name],
            n_jobs=-1,
            verbose=0,
            cv=rs,
            scoring="neg_log_loss"
        )
        
        cv_preds_train = mli.cv_predictions(search, data_X, data_Y, clone_model=True)
        calib = mli.SplineCalib()
        calib.fit(cv_preds_train, data_Y)
        search.fit(data_X, data_Y)
        model = ModelAndCalibrationCurve(model_name, search.best_estimator_, calib)

        return model

    def get_brier_skill_score(model):
        test_y_probs = model.predict_proba(X_test)[:, 1]
        brier_score = brier_score_loss(y_test, test_y_probs)
        brief_reference_score = brier_score_loss(y_test, [y_test.mean()] * len(y_test))
        return 1 - (brier_score / brief_reference_score) if brief_reference_score > 0 else 0

    best_brier_skill_score = float('-inf')
    best_model_name = None

    for model_name, _ in models.items():
        model = train_model_with_name(model_name, X_train, y_train)
        brier_skill_score = get_brier_skill_score(model)

        print(f"{model_name} Brier Skill Score: {brier_skill_score:.8f}")

        if brier_skill_score > best_brier_skill_score:
            best_brier_skill_score = brier_skill_score
            best_model_name = model_name
    
    # Retraining our best model on the full dataset.
    best_model = train_model_with_name(best_model_name, combined_df[MODEL_FEATURES], combined_df['was_successful'].astype(int))
    print(f"Saving best model: {best_model}, with Testing Brier Skill Score: {best_brier_skill_score:.8f}, into {model_save_path}")

    # Quick sanity test
    print(f"Sanity test, Best Model Brier (On test) = {get_brier_skill_score(best_model):.8f}")
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)