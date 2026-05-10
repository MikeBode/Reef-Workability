# Trains and evaluates classification models (RandomForest, LogisticRegression, SVM, XGBoost)
# using temporal cross-validation with SMOTE oversampling or class-weighting to predict
# reef survey workability from wave height and wind component features.

from re import search

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
import ml_insights as mli
import xgboost as xgb
import pathlib
import warnings
warnings.filterwarnings('ignore')


def preprocess_and_combine_data(data_directory: pathlib.Path):
    successful_visits = pd.read_csv(data_directory / 'surveyData[63]WithWindWaveData_Final.csv')
    failed_visits = pd.read_csv(data_directory / 'cots_withWindWaveData.csv')

    successful_visits['date'] = pd.to_datetime(successful_visits['date'])
    if 'Date' in failed_visits.columns:
        failed_visits['date'] = pd.to_datetime(failed_visits['Date'])
        failed_visits.drop('Date', axis=1, inplace=True)
    else:
        failed_visits['date'] = pd.to_datetime(failed_visits['date'])

    successful_visits['visit_status'] = 'Successful'
    failed_visits['visit_status'] = 'Failed'

    column_mapping = {
        'Reef': 'reefName',
        'Vessel': 'source',
        'Region': 'region',
    }

    failed_visits.rename(
        columns={k: v for k, v in column_mapping.items() if k in failed_visits.columns},
        inplace=True
    )

    successful_visits['month'] = successful_visits['date'].dt.month
    failed_visits['month'] = failed_visits['date'].dt.month
    successful_visits['year'] = successful_visits['date'].dt.year
    failed_visits['year'] = failed_visits['date'].dt.year
    successful_visits['quarter'] = successful_visits['date'].dt.quarter
    failed_visits['quarter'] = failed_visits['date'].dt.quarter

    common_columns = [
        'date', 'reefName', 'x', 'y', 'wave_height', 'u_wind', 'v_wind',
        'visit_status', 'month', 'year', 'quarter', 'day_of_year'
    ]

    if 'Days lost' in failed_visits.columns:
        failed_visits['days_lost'] = failed_visits['Days lost']
        common_columns.append('days_lost')
        successful_visits['days_lost'] = 0

    for col in common_columns:
        if col not in successful_visits.columns:
            successful_visits[col] = np.nan
        if col not in failed_visits.columns:
            failed_visits[col] = np.nan

    combined_df = pd.concat([
        successful_visits[common_columns],
        failed_visits[common_columns]
    ]).reset_index(drop=True)

    combined_df['wind_magnitude'] = np.sqrt(combined_df['u_wind']**2 + combined_df['v_wind']**2)
    combined_df = combined_df.dropna(subset=['wave_height', 'u_wind', 'v_wind'])
    combined_df = combined_df.sort_values('date').reset_index(drop=True)

    return combined_df

def train_and_evaluate_probability_models(combined_df):
    # We aim to train calibrated probability models, then evaluate them by their Brier Skill Scores.
    features = ['wave_height', 'u_wind', 'v_wind', 'wind_magnitude', 'month']

    train_data, test_data = train_test_split(combined_df, test_size=0.25, stratify=combined_df["visit_status"])

    SUCCESSFUL_DIVE_CLASS = 1
    FAILED_DIVE_CLASS = 0

    X_train = train_data[features]
    y_train = (train_data['visit_status'] == 'Successful').astype(int)
    X_test = test_data[features]
    y_test = (test_data['visit_status'] == 'Successful').astype(int)

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        print("Error: Insufficient training data")
        return None

    train_class_counts = pd.Series(y_train).value_counts()
    test_class_counts = pd.Series(y_test).value_counts()

    print(f"\nTraining set class distribution:")
    print(f"Failed visits ({FAILED_DIVE_CLASS}): {train_class_counts.get(FAILED_DIVE_CLASS, 0)} ({train_class_counts.get(FAILED_DIVE_CLASS, 0) / len(y_train) * 100:.1f}%)")
    print(f"Successful visits ({SUCCESSFUL_DIVE_CLASS}): {train_class_counts.get(SUCCESSFUL_DIVE_CLASS, 0)} ({train_class_counts.get(SUCCESSFUL_DIVE_CLASS, 0) / len(y_train) * 100:.1f}%)")

    model_params = {
        "RandomForest": {
            'classifier__n_estimators': [1000],     # There's no need to try a small number of trees, since random forest performance should only ever improve with additional trees.
            'classifier__max_depth': [None, 5],
            'classifier__min_samples_split': [2, 3, 5],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        "LogisticRegression": {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
        },
        "SVM": {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'classifier__kernel': ['rbf', 'sigmoid']
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
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=10000),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    print(f"\n{'=' * 80}")

    def model_to_pipeline(model_name):
        if model_name in ['SVM', 'LogisticRegression']:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', models[model_name])
            ])
        else:
            return Pipeline([
                ('classifier', models[model_name])
            ])

    for model_name, _ in models.items():
        print(f"\nTraining {model_name}")
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
        
        cv_preds_train = mli.cv_predictions(search, X_train, y_train, clone_model=True)
        calib = mli.SplineCalib()
        calib.fit(cv_preds_train, y_train)

        search.fit(X_train, y_train)

        test_y_probs = calib.calibrate(search.predict_proba(X_test))[:, 1]
        brier_score = brier_score_loss(y_test, test_y_probs)
        brief_reference_score = brier_score_loss(y_test, [y_test.mean()] * len(y_test))
        brier_skill_score = 1 - (brier_score / brief_reference_score) if brief_reference_score > 0 else 0
        print(f"{model_name} Brier Skill Score: {brier_skill_score:.8f}")


def main(data_directory: pathlib.Path):
    print("Preprocessing and combining datasets...")
    combined_df = preprocess_and_combine_data(data_directory)

    print("Training calibrated probability models.")
    train_and_evaluate_probability_models(combined_df, use_fancy_calibration=True)

if __name__ == "__main__":
    main(pathlib.Path("Data"))