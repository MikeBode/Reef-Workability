# Trains and evaluates classification models (RandomForest, LogisticRegression, SVM, XGBoost)
# using temporal cross-validation with SMOTE oversampling or class-weighting to predict
# reef survey workability from wave height and wind component features.

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
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

def train_and_evaluate_models(combined_df, strategy='oversampling'):
    features = ['wave_height', 'u_wind', 'v_wind', 'wind_magnitude', 'month']

    train_data, test_data = train_test_split(combined_df, test_size=0.25, stratify=combined_df["visit_status"])

    SUCCESSFUL_DIVE_CLASS = 0
    FAILED_DIVE_CLASS = 1

    X_train = train_data[features]
    y_train = (train_data['visit_status'] != 'Successful').astype(int)
    X_test = test_data[features]
    y_test = (test_data['visit_status'] != 'Successful').astype(int)

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
            'classifier__max_depth': [None, 1, 2, 5, 10],
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
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
              # Rule of thumb
        }
    }

    if strategy == "weighted":
        for model in model_params:
            if model != "XGBoost":
                model_params[model]["classifier__class_weight"] = [{FAILED_DIVE_CLASS: sum(y_train == SUCCESSFUL_DIVE_CLASS), SUCCESSFUL_DIVE_CLASS: sum(y_train == FAILED_DIVE_CLASS)}]
            else:
                model_params[model]['classifier__scale_pos_weight'] = [1, (len(y_train) - sum(y_train))/sum(y_train)]

    # The lambda allows us to reconstruct the model with the optimal parametres found during the grid search for threshold optimization.
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=10000),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    best_model = None
    best_score = 0
    best_model_name = ""
    model_results = []

    print(f"\n{'=' * 80}")
    print(f"MODEL EVALUATION RESULTS - {strategy.upper()} STRATEGY")
    print(f"{'=' * 80}")

    def model_to_pipeline(model_name):
        if strategy == 'oversampling':
            if model_name in ['SVM', 'LogisticRegression']:
                return Pipeline([
                    ('sampler', SMOTE(random_state=42)),
                    ('scaler', StandardScaler()),
                    ('classifier', models[model_name]),
                ])
            else:
                return Pipeline([
                    ('sampler', SMOTE(random_state=42)),
                    ('classifier', models[model_name])
                ])
        else:
            if model_name in ['SVM', 'LogisticRegression']:
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', models[model_name])
                ])
            else:
                return Pipeline([
                    ('classifier', models[model_name])
                ])

    for model_name, model in models.items():
        print(f"\nTraining {model_name} with {strategy}...")
        pipeline = model_to_pipeline(model_name)

        rs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_params[model_name],
            cv=rs,
            n_jobs=-1,
            verbose=0,
            scoring="neg_log_loss"
        )

        try:
            search.fit(X_train, y_train)
            
            y_pred = search.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision_successful = precision_score(y_test, y_pred, pos_label=SUCCESSFUL_DIVE_CLASS, zero_division=0)
            recall_successful = recall_score(y_test, y_pred, pos_label=SUCCESSFUL_DIVE_CLASS, zero_division=0)
            precision_failed = precision_score(y_test, y_pred, pos_label=FAILED_DIVE_CLASS, zero_division=0)
            recall_failed = recall_score(y_test, y_pred, pos_label=FAILED_DIVE_CLASS, zero_division=0)
            f1_failed = f1_score(y_test, y_pred, pos_label=FAILED_DIVE_CLASS, zero_division=0)
            f1_successful = f1_score(y_test, y_pred, pos_label=SUCCESSFUL_DIVE_CLASS, zero_division=0)

            result = {
                'model_name': model_name,
                'cv_score': search.best_score_,
                'accuracy': accuracy,
                'precision_successful': precision_successful,
                'recall_successful': recall_successful,
                'precision_failed': precision_failed,
                'recall_failed': recall_failed,
                'f1_failed': f1_failed,
                'f1_successful': f1_successful,
                'best_params': search.best_params_,
                'model_object': search.best_estimator_
            }

            model_results.append(result)

            print(f"\n{model_name} Results:")
            print(f"CV Score: {search.best_score_:.4f}")
            print(f"Overall Test Accuracy: {accuracy:.4f}")
            print(f"Precision (Successful): {precision_successful:.4f}")
            print(f"Recall (Successful): {recall_successful:.4f}")
            print(f"Precision (Failed): {precision_failed:.4f}")
            print(f"Recall (Failed): {recall_failed:.4f}")
            print(f"F1-score (Failed): {f1_failed:.4f}")
            print(f"F1-score (Successful): {f1_successful:.4f}")
            print(f"Best parameters: {search.best_params_}")

            print(f"\nClassification Report for {model_name}:")
            print(classification_report(y_test, y_pred,
                                        target_names=['Failed', 'Successful'] if SUCCESSFUL_DIVE_CLASS == 1 else ['Successful', 'Failed'],
                                        digits=4))
            print("-" * 60)

            if search.best_score_ > best_score:
                best_model = search.best_estimator_
                best_model_name = model_name
                best_score = search.best_score_

        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue

    print(f"\n{'=' * 80}")
    print(f"SUMMARY - {strategy.upper()} STRATEGY")
    print(f"{'=' * 80}")

    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'CV Score': r['cv_score'],
        'Accuracy': r['accuracy'],
        'Precision (Success)': r['precision_successful'],
        'Recall (Success)': r['recall_successful'],
        'Precision (Failed)': r['precision_failed'],
        'Recall (Failed)': r['recall_failed'],
        'F1 (Failed)': r['f1_failed'],
        'F1 (Successful)': r['f1_successful']
    } for r in model_results])

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Score: {best_score:.4f}")

    if best_model is not None:
        model_info = {
            'model': best_model,
            'features': features,
            'model_name': best_model_name,
            'cv_score (log loss)': best_score,
            'strategy': strategy,
            'all_results': model_results,
            'training_info': {
                'total_samples': len(combined_df),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': features,
                'train_class_distribution': train_class_counts.to_dict(),
                'test_class_distribution': test_class_counts.to_dict(),
                'train_date_range': f"{train_data['date'].min()} to {train_data['date'].max()}",
                'test_date_range': f"{test_data['date'].min()} to {test_data['date'].max()}"
            }
        }

        return model_info

    return None


def main(data_directory: pathlib.Path):
    print("Preprocessing and combining datasets...")
    combined_df = preprocess_and_combine_data(data_directory)

    print(f"Combined dataset created with {len(combined_df)} entries.")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

    print("\n" + "=" * 80)
    print("TRAINING MODELS WITH NO ADJUSTMENT STRATEGY")
    model_none_info = train_and_evaluate_models(combined_df, strategy='none')

    print("\n" + "=" * 80)
    print("TRAINING MODELS WITH OVERSAMPLING STRATEGY")
    print("=" * 80)

    model_a_info = train_and_evaluate_models(combined_df, strategy='oversampling')

    print("\n" + "=" * 80)
    print("TRAINING MODELS WITH WEIGHTED STRATEGY")
    print("=" * 80)

    model_b_info = train_and_evaluate_models(combined_df, strategy='weighted')

    if model_a_info:
        with open('model_a_oversampling.pkl', 'wb') as f:
            pickle.dump(model_a_info, f)
        print(f"\nModel A (Oversampling) saved as 'model_a_oversampling.pkl'")
        print(f"Best model: {model_a_info['model_name']}")
        print(f"Optimal threshold: {model_a_info['threshold']:.3f}")

    if model_b_info:
        with open('model_b_weighted.pkl', 'wb') as f:
            pickle.dump(model_b_info, f)
        print(f"\nModel B (Weighted) saved as 'model_b_weighted.pkl'")
        print(f"Best model: {model_b_info['model_name']}")
        print(f"Optimal threshold: {model_b_info['threshold']:.3f}")

    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON")
    print(f"{'=' * 80}")

    if model_a_info and model_b_info:
        print(f"Model A (Oversampling): {model_a_info['model_name']} - Score: {model_a_info['business_score']:.4f}")
        print(f"Model B (Weighted): {model_b_info['model_name']} - Score: {model_b_info['business_score']:.4f}")

        if model_a_info['business_score'] > model_b_info['business_score']:
            print(f"\nOverall Best: Model A ({model_a_info['model_name']} with oversampling)")
        else:
            print(f"\nOverall Best: Model B ({model_b_info['model_name']} with weighting)")

    print(f"\nUsage:")
    print(f"1. Load model: model_info = pickle.load(open('model_a_oversampling.pkl', 'rb'))")
    print(f"2. Get probabilities: proba = model_info['model'].predict_proba(X)[:, 1]")
    print(f"3. Apply threshold: predictions = proba >= model_info['threshold']")
    print(f"4. Interpret: 1 = Successful visit, 0 = Failed visit")


if __name__ == "__main__":
    main(pathlib.Path("Data"))