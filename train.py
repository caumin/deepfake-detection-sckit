import pandas as pd
import argparse
import joblib
import os
from features import get_feature_names

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_model(model_name, n_jobs=-1):
    """Returns a scikit-learn model instance based on its name."""
    if model_name == 'linsvm':
        return make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, class_weight='balanced', random_state=42))
    elif model_name == 'rbfsvm':
        return make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
    elif model_name == 'rf':
        return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=n_jobs))
    elif model_name == 'xgb':
        return XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=n_jobs)
    elif model_name == 'logreg':
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=42, n_jobs=n_jobs))
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

def main():
    parser = argparse.ArgumentParser(description="Train a model on extracted features.")
    parser.add_argument('--csv', type=str, required=True, help="Path to the features CSV file.")
    parser.add_argument('--model', type=str, default='rf', choices=['linsvm', 'rbfsvm', 'rf', 'xgb', 'logreg'], help="Model to train.")
    parser.add_argument('--out_model', type=str, default='out/model.joblib', help="Path to save the trained model.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    print(f"Loading features from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Dynamically get feature columns from the features module
    feature_columns = get_feature_names(feature_size_spec1d=len([c for c in df.columns if c.startswith('spec_bin_')]))
    X = df[feature_columns]
    y = df['label']
    
    print(f"Found {len(feature_columns)} feature columns.")
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.seed,
        stratify=y # Ensure same class distribution in train/test
    )
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # Get and train the model
    print(f"Training {args.model} model...")
    model = get_model(args.model)
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model on the validation set
    print("\n--- Validation Results ---")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['FAKE', 'REAL']))
    
    # Save the trained model
    print(f"\nSaving model to {args.out_model}...")
    joblib.dump(model, args.out_model)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
