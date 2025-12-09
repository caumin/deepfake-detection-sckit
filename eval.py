import pandas as pd
import argparse
import joblib
import os
import json
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from features import get_feature_names

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument('--csv', type=str, required=True, help="Path to the test features CSV file.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model (.joblib file).")
    parser.add_argument('--report_dir', type=str, default='out/report', help="Directory to save evaluation report and plots.")
    parser.add_argument('--feature-importance', action='store_true', help="If set, generate and save a feature importance plot for tree-based models.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.report_dir, exist_ok=True)

    # --- Load Data ---
    print(f"Loading test data from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Dynamically get feature columns
    try:
        feature_columns = get_feature_names(feature_size_spec1d=len([c for c in df.columns if c.startswith('spec_bin_')]))
        X_test = df[feature_columns]
        y_test = df['label']
    except Exception as e:
        print(f"Error preparing data: {e}")
        print("Ensure the CSV file contains the expected feature columns (e.g., 'spec_bin_...').")
        return
        
    print(f"Found {len(X_test)} samples in the test set.")

    # --- Load Model ---
    print(f"Loading model from {args.model}...")
    try:
        model = joblib.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Evaluation ---
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for AUROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1] # Probability of the 'REAL' class
        auroc = roc_auc_score(y_test, y_prob)
    else: # Models like SVC with linear kernel might not have predict_proba by default
        y_prob = None
        auroc = "N/A (model does not support predict_proba)"

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], output_dict=True)
    
    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc if isinstance(auroc, str) else f'{auroc:.4f}'}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    # --- Save Report ---
    report = {
        'model_path': args.model,
        'test_data_path': args.csv,
        'accuracy': accuracy,
        'auroc': auroc,
        'classification_report': report_dict
    }

    report_path = os.path.join(args.report_dir, 'report.json')
    print(f"\nSaving detailed report to {report_path}...")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print("Report saved.")

    # --- Save Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FAKE', 'REAL'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f'Confusion Matrix for {os.path.basename(args.model)}')
    
    cm_path = os.path.join(args.report_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")
    plt.close()

    # --- Feature Importance Plot ---
    if args.feature_importance:
        if hasattr(model, 'feature_importances_'):
            print("Generating feature importance plot...")
            
            # Get feature importances
            importances = model.feature_importances_
            feature_names = X_test.columns
            
            # Create a DataFrame for better handling
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            
            # Sort by importance and select top 20
            top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
            
            # Plot
            plt.figure(figsize=(12, 10))
            plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top 20 Feature Importances for {os.path.basename(args.model)}')
            plt.gca().invert_yaxis()  # Display the most important feature at the top
            plt.tight_layout()
            
            # Save the plot
            importance_plot_path = os.path.join(args.report_dir, 'feature_importance.png')
            plt.savefig(importance_plot_path)
            print(f"Feature importance plot saved to {importance_plot_path}")
            plt.close()
        else:
            print("Model does not have 'feature_importances_' attribute. Skipping plot.")

if __name__ == '__main__':
    main()
