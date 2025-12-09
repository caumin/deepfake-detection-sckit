import os
import subprocess
import argparse
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Runs a shell command and logs its output."""
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            logging.info("STDOUT:\n" + result.stdout)
        if result.stderr:
            logging.warning("STDERR:\n" + result.stderr)
    except FileNotFoundError as e:
        logging.error(f"Error: Command not found. Ensure '{command[0]}' is in your PATH. Details: {e}")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}.")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)
        raise

def find_real_fake_dirs(base_path):
    """Finds 'real' and 'fake' directories, accommodating case variations."""
    # Check for lowercase 'real'/'fake'
    real_dir = os.path.join(base_path, 'real')
    fake_dir = os.path.join(base_path, 'fake')
    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        return real_dir, fake_dir

    # Check for uppercase 'REAL'/'FAKE'
    real_dir_upper = os.path.join(base_path, 'REAL')
    fake_dir_upper = os.path.join(base_path, 'FAKE')
    if os.path.isdir(real_dir_upper) and os.path.isdir(fake_dir_upper):
        return real_dir_upper, fake_dir_upper

    raise FileNotFoundError(f"Could not find 'real'/'fake' or 'REAL'/'FAKE' directories in {base_path}")


def main(args):
    """Main pipeline orchestration function."""
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    output_dir = f"{dataset_name}_output"
    
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    # Define paths for train/test features
    train_csv = os.path.join(output_dir, f"{dataset_name}_train_features.csv")
    test_csv = os.path.join(output_dir, f"{dataset_name}_test_features.csv")

    # --- 1. Feature Extraction ---
    skip_extraction = False
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        logging.info(f"Feature files already exist: {train_csv} and {test_csv}")
        answer = input("Do you want to re-extract features? (y/N): ").lower().strip()
        if answer != 'y':
            logging.info("Skipping feature extraction.")
            skip_extraction = True

    if not skip_extraction:
        try:
            # For training data
            logging.info("--- Starting Feature Extraction for Training Data ---")
            train_base_path = os.path.join(args.data_dir, 'train')
            real_train_dir, fake_train_dir = find_real_fake_dirs(train_base_path)
            extract_cmd_train = [
                'python', 'extract_features.py',
                '--real_dir', real_train_dir,
                '--fake_dir', fake_train_dir,
                '--out_csv', train_csv
            ]
            run_command(extract_cmd_train)

            # For testing data
            logging.info("--- Starting Feature Extraction for Test Data ---")
            test_base_path = os.path.join(args.data_dir, 'test')
            real_test_dir, fake_test_dir = find_real_fake_dirs(test_base_path)
            extract_cmd_test = [
                'python', 'extract_features.py',
                '--real_dir', real_test_dir,
                '--fake_dir', fake_test_dir,
                '--out_csv', test_csv
            ]
            run_command(extract_cmd_test)
            logging.info("--- Feature Extraction Complete ---")

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"Failed during feature extraction. Aborting pipeline. Error: {e}")
            return
    else:
        # We need to make sure the subsequent steps can run.
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
            logging.error("Feature files not found, and extraction was skipped. Aborting pipeline.")
            logging.error(f"Expected train features at: {train_csv}")
            logging.error(f"Expected test features at: {test_csv}")
            return

    # --- 2. & 3. Training and Evaluation Loop ---
    supported_models = ['linsvm', 'rbfsvm', 'xgb', 'rf']
    logging.info(f"--- Starting Training and Evaluation for models: {supported_models} ---")

    for model_name in supported_models:
        try:
            logging.info(f"--- Processing model: {model_name.upper()} ---")
            
            # Define model and report paths
            model_path = os.path.join(output_dir, f"{dataset_name}_{model_name}.joblib")
            report_dir = os.path.join(output_dir, f"report_{dataset_name}_{model_name}")
            os.makedirs(report_dir, exist_ok=True) # eval.py might need the folder to exist
            report_path = os.path.join(report_dir, 'report.json')


            # Train the model
            logging.info(f"Training {model_name}...")
            train_cmd = [
                'python', 'train.py',
                '--csv', train_csv,
                '--model', model_name,
                '--out_model', model_path
            ]
            run_command(train_cmd)

            # Evaluate the model
            logging.info(f"Evaluating {model_name}...")
            eval_cmd = [
                'python', 'eval.py',
                '--csv', test_csv,
                '--model', model_path,
                '--report_dir', report_dir
            ]
            
            # Add feature importance flag for supported models
            if args.feature_importance and model_name in ['rf', 'xgb']:
                eval_cmd.append('--feature-importance')

            run_command(eval_cmd)
            logging.info(f"--- Finished processing model: {model_name.upper()} ---")

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"Pipeline failed for model '{model_name}'. Skipping to next model. Error: {e}")
            continue # Move to the next model
            
    logging.info("--- Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Orchestration pipeline for image authenticity detection.")
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path to the dataset directory (e.g., 'data/CIFAKE')."
    )
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help="If set, generate feature importance plots for tree-based models (rf, xgb)."
    )
    args = parser.parse_args()
    main(args)
