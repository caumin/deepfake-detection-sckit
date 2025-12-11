import os
import glob
import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
from tqdm import tqdm
import argparse
import pandas as pd

from joblib import Parallel, delayed
from features import extract_all_features, get_feature_names

def process_image(path, label_int, img_size, reencode_jpeg, spec_bins, color_bins, residual_mode):
    """
    Worker function to process a single image.
    Performs loading, preprocessing, and feature extraction.
    """
    try:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # 1. Preprocessing
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        if reencode_jpeg:
            _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), reencode_jpeg])
            img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)

        # 2. Feature Extraction
        feature_vector = extract_all_features(img, feature_size_spec1d=spec_bins, color_hist_bins=color_bins, residual_mode=residual_mode)
        
        if feature_vector is not None:
            return feature_vector, label_int, path
        return None
    except Exception as e:
        print(f"Skipping {path} due to error: {e}")
        return None

def extract_features_from_dir(real_dir, fake_dir, img_size=256, reencode_jpeg=None, spec_bins=128, color_bins=32, n_jobs=-1, residual_mode='denoise'):
    """
    Extracts features from all images in parallel using joblib.
    """
    tasks = []
    
    # --- Prepare list of tasks ---
    image_dirs = {
        'REAL': (real_dir, 1),
        'FAKE': (fake_dir, 0)
    }

    for label_str, (directory, label_int) in image_dirs.items():
        print(f"Preparing tasks for {label_str} images from: {directory}")
        image_paths = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)

        if not image_paths:
            print(f"Warning: No images found in {directory}")
            continue

        for path in image_paths:
            tasks.append(delayed(process_image)(path, label_int, img_size, reencode_jpeg, spec_bins, color_bins, residual_mode))

    # --- Run tasks in parallel ---
    print(f"\nExtracting features from {len(tasks)} images using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
    results = Parallel(n_jobs=n_jobs)(tqdm(tasks))
    
    # --- Process results ---
    all_features = []
    labels = []
    paths = []
    for res in results:
        if res is not None:
            feature_vector, label_int, path = res
            all_features.append(feature_vector)
            labels.append(label_int)
            paths.append(path)
            
    if not all_features:
        print("Error: No features were extracted. Check image paths and file integrity.")
        return pd.DataFrame()
            
    # Combine into a pandas DataFrame
    feature_names = get_feature_names(feature_size_spec1d=spec_bins, color_hist_bins=color_bins)
    df = pd.DataFrame(all_features, columns=feature_names)
    df['label'] = labels
    df['path'] = paths
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Extract features from real and fake image datasets.")
    parser.add_argument('--real_dir', type=str, required=True, help="Directory for real images.")
    parser.add_argument('--fake_dir', type=str, required=True, help="Directory for fake images.")
    parser.add_argument('--out_csv', type=str, default='out/features.csv', help="Path to save the output CSV file.")
    parser.add_argument('--img_size', type=int, default=256, help="Size to resize images to (e.g., 256).")
    parser.add_argument('--reencode_jpeg', type=int, default=95, help="JPEG quality for re-encoding (e.g., 95). Set to 0 to disable.")
    parser.add_argument('--bins', type=int, default=128, help="Number of bins for the 1D power spectrum feature.")
    parser.add_argument('--color_bins', type=int, default=32, help="Number of bins for the color saturation histogram.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs to run (-1 uses all available cores).")
    parser.add_argument('--residual_mode', type=str, default='denoise', choices=['denoise', 'highpass'], help="Method for noise residual calculation ('denoise' is slow, 'highpass' is fast).")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    # Handle reencode_jpeg=0 case
    reencode_jpeg_quality = args.reencode_jpeg if args.reencode_jpeg > 0 else None

    features_df = extract_features_from_dir(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        img_size=args.img_size,
        reencode_jpeg=reencode_jpeg_quality,
        spec_bins=args.bins,
        color_bins=args.color_bins,
        n_jobs=args.n_jobs,
        residual_mode=args.residual_mode
    )
    
    print(f"Saving {len(features_df)} features to {args.out_csv}...")
    if not features_df.empty:
        features_df.to_csv(args.out_csv, index=False)
    print("Done.")

if __name__ == '__main__':
    main()

