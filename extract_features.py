import os
import glob
import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
from tqdm import tqdm
import argparse
import pandas as pd

from features import extract_all_features, get_feature_names

def extract_features_from_dir(real_dir, fake_dir, img_size=256, reencode_jpeg=None, feature_bins=128):
    """
    Extracts features from all images in the real and fake directories.
    Based on gemini.md guidance.
    """
    all_features = []
    labels = []
    paths = []

    # --- Process Real and Fake Images ---
    image_dirs = {
        'REAL': (real_dir, 1),
        'FAKE': (fake_dir, 0)
    }

    for label_str, (directory, label_int) in image_dirs.items():
        print(f"Processing {label_str} images from: {directory}")
        
        # Using recursive glob to find all images
        image_paths = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)

        if not image_paths:
            print(f"Warning: No images found in {directory}")
            continue

        for path in tqdm(image_paths, desc=f"Extracting {label_str} features"):
            try:
                img_array = np.fromfile(path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # 1. Preprocessing (as per gemini.md)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                if reencode_jpeg:
                    _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), reencode_jpeg])
                    img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)

                # 2. Feature Extraction
                feature_vector = extract_all_features(img, feature_size_spec1d=feature_bins)
                
                if feature_vector is not None:
                    all_features.append(feature_vector)
                    labels.append(label_int)
                    paths.append(path)

            except Exception as e:
                print(f"Skipping {path} due to error: {e}")
            
    # Combine into a pandas DataFrame as suggested by gemini.md for clarity
    feature_names = get_feature_names(feature_size_spec1d=feature_bins)
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
        feature_bins=args.bins
    )
    
    print(f"Saving {len(features_df)} features to {args.out_csv}...")
    features_df.to_csv(args.out_csv, index=False)
    print("Done.")

if __name__ == '__main__':
    main()

