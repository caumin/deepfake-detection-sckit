import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift

# Feature Engineering Implementations based on gemini.md

# ------------------------------------------------------------------------------
# (A) 1D Power Spectrum + Azimuthal Integration
# ------------------------------------------------------------------------------
def get_azimuthal_average(gray_img, feature_size=128):
    """
    Calculates the 1D power spectrum (azimuthal average) of a grayscale image.
    This is Feature Set (A) from gemini.md.
    """
    if gray_img is None:
        return None
    
    # FFT -> Power Spectrum
    f = fft2(gray_img)
    fshift = fftshift(f)
    power_spectrum = np.abs(fshift)**2
    
    # Calculate radial profile
    h, w = gray_img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    # Sum of spectrum values for each radius
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    # Number of pixels for each radius
    nr = np.bincount(r.ravel())
    
    # Avoid division by zero
    nr[nr == 0] = 1
    
    radial_profile = tbin / nr
    
    # Log scale is more stable and often more informative
    radial_profile = np.log(radial_profile + 1e-6)

    # Normalize feature vector length
    if len(radial_profile) >= feature_size:
        return radial_profile[:feature_size]
    else:
        # Pad if the profile is shorter than feature_size
        return np.pad(radial_profile, (0, feature_size - len(radial_profile)), 'mean')

# TODO: Implement other feature sets from gemini.md below
# (B) "스펙트럼 왜곡" 밴드 통계 (Upsampling artifacts)
# (C) Color Cues: Saturation 통계 + 채널 상관 구조
# (D) Noise residual 기반 특징

def extract_all_features(img, feature_size_spec1d=128):
    """
    Top-level function to extract and concatenate all feature sets.
    F(x) = [F_spec1D(x), F_specDistort(x), F_color(x), F_residual(x)]
    """
    # For now, only one feature set is implemented.
    # The function is designed to be extended.
    
    # Convert to grayscale for spectrum analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # (A) 1D Power Spectrum
    spec1d_features = get_azimuthal_average(gray_img, feature_size=feature_size_spec1d)
    
    # Concatenate all features (add others here as they are implemented)
    final_feature_vector = spec1d_features
    
    return final_feature_vector

def get_feature_names(feature_size_spec1d=128):
    """
    Returns a list of column names for the features.
    This helps keep train.py and predict.py in sync.
    """
    spec1d_names = [f'spec_bin_{i}' for i in range(feature_size_spec1d)]
    
    # Concatenate all feature names here
    final_feature_names = spec1d_names
    
    return final_feature_names
