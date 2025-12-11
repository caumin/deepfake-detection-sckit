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

# ------------------------------------------------------------------------------
# (B) "스펙트럼 왜곡" 밴드 통계 (Upsampling artifacts)
# ------------------------------------------------------------------------------
def get_spectral_distortion_features(power_spectrum_1d):
    """
    Calculates features based on spectral distortions, as per gemini.md (B).
    Starts with band energy ratios.
    """
    if power_spectrum_1d is None:
        return np.zeros(3) # Return a zero vector of the correct size

    feature_size = len(power_spectrum_1d)
    
    # Define frequency bands
    low_band_end = int(feature_size * 0.25) # First 25%
    mid_band_end = int(feature_size * 0.75) # Next 50%
    
    # Calculate band energies (using mean of log-power)
    E_low = np.mean(power_spectrum_1d[0:low_band_end])
    E_mid = np.mean(power_spectrum_1d[low_band_end:mid_band_end])
    E_high = np.mean(power_spectrum_1d[mid_band_end:])
    
    # Calculate energy ratios (add a small epsilon to avoid division by zero)
    eps = 1e-6
    ratio_high_low = E_high / (E_low + eps)
    ratio_mid_low = E_mid / (E_low + eps)
    ratio_high_mid = E_high / (E_mid + eps)
    
    return np.array([ratio_high_low, ratio_mid_low, ratio_high_mid])

# ------------------------------------------------------------------------------
# (C) Color Cues: Saturation 통계 + 채널 상관 구조
# ------------------------------------------------------------------------------
def get_color_cues_features(img_rgb, hist_bins=32):
    """
    Calculates color-based features, as per gemini.md (C).
    - Saturation statistics (histogram, mean, std)
    - RGB channel correlations
    """
    if img_rgb is None:
        return np.zeros(hist_bins + 2 + 3)

    # 1. Saturation Statistics
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    # Saturation histogram
    s_hist, _ = np.histogram(s_channel.ravel(), bins=hist_bins, range=[0, 256])
    s_hist = s_hist.astype(np.float32) / (s_hist.sum() + 1e-6) # Normalize
    
    # Saturation summary stats
    s_mean = np.mean(s_channel)
    s_std = np.std(s_channel)
    
    saturation_features = np.concatenate([s_hist, [s_mean, s_std]])

    # 2. RGB Channel Correlations
    r, g, b = img_rgb[:,:,2].ravel(), img_rgb[:,:,1].ravel(), img_rgb[:,:,0].ravel()
    
    # Use np.corrcoef and handle potential std dev of zero
    corr_rg = np.corrcoef(r, g)[0, 1]
    corr_rb = np.corrcoef(r, b)[0, 1]
    corr_gb = np.corrcoef(g, b)[0, 1]
    
    correlation_features = np.array([corr_rg, corr_rb, corr_gb])
    # Replace NaN with 0 (can happen if a channel is constant)
    correlation_features = np.nan_to_num(correlation_features)

    return np.concatenate([saturation_features, correlation_features])

# ------------------------------------------------------------------------------
# (D) Noise residual 기반 특징
# ------------------------------------------------------------------------------
def get_noise_residual_features(img_rgb, mode='denoise'):
    """
    Calculates features from the noise residual, as per gemini.md (D).
    - Denoise-based residual (mode='denoise', slow)
    - High-pass-based residual (mode='highpass', fast)
    """
    if img_rgb is None:
        return np.zeros(3)

    if mode == 'denoise':
        # Use a faster denoising setting for efficiency
        denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        # Calculate residual (and convert to float for covariance)
        residual = img_rgb.astype(np.float32) - denoised.astype(np.float32)
    
    elif mode == 'highpass':
        # Apply Laplacian filter to each channel
        # ksize=3 is a common choice for this filter
        residual = cv2.Laplacian(img_rgb, cv2.CV_32F, ksize=3)
        
    else:
        raise ValueError(f"Unknown residual mode: {mode}")

    res_r, res_g, res_b = residual[:,:,2].ravel(), residual[:,:,1].ravel(), residual[:,:,0].ravel()
    
    # Covariance matrix
    cov_matrix = np.cov(np.vstack([res_r, res_g, res_b]))
    
    # Extract the upper triangle of the covariance matrix
    cov_rg = cov_matrix[0, 1]
    cov_rb = cov_matrix[0, 2]
    cov_gb = cov_matrix[1, 2]
    
    return np.array([cov_rg, cov_rb, cov_gb])


def extract_all_features(img, feature_size_spec1d=128, color_hist_bins=32, residual_mode='denoise'):
    """
    Top-level function to extract and concatenate all feature sets.
    F(x) = [F_spec1D(x), F_specDistort(x), F_color(x), F_residual(x)]
    """
    # Convert to grayscale for spectrum analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # (A) 1D Power Spectrum
    spec1d_features = get_azimuthal_average(gray_img, feature_size=feature_size_spec1d)
    
    # (B) Spectral Distortion
    spec_dist_features = get_spectral_distortion_features(spec1d_features)
    
    # (C) Color Cues
    color_features = get_color_cues_features(img, hist_bins=color_hist_bins)
    
    # (D) Noise Residuals
    residual_features = get_noise_residual_features(img, mode=residual_mode)
    
    # Concatenate all features into a single vector
    final_feature_vector = np.concatenate([
        spec1d_features,
        spec_dist_features,
        color_features,
        residual_features
    ])
    
    return final_feature_vector

def get_feature_names(feature_size_spec1d=128, color_hist_bins=32):
    """
    Returns a list of column names for the features.
    This helps keep train.py and predict.py in sync.
    """
    # (A)
    spec1d_names = [f'spec_bin_{i}' for i in range(feature_size_spec1d)]
    
    # (B)
    spec_dist_names = ['spec_ratio_high_low', 'spec_ratio_mid_low', 'spec_ratio_high_mid']
    
    # (C)
    color_hist_names = [f'sat_hist_{i}' for i in range(color_hist_bins)]
    color_stat_names = ['sat_mean', 'sat_std']
    color_corr_names = ['corr_rg', 'corr_rb', 'corr_gb']
    color_names = color_hist_names + color_stat_names + color_corr_names
    
    # (D)
    residual_names = ['res_cov_rg', 'res_cov_rb', 'res_cov_gb']
    
    # Concatenate all feature names in the same order as extract_all_features
    final_feature_names = spec1d_names + spec_dist_names + color_names + residual_names
    
    return final_feature_names
