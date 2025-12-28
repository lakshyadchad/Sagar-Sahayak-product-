import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import math

class ImageMetrics:
    """Wrapper class for image quality metrics"""
    
    @staticmethod
    def calculate_psnr(enhanced, reference):
        """Peak Signal-to-Noise Ratio"""
        return calculate_psnr(enhanced, reference)
    
    @staticmethod
    def calculate_ssim(enhanced, reference):
        """Structural Similarity Index"""
        return calculate_ssim(enhanced, reference)
    
    @staticmethod
    def calculate_uiqm(img):
        """Underwater Image Quality Measure"""
        return calculate_uiqm(img)

def calculate_psnr(enhanced, reference):
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. Typical good values: 20-30 dB.
    """
    # Ensure shapes match
    if enhanced.shape != reference.shape:
        reference = cv2.resize(reference, (enhanced.shape[1], enhanced.shape[0]))
    
    score = psnr(reference, enhanced, data_range=255)
    return score

def calculate_ssim(enhanced, reference):
    """
    Structural Similarity Index.
    1.0 is perfect match. 0.0 is no match.
    """
    if enhanced.shape != reference.shape:
        reference = cv2.resize(reference, (enhanced.shape[1], enhanced.shape[0]))
        
    # Convert to Gray for SSIM structure check
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # ssim returns (score, full_image) when full=True
    result = ssim(gray_ref, gray_enh, full=True)
    score = result[0]  # Extract just the score
    return score

def calculate_uiqm(img):
    """
    Reference-Free Underwater Image Quality Measure (Simplified).
    Combines Colorfulness (UICM), Sharpness (UISM), and Contrast (UIConM).
    Formula: UIQM = c1*UICM + c2*UISM + c3*UIConM
    """
    # Weights from the original paper
    c1, c2, c3 = 0.0282, 0.2953, 3.5753

    # 1. UICM (Colorfulness) - simplified as standard deviation of Lab channels
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    uicm = np.sqrt((np.var(a) + np.var(b)))

    # 2. UISM (Sharpness) - Edge strength using Sobel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    uism = np.mean(np.sqrt(sobelx**2 + sobely**2))

    # 3. UIConM (Contrast) - LogAMEE (Logarithmic Michelson Contrast)
    # Simplified here as standard deviation of luminance for speed
    uiconm = np.std(gray)

    # Final Composite Score
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    
    # Normalize roughly to 0-5 scale for dashboard readability
    return uiqm / 100.0