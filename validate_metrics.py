"""
Validation Script - PSNR & SSIM Testing on Paired Images
Use this OFFLINE to validate model quality on test dataset with reference images
"""
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys
sys.path.append('.')
from metrics import ImageMetrics

# Configuration
ENHANCER_PATH = ""
TEST_DIR_RAW = ""  # Degraded underwater images
TEST_DIR_CLEAN = ""  # Clean reference images (if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_enhancer():
    """Load the enhancement model"""
    from model_architecture import UNetGenerator
    
    enhancer = UNetGenerator().to(DEVICE)
    enhancer.load_state_dict(torch.load(ENHANCER_PATH, map_location=DEVICE))
    enhancer.eval()
    return enhancer

def validate_on_paired_images():
    """
    Validate PSNR & SSIM on paired test images
    Requires paired images: testA (degraded) and testB (clean reference)
    """
    print("=" * 70)
    print("VALIDATION: PSNR & SSIM Assessment")
    print("=" * 70)
    
    # Check if testB exists
    test_clean_path = Path(TEST_DIR_CLEAN)
    if not test_clean_path.exists():
        print(f"\n⚠️  WARNING: Reference images not found at {TEST_DIR_CLEAN}")
        print("   PSNR and SSIM require clean reference images for comparison.")
        print("   For live demo, use UIQM (reference-free) instead.")
        return
    
    # Load model
    print(f"\nLoading enhancer model on {DEVICE}...")
    enhancer = load_enhancer()
    metrics = ImageMetrics()
    
    # Get test images
    raw_images = sorted(Path(TEST_DIR_RAW).glob("*.jpg"))[:10]  # Test on 10 images
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    results = []
    print(f"\nProcessing {len(raw_images)} test images...\n")
    
    for i, raw_path in enumerate(raw_images, 1):
        # Try to find matching clean image
        clean_path = test_clean_path / raw_path.name
        if not clean_path.exists():
            print(f"  Skipping {raw_path.name} - no reference found")
            continue
          # Load images
        raw_img = cv2.imread(str(raw_path))
        clean_ref = cv2.imread(str(clean_path))
        
        # Enhance with model
        img_pil = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))  # type: ignore[arg-type]
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)  # type: ignore[attr-defined]
        
        with torch.no_grad():
            enhanced_tensor = enhancer(img_tensor)
        
        enhanced_img = enhanced_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        enhanced_img = (enhanced_img + 1) / 2.0
        enhanced_img = np.clip(enhanced_img, 0, 1)
        enhanced_uint8 = (enhanced_img * 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
        
        # Resize to match
        if enhanced_bgr.shape != clean_ref.shape:  # type: ignore[union-attr]
            clean_ref = cv2.resize(clean_ref, (enhanced_bgr.shape[1], enhanced_bgr.shape[0]))  # type: ignore[arg-type]
        
        # Calculate metrics
        psnr_val = metrics.calculate_psnr(enhanced_bgr, clean_ref)
        ssim_val = metrics.calculate_ssim(enhanced_bgr, clean_ref)
        uiqm_raw = metrics.calculate_uiqm(raw_img)
        uiqm_enhanced = metrics.calculate_uiqm(enhanced_bgr)
        
        results.append({
            'file': raw_path.name,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'uiqm_raw': uiqm_raw,
            'uiqm_enhanced': uiqm_enhanced
        })
        
        print(f"  [{i:2d}] {raw_path.name:30s} | PSNR: {psnr_val:6.2f} dB | SSIM: {ssim_val:.4f} | UIQM: {uiqm_raw:.2f}→{uiqm_enhanced:.2f}")
    
    if not results:
        print("\n❌ No paired images found for validation!")
        return
    
    # Calculate averages
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_uiqm_raw = np.mean([r['uiqm_raw'] for r in results])
    avg_uiqm_enhanced = np.mean([r['uiqm_enhanced'] for r in results])
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Average PSNR:          {avg_psnr:.2f} dB")
    print(f"  Average SSIM:          {avg_ssim:.4f}")
    print(f"  Average UIQM (Raw):    {avg_uiqm_raw:.2f}")
    print(f"  Average UIQM (Enhanced): {avg_uiqm_enhanced:.2f}")
    print(f"  UIQM Improvement:      {((avg_uiqm_enhanced - avg_uiqm_raw) / avg_uiqm_raw * 100):.1f}%")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("  PSNR Ranges:")
    print("    • >30 dB   : Excellent quality")
    print("    • 20-30 dB : Good quality")
    print("    • <20 dB   : Poor quality")
    print()
    print("  SSIM Ranges:")
    print("    • >0.9     : Excellent similarity")
    print("    • 0.7-0.9  : Good similarity")
    print("    • <0.7     : Poor similarity")
    print()
    print("  UIQM:")
    print("    • Enhanced should be HIGHER than raw")
    print("    • Typical range: 0.5-5.0")
    print("=" * 70)

if __name__ == "__main__":
    validate_on_paired_images()
