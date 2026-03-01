import cv2
import numpy as np


def apply_gaussian_smoothing(image, kernel_size=5):
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return smoothed


def apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Compute the sharpened result
    # Use float to prevent overflow/underflow
    image_float = image.astype(np.float64)
    blurred_float = blurred.astype(np.float64)

    # Detail mask = original - blurred
    detail = image_float - blurred_float

    # Apply threshold — only sharpen where the difference is significant
    if threshold > 0:
        mask = np.abs(detail) >= threshold
        detail = detail * mask

    # Sharpened = original + amount * detail
    sharpened = image_float + amount * detail

    # Clip to valid range and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

def enhance_document(image, smooth_kernel=5, sharp_kernel=5, sigma=1.0, amount=1.5):

    print("\n[STEP 2] Spatial Domain Enhancement")

    # Step 2a: Gaussian Smoothing
    print(f"  Applying Gaussian smoothing (kernel={smooth_kernel})...")
    smoothed = apply_gaussian_smoothing(image, kernel_size=smooth_kernel)

    # Step 2b: Unsharp Masking
    print(f"  Applying unsharp masking (kernel={sharp_kernel}, sigma={sigma}, amount={amount})...")
    enhanced = apply_unsharp_mask(smoothed, kernel_size=sharp_kernel, sigma=sigma, amount=amount)

    print("  Enhancement complete.")
    return enhanced, smoothed