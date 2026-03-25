import cv2
import numpy as np


# --- Gaussian Smoothing (Noise Reduction) Application ---
# This function applies a Gaussian Blur filter to the image to suppress high-frequency noise
# and minor artifacts. By smoothing out these "salt-and-pepper" or grainy pixels,
# it creates a cleaner foundation for the subsequent sharpening process, preventing
# the enhancement of unwanted noise.
def apply_gaussian_smoothing(image, kernel_size=5):
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return smoothed

# --- Sharpening (Unsharp Masking) Algorithm ---
# This function implements the "Unsharp Mask" technique to enhance edge contrast and clarity.
# The underlying logic involves subtracting a blurred version of the image from the
# original to isolate high-frequency details (the mask). This detail mask is then
# amplified and blended back into the original image to make features sharper.
def apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    # Step 1: Generate a blurred (low-frequency) version of the original image
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Step 2: Convert the image data type to float64 to ensure high precision
    # and prevent pixel value overflow/underflow during arithmetic operations.
    image_float = image.astype(np.float64)
    blurred_float = blurred.astype(np.float64)

    # Step 3: Extract the detail mask by calculating the difference (Detail = Original - Blurred)
    detail = image_float - blurred_float

    # Step 4: Apply a threshold to the detail mask. This ensures that only
    # significant differences are sharpened, helping to avoid amplifying
    # subtle noise in flat areas of the image.
    if threshold > 0:
        mask = np.abs(detail) >= threshold
        detail = detail * mask

    # Step 5: Construct the final sharpened image by adding the weighted
    # detail mask back to the original (Sharpened = Original + Amount * Detail)
    sharpened = image_float + amount * detail

    # Step 6: Clip the resulting values to the valid 0-255 range and convert
    # the data back to the standard 8-bit unsigned integer (uint8) format.
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

# --- Main Document Enhancement Workflow ---
# This function executes a two-stage spatial domain enhancement pipeline.
# It first reduces noise through a smoothing pass and then emphasizes text
# and structural boundaries using unsharp masking to improve document readability.
def enhance_document(image, smooth_kernel=5, sharp_kernel=5, sigma=1.0, amount=1.5):

    print("\n[STEP 2] Spatial Domain Enhancement")

    # Step 2a: Noise reduction phase (Smoothing)
    print(f"  Applying Gaussian smoothing (kernel={smooth_kernel})...")
    smoothed = apply_gaussian_smoothing(image, kernel_size=smooth_kernel)

    # Step 2b: Detail enhancement phase (Sharpening)
    print(f"  Applying unsharp masking (kernel={sharp_kernel}, sigma={sigma}, amount={amount})...")
    enhanced = apply_unsharp_mask(smoothed, kernel_size=sharp_kernel, sigma=sigma, amount=amount)

    print("  Enhancement complete.")
    return enhanced, smoothed