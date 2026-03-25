import cv2
import numpy as np

# --- Sauvola Local Thresholding Algorithm ---
# This function calculates a unique threshold value for every individual pixel
# based on its local neighborhood. It is particularly effective at cleaning up
# documents with uneven lighting, shadows, or background noise (stains).
# Formula: T = mean * (1 + k * (std / R - 1))
def apply_sauvola_threshold(image, window_size=25, k=0.2, R=128):
    # Convert the image to grayscale if it is currently in color (BGR)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure the window size is an odd number to maintain a centered pixel
    if window_size % 2 == 0:
        window_size += 1

    gray_float = gray.astype(np.float64)

    # Step 1: Compute the local mean using a box filter (moving average)
    local_mean = cv2.blur(gray_float, (window_size, window_size))

    # Step 2: Compute the local standard deviation (measure of contrast/noise)
    # Using the formula: std = sqrt(E[X^2] - (E[X])^2)
    local_mean_sq = cv2.blur(gray_float ** 2, (window_size, window_size))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    # Step 3: Generate the Sauvola threshold map
    # k: Sensitivity factor (controls the threshold level), R: Dynamic range of std
    threshold_map = local_mean * (1.0 + k * (local_std / R - 1.0))

    # Step 4: Binarize the image based on the calculated threshold map
    # Pixels equal to or greater than the threshold become white (255)
    binary = np.zeros_like(gray)
    binary[gray_float >= threshold_map] = 255

    params = {"window_size": window_size, "k": k, "R": R}
    return binary, params

# --- Niblack Local Thresholding Algorithm ---
# This is a classic local thresholding method that serves as the basis for Sauvola.
# It uses the local mean and standard deviation to separate text from background.
# Formula: T = mean + k * std
def apply_niblack_threshold(image, window_size=25, k=-0.2):
    # Convert to grayscale if the input is a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure window_size is an odd integer
    if window_size % 2 == 0:
        window_size += 1

    gray_float = gray.astype(np.float64)

    # Compute the local mean across the specified window
    local_mean = cv2.blur(gray_float, (window_size, window_size))

    # Compute the local standard deviation to gauge pixel variance
    local_mean_sq = cv2.blur(gray_float ** 2, (window_size, window_size))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    # Niblack threshold calculation: T = local_mean + k * local_std
    # Usually, a negative 'k' value is used to extract dark text from light backgrounds.
    threshold_map = local_mean + k * local_std

    # Apply binarization
    binary = np.zeros_like(gray)
    binary[gray_float >= threshold_map] = 255

    params = {"window_size": window_size, "k": k}
    return binary, params

# --- Standard Adaptive Thresholding (OpenCV) ---
# This utilizes the built-in OpenCV function to binarize the image.
# It uses a Gaussian-weighted sum of the neighborhood, providing a
# computationally fast and generally effective result for standard documents.
def apply_adaptive_threshold(image, block_size=11, C=2):
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian-based adaptive thresholding
    # This adjusts the threshold dynamically over small regions of the image.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )

    return binary


# --- Thresholding Methods Comparative Analysis ---
# This function runs Sauvola, Niblack, and Adaptive Thresholding simultaneously.
# This allows for a side-by-side comparison to determine which method
# produces the most legible text and cleanest background for a specific document.
def compare_thresholds(image, window_size=25, k=0.2, R=128, block_size=11, C=2, niblack_k=-0.2):
    print("\n[STEP 3] Image Segmentation (Thresholding Comparison)")

    # Execute Sauvola Method
    print(f"  Applying Sauvola thresholding (window={window_size}, k={k}, R={R})...")
    sauvola_result, sauvola_params = apply_sauvola_threshold(
        image, window_size=window_size, k=k, R=R
    )

    # Execute Niblack's Method
    print(f"  Applying Niblack thresholding (window={window_size}, k={niblack_k})...")
    niblack_result, niblack_params = apply_niblack_threshold(
        image, window_size=window_size, k=niblack_k
    )

    # Execute OpenCV's Adaptive Method
    print(f"  Applying adaptive thresholding (block_size={block_size}, C={C})...")
    adaptive_result = apply_adaptive_threshold(image, block_size=block_size, C=C)

    print("  Thresholding comparison complete.")

    # Return results and used parameters in a structured dictionary
    return {
        "sauvola": sauvola_result,
        "sauvola_params": sauvola_params,
        "niblack": niblack_result,
        "niblack_params": niblack_params,
        "adaptive": adaptive_result,
        "block_size": block_size,
        "C": C,
    }