import numpy as np


# --- Mean Squared Error (MSE) Calculation ---
# This function measures the average of the squares of the errors—that is, the
# average squared difference between the estimated values (processed pixels)
# and the actual value (reference pixels). A value closer to 0 indicates higher
# structural similarity between the two images.
def compute_mse(image_a, image_b):
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Images must have the same shape. Got {image_a.shape} and {image_b.shape}."
        )

    # Use float64 to prevent data overflow during the squaring operation,
    # ensuring precision for high-intensity pixel differences.
    err = np.sum((image_a.astype("float64") - image_b.astype("float64")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # If the image is multi-channel (e.g., RGB), normalize the error by dividing
    # it by the number of channels to maintain a consistent scale.
    if len(image_a.shape) == 3:
        err /= float(image_a.shape[2])

    return err


# --- Peak Signal-to-Noise Ratio (PSNR) Calculation ---
# PSNR provides a measurement of the reconstruction quality of an image.
# It is expressed in decibels (dB) using a logarithmic scale. A higher PSNR
# value typically indicates that the processed image is more faithful to
# the original "ground truth" reference.
def compute_psnr(image_a, image_b, max_pixel=255.0):
    mse = compute_mse(image_a, image_b)

    # If the error is 0, the images are identical; mathematically,
    # this results in an infinite signal-to-noise ratio.
    if mse == 0:
        return float("inf")

    # Formula: 10 * log10(Max_Pixel^2 / MSE)
    # This ratio compares the maximum possible power of a signal to the
    # power of corrupting noise that affects the fidelity of its representation.
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return psnr


# --- Comparative Analysis and Reporting ---
# This function compares the results of Sauvola, Niblack, and Adaptive Thresholding
# against a reference image. It computes quality metrics, generates a formatted
# comparison table, and determines which method preserved the most signal (highest PSNR).
def evaluate_methods(reference, sauvola_result, adaptive_result, niblack_result=None):
    import cv2

    print("\n[STEP 4] Quantitative Evaluation")

    # Convert the reference image (ground truth) to grayscale to ensure
    # compatibility with the thresholded binary results.
    if len(reference.shape) == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Calculate metrics for the Sauvola method
    mse_sauvola = compute_mse(reference, sauvola_result)
    psnr_sauvola = compute_psnr(reference, sauvola_result)

    # Calculate metrics for the Niblack method
    mse_niblack = compute_mse(reference, niblack_result)
    psnr_niblack = compute_psnr(reference, niblack_result)

    # Calculate metrics for the Adaptive Gaussian method
    mse_adaptive = compute_mse(reference, adaptive_result)
    psnr_adaptive = compute_psnr(reference, adaptive_result)

    # Printing the Evaluation Results Table
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║       Thresholding Methods — Evaluation Results      ║")
    print("  ╠══════════════════════╦════════════════╦══════════════╣")
    print("  ║       Method         ║   MSE          ║  PSNR (dB)   ║")
    print("  ╠══════════════════════╬════════════════╬══════════════╣")
    print(f"  ║ Sauvola (Local)      ║  {mse_sauvola:>12.2f}  ║  {psnr_sauvola:>10.2f}  ║")
    print(f"  ║ Niblack (Local)      ║  {mse_niblack:>12.2f}  ║  {psnr_niblack:>10.2f}  ║")
    print(f"  ║ Adaptive (Gaussian)  ║  {mse_adaptive:>12.2f}  ║  {psnr_adaptive:>10.2f}  ║")
    print("  ╚══════════════════════╩════════════════╩══════════════╝")

    # Determine the winner based on the highest PSNR score
    psnr_scores = {
        "Sauvola Thresholding": psnr_sauvola,
        "Niblack Thresholding": psnr_niblack,
        "Adaptive Thresholding": psnr_adaptive,
    }
    winner = max(psnr_scores, key=psnr_scores.get)

    # Check for ties in the scoring
    max_psnr = psnr_scores[winner]
    tied = [name for name, score in psnr_scores.items() if score == max_psnr]
    if len(tied) > 1:
        winner = "Tie"

    print("\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  Best Method         ║{winner:<31s}║")
    print("  ╚══════════════════════╩═══════════════════════════════╝")

    results = {
        "sauvola": {"mse": mse_sauvola, "psnr": psnr_sauvola},
        "niblack": {"mse": mse_niblack, "psnr": psnr_niblack},
        "adaptive": {"mse": mse_adaptive, "psnr": psnr_adaptive},
        "winner": winner,
    }

    return results
