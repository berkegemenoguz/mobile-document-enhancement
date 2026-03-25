import cv2
import numpy as np


# --- Consistently Ordering Corner Points ---
# This function takes 4 random corner points and sorts them based on their (x, y) coordinates.
# It ensures a fixed order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
# This consistency is mandatory for calculating the correct perspective transformation matrix.
def order_points(pts):
    # Create a placeholder matrix for the 4 ordered points
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    # The top-left point will have the smallest sum (x+y),
    # while the bottom-right point will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # The top-right point will have the smallest difference (y-x),
    # while the bottom-left point will have the largest difference.
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]  # Top-right
    rect[3] = pts[np.argmax(d)]  # Bottom-left

    return rect


# --- Detecting the Document within the Image ---
# This function uses grayscale conversion, Gaussian blurring, and Canny edge detection
# to identify the boundaries of the paper. It then captures the largest contour
# that resembles a 4-cornered polygon (the document).
def detect_document(image):
    # Pre-processing: Convert to grayscale and blur to reduce high-frequency noise
    # that could lead to false edge detection.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using the Canny algorithm
    edged = cv2.Canny(blurred, 50, 200)

    # Apply dilation to close small gaps in the detected edges,
    # making the contour discovery more robust.
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    # Find all contours (line segments/areas) in the processed image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order and keep the top 5 largest candidates
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    document_contour = None

    for contour in contours:
        # Approximate the contour to simplify it into a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the simplified polygon has exactly 4 corners, we assume we found the document
        if len(approx) == 4:
            document_contour = approx
            break

    if document_contour is not None:
        return document_contour.reshape(4, 2)

    return None


# --- Perspective Transformation and Bird's-Eye View ---
# Using the 4 detected corners, this function "warps" the image to correct
# the perspective, making the document appear as a flat rectangle viewed
# directly from above.
def four_point_warp(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the width of the new image by finding the maximum distance
    # between bottom-right/bottom-left and top-right/top-left x-coordinates.
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Calculate the height of the new image by finding the maximum distance
    # between top-right/bottom-right and top-left/bottom-left y-coordinates.
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Define the destination points for the "unfolded" view
    # (the corners of the new, perfectly rectangular image).
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it to the source image
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


# --- Main Perspective Correction Process ---
# This coordinates the document detection and the warping process.
# It serves as the entry point for Step 1 of the document processing pipeline.
def correct_perspective(image):
    print("\n[STEP 1] Perspective Correction")
    print("  Detecting document boundaries...")

    # Locate the four corners of the document
    corners = detect_document(image)

    if corners is None:
        print("  WARNING: No document contour detected. Using original image.")
        print("  TIP: Ensure the document has clear edges against the background.")
        return image.copy(), False

    print(f"  Document corners found: {corners.tolist()}")

    # Rectify the image based on the detected corners
    warped = four_point_warp(image, corners)
    print(f"  Warped image size: {warped.shape[1]}x{warped.shape[0]}")

    return warped, True