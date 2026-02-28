# Automated Document Perspective Correction and Adaptive Text Enhancement

This project addresses the common issues of perspective distortion and poor legibility in document photos captured via mobile devices. The primary goal is to transform skewed, low-contrast images into scan-quality digital documents.

## 📌 Problem Definition
Mobile document captures often suffer from:
* **Perspective Distortion:** Skewed angles making the document appear non-rectangular.
* **Poor Legibility:** Low contrast and uneven lighting (shadows, glares).
* **Noise:** Grainy artifacts from mobile sensors.

---

## 🛠 Planned Methods (Course Components)

### 1. Geometric Transformations
I will implement **4-point perspective warping** to rectify the document's orientation. This process involves detecting the document corners and mapping them to a standard rectangular coordinate system.

### 2. Spatial Domain Processing
* **Noise Reduction:** Applying **Gaussian Smoothing** for initial noise reduction.
* **Text Enhancement:** Following smoothing with **Unsharp Masking** to enhance text edges and improve readability.

### 3. Image Segmentation
**Adaptive Thresholding** techniques will be used to effectively separate text from backgrounds under various lighting conditions, overcoming the limitations of static thresholding.

### 4. Implementation Stack
The project is developed using the following tools:
* **Language:** Python
* **Libraries:** OpenCV, NumPy, Matplotlib

---

## 📊 Evaluation & Comparison
To evaluate the effectiveness of the proposed system, a comparative analysis is performed:

* **Technique Comparison:** Global Thresholding (Otsu) versus **Adaptive Thresholding** to evaluate text clarity.
* **Quantitative Metrics:** * **MSE** (Mean Squared Error)
    * **PSNR** (Peak Signal-to-Noise Ratio)

The PSNR is calculated as follows:
$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$

---

## 📂 Dataset
I plan to use a **self-collected dataset** of document photos taken under different angles, various distances, and diverse lighting environments to ensure the robustness of the algorithm.

---

## 🚀 How to Run (Placeholder)
*(This section can be updated as the project develops)*
1. Clone the repository.
2. Install dependencies: `pip install opencv-python numpy matplotlib`
3. Run the main script: `python main.py`
