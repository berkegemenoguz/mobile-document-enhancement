# 📄 Mobile Document Enhancement

**Automated Document Perspective Correction and Adaptive Text Enhancement**

Transform skewed, low-contrast mobile document photos into clean, scan-quality digital images — all from the command line.

---

## ✨ What It Does

Mobile cameras capture documents at awkward angles, under uneven lighting, and with sensor noise. This tool runs a **4-stage image processing pipeline** that automatically:

1. **Detects & corrects perspective** — finds the document boundary and warps it into a flat, rectangular view.
2. **Reduces noise & sharpens text** — applies Gaussian smoothing followed by unsharp masking for crisp, readable text.
3. **Binarizes with adaptive thresholding** — compares **Sauvola** and **Adaptive Gaussian** thresholding to separate text from the background, even under uneven illumination.
4. **Evaluates quality quantitatively** — computes **MSE** and **PSNR** metrics to objectively determine which thresholding method produced the better result.

Every intermediate result is saved, and an optional visual comparison grid is displayed via Matplotlib.

---

## 🏗️ Project Structure

```
mobile-document-enhancement/
├── main.py                   # Pipeline orchestrator & CLI entry point
├── perspective_correction.py # Document detection & 4-point warp
├── enhancement.py            # Gaussian smoothing & unsharp masking
├── thresholding.py           # Sauvola & adaptive Gaussian thresholding
├── evaluation.py             # MSE / PSNR metric computation
├── utils.py                  # Image I/O & Matplotlib display helpers
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/berkegemenoguz/mobile-document-enhancement.git
cd mobile-document-enhancement

# Install dependencies
pip install -r requirements.txt
```

### Quick Run

```bash
python main.py --image path/to/document.jpg
```

---

## 📖 Usage

```
python main.py --image <IMAGE_PATH> [--output-dir <DIR>] [--no-display]
```

| Argument        | Required | Default    | Description                                           |
|-----------------|----------|------------|-------------------------------------------------------|
| `--image`       | ✅       | —          | Path to the input document photo                      |
| `--output-dir`  | ❌       | `output/`  | Directory where all output images are saved           |
| `--no-display`  | ❌       | off        | Disable the Matplotlib pop-up windows (headless mode) |

### Examples

```bash
# Basic usage — process a document and display results
python main.py --image samples/receipt.jpg

# Save outputs to a custom folder, skip visual display
python main.py --image samples/receipt.jpg --output-dir results/ --no-display
```

### Output Files

After a successful run the output directory will contain:

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `01_original.jpg`             | Copy of the input image                          |
| `02_perspective_corrected.jpg`| Image after 4-point perspective warp             |
| `03_smoothed.jpg`             | After Gaussian noise reduction                   |
| `04_enhanced.jpg`             | After unsharp masking (sharpened text)            |
| `05_sauvola_threshold.jpg`    | Binary result — Sauvola thresholding             |
| `06_adaptive_threshold.jpg`   | Binary result — Adaptive Gaussian thresholding   |
| `07_comparison_grid.jpg`      | Side-by-side grid of all stages                  |
| `08_threshold_comparison.jpg` | Direct comparison of the two thresholding methods|

---

## 🔬 Pipeline Details

### Stage 1 — Perspective Correction

- Converts the image to grayscale and applies **Canny edge detection**.
- Finds the largest quadrilateral contour (the document boundary).
- Uses `cv2.getPerspectiveTransform` to warp the document into a top-down rectangular view.
- If no document boundary is detected, the original image is passed through unchanged.

### Stage 2 — Spatial Enhancement

- **Gaussian Smoothing** removes high-frequency noise while preserving document structure.
- **Unsharp Masking** (`sharpened = original + amount × (original − blurred)`) boosts text edges for improved readability.

### Stage 3 — Thresholding Comparison

Two local thresholding techniques are run and compared:

| Method              | How It Works |
|---------------------|--------------|
| **Sauvola**         | `T(x,y) = mean × [1 + k × (std / R − 1)]` — adapts to local mean *and* standard deviation |
| **Adaptive Gaussian** | Weighted Gaussian mean of a local neighborhood minus a constant `C` |

Both handle uneven lighting far better than a single global threshold.

### Stage 4 — Quantitative Evaluation

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | `(1/N) × Σ(Aᵢ − Bᵢ)²` | Lower is better (less distortion) |
| **PSNR** | `10 · log₁₀(MAX² / MSE)` | Higher is better (dB scale) |

The method with the higher PSNR is declared the winner and printed in the terminal summary.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language  | Python 3   |
| Computer Vision | OpenCV (`opencv-python ≥ 4.5`) |
| Numerical Computing | NumPy (`≥ 1.21`) |
| Visualization | Matplotlib (`≥ 3.4`) |

---

## 📝 License

This project is developed for academic purposes.