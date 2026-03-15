import cv2
import numpy as np

# --- Sauvola Yerel Eşikleme Algoritması ---
# Bu fonksiyon, görüntünün her bölgesi için ayrı bir eşik değeri hesaplar.
# Özellikle düzensiz aydınlatma ve kağıt üzerindeki lekeleri (noise) temizlemede çok başarılıdır.
# Formül: T = mean * (1 + k * (std / R - 1))
def apply_sauvola_threshold(image, window_size=25, k=0.2, R=128):
    # Gerekirse görüntüyü gri tonlamaya çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Pencere boyutunun (window_size) tek sayı olduğundan emin ol
    if window_size % 2 == 0:
        window_size += 1

    gray_float = gray.astype(np.float64)

    # 1. Adım: Kutu filtresi (box filter) ile yerel ortalamayı (mean) hesapla
    local_mean = cv2.blur(gray_float, (window_size, window_size))

    # 2. Adım: Yerel standart sapmayı (std) hesapla
    # Formül: std = sqrt(E[X^2] - (E[X])^2)
    local_mean_sq = cv2.blur(gray_float ** 2, (window_size, window_size))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    # 3. Adım: Sauvola eşik haritasını oluştur
    # k: Hassasiyet çarpanı, R: Standart sapmanın dinamik aralığı
    threshold_map = local_mean * (1.0 + k * (local_std / R - 1.0))

    # 4. Adım: Eşik değerine göre pikselleri siyah veya beyaz yap
    binary = np.zeros_like(gray)
    binary[gray_float >= threshold_map] = 255

    params = {"window_size": window_size, "k": k, "R": R}
    return binary, params

def apply_niblack_threshold(image, window_size=25, k=-0.2):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    gray_float = gray.astype(np.float64)

    # Compute local mean using box filter
    local_mean = cv2.blur(gray_float, (window_size, window_size))

    # Compute local standard deviation
    # std = sqrt(E[X^2] - (E[X])^2)
    local_mean_sq = cv2.blur(gray_float ** 2, (window_size, window_size))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    # Niblack threshold: T = mean + k * std
    threshold_map = local_mean + k * local_std

    # Apply thresholding
    binary = np.zeros_like(gray)
    binary[gray_float >= threshold_map] = 255

    params = {"window_size": window_size, "k": k}
    return binary, params

# --- Standart Uyarlanabilir (Adaptive) Eşikleme ---
# OpenCV'nin hazır fonksiyonunu kullanarak görüntüyü ikili (siyah-beyaz) hale getirir.
# Gaussian ağırlıklı ortalama kullanarak hızlı ve etkili bir sonuç sunar.
def apply_adaptive_threshold(image, block_size=11, C=2):
    # Gerekirse görüntüyü gri tonlamaya çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Gaussian ağırlıklı uyarlanabilir eşikleme uygula
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )

    return binary


# --- Eşikleme Yöntemlerini Karşılaştırma ---
# Hem Sauvola hem de Adaptive Threshold yöntemlerini aynı anda çalıştırarak
# hangi metodun döküman üzerinde daha net sonuç verdiğini görmenizi sağlar.
def compare_thresholds(image, window_size=25, k=0.2, R=128, block_size=11, C=2, niblack_k=-0.2):
    print("\n[STEP 3] Image Segmentation (Thresholding Comparison)")

    # Sauvola Metodu Uygulanıyor
    print(f"  Applying Sauvola thresholding (window={window_size}, k={k}, R={R})...")
    sauvola_result, sauvola_params = apply_sauvola_threshold(
        image, window_size=window_size, k=k, R=R
    )

    # Niblack's method
    print(f"  Applying Niblack thresholding (window={window_size}, k={niblack_k})...")
    niblack_result, niblack_params = apply_niblack_threshold(
        image, window_size=window_size, k=niblack_k
    )

    # OpenCV Adaptive Metodu Uygulanıyor
    print(f"  Applying adaptive thresholding (block_size={block_size}, C={C})...")
    adaptive_result = apply_adaptive_threshold(image, block_size=block_size, C=C)

    print("  Thresholding comparison complete.")

    # Sonuçları ve kullanılan parametreleri bir sözlük yapısında döndür
    return {
        "sauvola": sauvola_result,
        "sauvola_params": sauvola_params,
        "niblack": niblack_result,
        "niblack_params": niblack_params,
        "adaptive": adaptive_result,
        "block_size": block_size,
        "C": C,
    }