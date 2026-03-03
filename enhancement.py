import cv2
import numpy as np


# --- Gaussian Yumuşatma (Gürültü Azaltma) Uygulama ---
# Görüntüdeki yüksek frekanslı gürültüleri (noktalanmaları) temizlemek için
# Gaussian Blur filtresini kullanır. Bu, keskinleştirme öncesi zemini hazırlar.
def apply_gaussian_smoothing(image, kernel_size=5):
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return smoothed

# --- Keskinleştirme (Unsharp Masking) Algoritması ---
# Görüntüyü daha net hale getirmek için "Unsharp Mask" tekniğini uygular.
# Mantık: Orijinal görüntüden bulanık versiyonu çıkarılır (detay maskesi),
# bu maske güçlendirilerek tekrar orijinalin üzerine eklenir.
def apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    # 1. Adım: Görüntünün bulanık (düşük frekanslı) bir versiyonunu oluştur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # 2. Adım: Hassas hesaplama için veri tipini float'a çevir (taşmaları önlemek için)
    image_float = image.astype(np.float64)
    blurred_float = blurred.astype(np.float64)

    # 3. Adım: Detay maskesini hesapla (Detay = Orijinal - Bulanık)
    detail = image_float - blurred_float

    # 4. Adım: Eşikleme uygula (Sadece belirli bir seviyenin üzerindeki farkları keskinleştir)
    if threshold > 0:
        mask = np.abs(detail) >= threshold
        detail = detail * mask

    # 5. Adım: Keskinleştirilmiş görüntüyü oluştur (Orijinal + Miktar * Detay)
    sharpened = image_float + amount * detail

    # 6. Adım: Değerleri 0-255 aralığına kısıtla ve tekrar standart görüntü formatına (uint8) dönüştür
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

# --- Ana Döküman İyileştirme İş Akışı ---
# Önce gürültüyü azaltan (smoothing), ardından metinleri belirginleştiren (sharpening)
# iki aşamalı bir iyileştirme pipeline'ı yürütür.
def enhance_document(image, smooth_kernel=5, sharp_kernel=5, sigma=1.0, amount=1.5):

    print("\n[STEP 2] Spatial Domain Enhancement")

    # Adım 2a: Yumuşatma aşaması
    print(f"  Applying Gaussian smoothing (kernel={smooth_kernel})...")
    smoothed = apply_gaussian_smoothing(image, kernel_size=smooth_kernel)

    # Adım 2b: Keskinleştirme aşaması
    print(f"  Applying unsharp masking (kernel={sharp_kernel}, sigma={sigma}, amount={amount})...")
    enhanced = apply_unsharp_mask(smoothed, kernel_size=sharp_kernel, sigma=sigma, amount=amount)

    print("  Enhancement complete.")
    return enhanced, smoothed