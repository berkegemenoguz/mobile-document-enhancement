import numpy as np

# --- Ortalama Kare Hata (Mean Squared Error - MSE) Hesaplama ---
# İki görüntü arasındaki piksel farklarının karelerinin ortalamasını alır.
# Değer 0'a ne kadar yakınsa, görüntüler birbirine o kadar benziyor demektir.
def compute_mse(image_a, image_b):
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Images must have the same shape. Got {image_a.shape} and {image_b.shape}."
        )

    # Kare alma işlemi sırasında veri taşmasını önlemek için float64 kullanılır
    err = np.sum((image_a.astype("float64") - image_b.astype("float64")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # Eğer görüntü renkliyse (RGB), hata değerini kanal sayısına bölerek normalize et
    if len(image_a.shape) == 3:
        err /= float(image_a.shape[2])

    return err


# --- Tepe Sinyal-Gürültü Oranı (Peak Signal-to-Noise Ratio - PSNR) Hesaplama ---
# Görüntü kalitesini logaritmik bir ölçekte (decibel - dB) ölçer.
# PSNR değeri ne kadar yüksekse, işlenmiş görüntü orijinaline o kadar sadıktır.
def compute_psnr(image_a, image_b, max_pixel=255.0):
    mse = compute_mse(image_a, image_b)

    # Eğer hata 0 ise görüntüler birebir aynıdır
    if mse == 0:
        return float("inf")

    # Formül: 10 * log10(Max_Pixel^2 / MSE)
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return psnr


# --- Yöntemlerin Karşılaştırmalı Analizi ve Raporlanması ---
# Referans alınan görüntüye göre Sauvola ve Adaptive Threshold sonuçlarını kıyaslar,
# verileri şık bir tablo şeklinde ekrana yazdırır ve kazananı belirler.
def evaluate_methods(reference, sauvola_result, adaptive_result, niblack_result=None):
    import cv2

    print("\n[STEP 4] Quantitative Evaluation")

    # Referans görüntüyü (temiz hali) karşılaştırma için gri tona çevir
    if len(reference.shape) == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Sauvola yöntemi için metrikleri hesapla
    mse_sauvola = compute_mse(reference, sauvola_result)
    psnr_sauvola = compute_psnr(reference, sauvola_result)

    # Niblack yöntemi için metrik hesapla
    mse_niblack = compute_mse(reference, niblack_result)
    psnr_niblack = compute_psnr(reference, niblack_result)

    # Adaptive yöntemi için metrikleri hesapla
    mse_adaptive = compute_mse(reference, adaptive_result)
    psnr_adaptive = compute_psnr(reference, adaptive_result)

    # Sonuç Tablosu Yazdırılıyor
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║       Thresholding Methods — Evaluation Results      ║")
    print("  ╠══════════════════════╦════════════════╦══════════════╣")
    print("  ║       Method         ║   MSE          ║  PSNR (dB)   ║")
    print("  ╠══════════════════════╬════════════════╬══════════════╣")
    print(f"  ║ Sauvola (Local)      ║  {mse_sauvola:>12.2f}  ║  {psnr_sauvola:>10.2f}  ║")
    print(f"  ║ Niblack (Local)      ║  {mse_niblack:>12.2f}  ║  {psnr_niblack:>10.2f}  ║")
    print(f"  ║ Adaptive (Gaussian)  ║  {mse_adaptive:>12.2f}  ║  {psnr_adaptive:>10.2f}  ║")
    print("  ╚══════════════════════╩════════════════╩══════════════╝")

    # Determine winner (highest PSNR)
    psnr_scores = {
        "Sauvola Thresholding": psnr_sauvola,
        "Niblack Thresholding": psnr_niblack,
        "Adaptive Thresholding": psnr_adaptive,
    }
    winner = max(psnr_scores, key=psnr_scores.get)

    # Check for ties
    max_psnr = psnr_scores[winner]
    tied = [name for name, score in psnr_scores.items() if score == max_psnr]
    if len(tied) > 1:
        winner = "Tie"

    print(f"  ║  Best Method         ║{winner:<31s}  ║")
    print("  ╚══════════════════════╩═══════════════════════════════╝")

    results = {
        "sauvola": {"mse": mse_sauvola, "psnr": psnr_sauvola},
        "niblack": {"mse": mse_niblack, "psnr": psnr_niblack},
        "adaptive": {"mse": mse_adaptive, "psnr": psnr_adaptive},
        "winner": winner,
    }

    return results
