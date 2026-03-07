import argparse
import os
import sys
import cv2

# --- Özel Modüllerin İçe Aktarılması ---
# Önceki adımlarda yazdığımız yardımcı fonksiyonları ve
# işleme aşamalarını projeye dahil ediyoruz.
from utils import load_image, save_image, display_images, display_comparison_grid
from perspective_correction import correct_perspective
from enhancement import enhance_document
from thresholding import compare_thresholds
from evaluation import evaluate_methods

# --- Çıktı Klasörü Oluşturma ---
# İşlenen resimlerin kaydedileceği dizini kontrol eder,
# eğer yoksa hata almamak için yeni bir klasör oluşturur.
def create_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Output directory: {output_dir}")

# --- Ana İşleme Hattı (Pipeline) ---
# Döküman iyileştirme sürecinin tüm adımlarını (yükleme, düzeltme,
# netleştirme, eşikleme ve değerlendirme) koordine eden ana fonksiyondur.
def run_pipeline(image_path, output_dir="output", show_display=True):
    print("=" * 60)
    print("  DOCUMENT ENHANCEMENT PIPELINE")
    print("=" * 60)

    # --- Resim Yükleme ---
    print(f"\n  Loading image: {image_path}")
    original = load_image(image_path)
    print(f"  Image size: {original.shape[1]}x{original.shape[0]}")

    create_output_dir(output_dir)

    # Orijinal hali yedekle
    save_image(os.path.join(output_dir, "01_original.jpg"), original)

    # --- 1. AŞAMA: Perspektif Düzeltme ---
    # Eğri duran kağıdı tespit eder ve düz bir dikdörtgen haline getirir.
    corrected, perspective_ok = correct_perspective(original)
    save_image(os.path.join(output_dir, "02_perspective_corrected.jpg"), corrected)

    # --- 2. AŞAMA: Uzamsal İyileştirme ---
    # Gürültüyü azaltır (Smooth) ve kenarları belirginleştirir (Sharpen).
    enhanced, smoothed = enhance_document(corrected)
    save_image(os.path.join(output_dir, "03_smoothed.jpg"), smoothed)
    save_image(os.path.join(output_dir, "04_enhanced.jpg"), enhanced)

    # --- 3. AŞAMA: Eşikleme Karşılaştırması ---
    # Görüntüyü metin ve arka plan olarak ayırmak için iki farklı metodu dener.
    thresh_results = compare_thresholds(enhanced)

    sauvola_binary = thresh_results["sauvola"]
    adaptive_binary = thresh_results["adaptive"]

    save_image(os.path.join(output_dir, "05_sauvola_threshold.jpg"), sauvola_binary)
    save_image(os.path.join(output_dir, "06_adaptive_threshold.jpg"), adaptive_binary)

    # --- 4. AŞAMA: Sayısal Değerlendirme ---
    # Hangi yöntemin daha başarılı olduğunu MSE ve PSNR değerleriyle ölçer.
    eval_results = evaluate_methods(enhanced, sauvola_binary, adaptive_binary)

    # --- GÖRSELLEŞTİRME ---
    # Tüm işlem basamaklarını ve karşılaştırmaları ekranda gösterir.
    if show_display:
        print("\n[DISPLAY] Showing visual comparison...")

        # Tüm aşamaları içeren ızgara görünümü
        stages = {
            "1. Original": original,
            "2. Perspective Corrected": corrected,
            "3. Gaussian Smoothed": smoothed,
            "4. Unsharp Enhanced": enhanced,
            "5. Sauvola Threshold": sauvola_binary,
            "6. Adaptive Threshold": adaptive_binary,
        }

        display_comparison_grid(
            stages,
            figsize=(18, 12),
            save_path=os.path.join(output_dir, "07_comparison_grid.jpg"),
        )

        # Eşikleme yöntemlerini yan yana gösteren karşılaştırma
        display_images(
            [sauvola_binary, adaptive_binary],
            ["Sauvola Thresholding", "Adaptive Thresholding"],
            figsize=(14, 7),
            save_path=os.path.join(output_dir, "08_threshold_comparison.jpg"),
        )

    # --- ÖZET RAPOR ---
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Input:  {image_path}")
    print(f"  Output: {output_dir}/")
    print(f"  Perspective correction: {'Applied' if perspective_ok else 'Skipped (no contour found)'}")
    print(f"  Winner: {eval_results['winner']}")
    print("=" * 60)

    return eval_results

# --- Komut Satırı Argümanlarını Ayıklama ---
# Kullanıcının terminalden dosya yolu, çıktı klasörü gibi
# parametreleri kolayca girmesini sağlar.
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Document Perspective Correction and Adaptive Text Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input document image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output images (default: output/)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable Matplotlib display (useful for headless environments)",
    )

    return parser.parse_args()

# --- Programın Giriş Noktası ---
# Script doğrudan çalıştırıldığında argümanları okur ve süreci başlatır.
def main():
    args = parse_arguments()
    run_pipeline(args.image, output_dir=args.output_dir, show_display=not args.no_display)

if __name__ == "__main__":
    main()
