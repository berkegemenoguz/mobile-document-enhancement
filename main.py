import argparse
import os
import sys
import glob
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
   niblack_binary = thresh_results["niblack"]
   adaptive_binary = thresh_results["adaptive"]


   save_image(os.path.join(output_dir, "05_sauvola_threshold.jpg"), sauvola_binary)
   save_image(os.path.join(output_dir, "06_niblack_threshold.jpg"), niblack_binary)
   save_image(os.path.join(output_dir, "07_adaptive_threshold.jpg"), adaptive_binary)

   # --- 4. AŞAMA: Sayısal Değerlendirme ---
   # Hangi yöntemin daha başarılı olduğunu MSE ve PSNR değerleriyle ölçer.
   eval_results = evaluate_methods(enhanced, sauvola_binary, adaptive_binary, niblack_binary)


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
           "6. Niblack Threshold": niblack_binary,
           "7. Adaptive Threshold": adaptive_binary,
       }


       display_comparison_grid(
           stages,
           figsize=(18, 14),
           save_path=os.path.join(output_dir, "09_comparison_grid.jpg"),
       )


       # Eşikleme yöntemlerini yan yana gösteren karşılaştırma
       display_images(
           [sauvola_binary,niblack_binary, adaptive_binary],
           ["Sauvola Thresholding","Niblack Thresholding", "Adaptive Thresholding"],
           figsize=(18, 7),
           save_path=os.path.join(output_dir, "09_threshold_comparison.jpg"),
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


# --- Desteklenen Resim Uzantıları ---
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')


# --- Proje Dizinindeki Resimleri Tarama ---
# Programın bulunduğu klasördeki desteklenen resim dosyalarını bulur
# ve boyut bilgisiyle birlikte döndürür.
def scan_images():
   """Proje dizinindeki desteklenen resim dosyalarını bulur."""
   script_dir = os.path.dirname(os.path.abspath(__file__))
   found = []
   for f in sorted(os.listdir(script_dir)):
       if os.path.isfile(os.path.join(script_dir, f)) and f.lower().endswith(SUPPORTED_EXTENSIONS):
           full_path = os.path.join(script_dir, f)
           size = os.path.getsize(full_path)
           found.append((f, full_path, size))
   return found






# --- İnteraktif Terminal Modu ---
# Kullanıcıdan terminalde dosya seçimi, çıktı klasörü ve
# görselleştirme tercihi alarak pipeline'ı çalıştırır.
def interactive_mode():
   print("=" * 60)
   print("     DOCUMENT ENHANCEMENT")
   print("=" * 60)


   # --- Proje dizinindeki görselleri tara ---
   images = scan_images()


   if images:
       print(f"\n The images in the project directory:")
       print("  " + "─" * 44)
       for i, (name, path, size) in enumerate(images, 1):
           print(f"    [{i}] {name}")
       manual_option = len(images) + 1
       print(f"    [{manual_option}] Enhancement via file path.")
       print("  " + "─" * 44)
   else:
       print("\n  Can not find the image in the project directory.")
       manual_option = 1
       print(f"    [{manual_option}] Enhancement via file path.")
       print("  " + "─" * 44)


   # --- Dosya Seçimi ---
   while True:
       try:
           choice = input(f"\n  Your choice is (1-{manual_option}): ").strip()
           if not choice:
               continue


           choice_num = int(choice)


           if choice_num == manual_option:
               # Doğrudan dosya yolu girme
               while True:
                   file_path = input("\n Enter the file path: ").strip()
                   # ~ karakterini genişlet
                   file_path = os.path.expanduser(file_path)
                   if os.path.isfile(file_path):
                       image_path = file_path
                       break
                   else:
                       print(f"  Can not find the file: {file_path}")
                       print("  Try again.")
               break
           elif 1 <= choice_num <= len(images):
               image_path = images[choice_num - 1][1]
               break
           else:
               print(f"Please choose between 1 and {manual_option}.")
       except ValueError:
           print(f"Invalid option. Please insert a number.")


   # --- Çıktı Klasörü (varsayılan: output/) ---
   output_dir = "output"


   # --- Onay ---
   print("\n  " + "─" * 44)
   print(f"  Choosen file:   {image_path}")
   print(f"  Output file:   {output_dir}/")
   print("  " + "─" * 44)


   confirm = input("  Press Enter to continue (Type cancel to cancel): ").strip().lower()
   if confirm == 'cancel':
       print("\n  Process has been cancelled.")
       sys.exit(0)


   # --- Pipeline'ı Çalıştır ---
   print()
   run_pipeline(image_path, output_dir=output_dir, show_display=True)




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
# Argümansız çalıştırılırsa interaktif mod, argümanlı çalıştırılırsa
# klasik argparse modu ile çalışır.
def main():
   if len(sys.argv) == 1:
       interactive_mode()
   else:
       args = parse_arguments()
       run_pipeline(args.image, output_dir=args.output_dir, show_display=not args.no_display)


if __name__ == "__main__":
   main()

