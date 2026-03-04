import cv2
import numpy as np


# --- Köşe Noktalarını Tutarlı Bir Şekilde Sıralama ---
# Rastgele sırada gelen 4 köşe noktasını (x, y) koordinatlarına göre;
# [Üst-Sol, Üst-Sağ, Alt-Sağ, Alt-Sol] şeklinde sabit bir sıraya dizer.
def order_points(pts):
    # 4 nokta için boş bir matris oluştur
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    # Üst-sol nokta en küçük toplama (x+y), alt-sağ en büyük toplama sahiptir
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # Üst-sağ nokta en küçük farka (y-x), alt-sol en büyük farka sahiptir
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]  # Top-right
    rect[3] = pts[np.argmax(d)]  # Bottom-left

    return rect


# --- Görüntü İçindeki Dökümanı Tespit Etme ---
# Gri tonlama, bulanıklaştırma ve Canny kenar algılama kullanarak
# kağıdın sınırlarını bulur ve en büyük 4 köşeli alanı (konturu) yakalar.
def detect_document(image):
    # Ön işleme: Kenarları daha iyi yakalamak için griye çevir ve gürültüyü azalt
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny algoritması ile kenarları belirle
    edged = cv2.Canny(blurred, 50, 200)

    # Kenarlardaki küçük boşlukları kapatmak için genişletme (dilation) uygula
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    # Tüm konturları (çizgisel alanları) bul
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Konturları alanlarına göre büyükten küçüğe sırala ve ilk 5'i al
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    document_contour = None

    for contour in contours:
        # Konturu basitleştir (poligon yaklaşımı)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Eğer basitleştirilmiş kontur 4 köşeye sahipse, dökümanı bulduk demektir
        if len(approx) == 4:
            document_contour = approx
            break

    if document_contour is not None:
        return document_contour.reshape(4, 2)

    return None


# --- Perspektif Dönüşümü ve Kuş Bakışı Görünüm ---
# Tespit edilen 4 noktayı kullanarak görüntüyü "gerer" (warp)
# ve dökümanı dikdörtgen formunda, tam karşıdan bakılıyormuş gibi yeniden oluşturur.
def four_point_warp(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Yeni görüntünün genişliğini hesapla (alt ve üst kenarların maksimumu)
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Yeni görüntünün yüksekliğini hesapla (sağ ve sol kenarların maksimumu)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Hedef noktalar (görüntünün yeni, düz köşeleri)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Dönüşüm matrisini hesapla ve perspektifi uygula
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


# --- Perspektif Düzeltme Ana Süreci ---
# Döküman algılama ve perspektif bükme (warp) işlemlerini koordine eder.
def correct_perspective(image):
    print("\n[STEP 1] Perspective Correction")
    print("  Detecting document boundaries...")

    # Köşeleri bul
    corners = detect_document(image)

    if corners is None:
        print("  WARNING: No document contour detected. Using original image.")
        print("  TIP: Ensure the document has clear edges against the background.")
        return image.copy(), False

    print(f"  Document corners found: {corners.tolist()}")

    # Bulunan köşelere göre görüntüyü düzelt
    warped = four_point_warp(image, corners)
    print(f"  Warped image size: {warped.shape[1]}x{warped.shape[0]}")

    return warped, True