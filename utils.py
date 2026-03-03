import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Görüntü Dosyasını Güvenli Bir Şekilde Yükleme ---
# Bu fonksiyon, verilen dosya yolunda resmin olup olmadığını kontrol eder
# ve OpenCV formatında belleğe yükler.
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


# --- İşlenen Görüntüyü Diske Kaydetme ---
# Belirtilen klasör yolu yoksa otomatik olarak oluşturur
# ve üzerinde çalışılan görüntüyü dosyaya yazar.
def save_image(path, image):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    cv2.imwrite(path, image)
    print(f"  [SAVED] {path}")


# --- Görüntüyü Ekrana Sığdırmak İçin Yeniden Boyutlandırma ---
# Çok büyük çözünürlüklü görsellerin ekrana düzgün sığması için
# en-boy oranını bozmadan genişliği sınırlar.
def resize_for_display(image, max_width=800):
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# --- Birden Fazla Görüntüyü Yan Yana Listeleme ---
# Matplotlib kullanarak birden fazla resmi tek bir satırda gösterir.
# OpenCV (BGR) ve Matplotlib (RGB) arasındaki renk farkını otomatik düzenler.
def display_images(images, titles, figsize=(16, 10), save_path=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        # Renkli resimleri RGB'ye çevirir, gri tonlamalıları olduğu gibi bırakır
        if len(img.shape) == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(display_img)
        else:
            ax.imshow(img, cmap="gray")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")

    plt.show()


# --- Detaylı Karşılaştırma Izgarası (Grid) Oluşturma ---
# İşleme aşamalarını (örneğin: orijinal, gri ton, eşikleme, temizleme)
# düzenli bir tablo (ızgara) şeklinde sunmak için kullanılır.
def display_comparison_grid(images_dict, figsize=(16, 12), save_path=None):
    n = len(images_dict)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (title, img) in enumerate(images_dict.items()):
        ax = axes[idx]
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Kullanılmayan boş kutucukları gizler
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Document Enhancement Pipeline — Stage Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [SAVED] {save_path}")

    plt.show()