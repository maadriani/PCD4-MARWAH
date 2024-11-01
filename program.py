import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import histogram # type: ignore

def histogram_equalization(image):
    # Menghitung histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    
    # Menghitung CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalisasi CDF
    
    # Menerapkan ekualisasi histogram
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return image_equalized.reshape(image.shape)

# Membaca citra
image_path = 'C:\\Users\\marwa\\Pictures\\turkey.jpg'  # Ganti dengan path citra Anda
image = imageio.imread(image_path)

# Melakukan ekualisasi histogram
image_equalized = histogram_equalization(image)

# Menampilkan hasil
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Citra Awal')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Citra Setelah Ekualisasi')
plt.imshow(image_equalized, cmap='gray')
plt.show()