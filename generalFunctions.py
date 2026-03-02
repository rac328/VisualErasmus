import os
import cv2 as cv
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def displayImage(img):

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    backend = matplotlib.get_backend().lower()

    if backend.startswith('agg') or not has_display:
        # No hay GUI: no mostrar y no guardar preview
        print(f"Entorno sin servidor gráfico (`{matplotlib.get_backend()}`) — no se mostrará ni se guardará ninguna vista previa.")
        return

    # Entorno gráfico: convertir BGR->RGB y mostrar con matplotlib
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def reshape_image(principal_image, image_to_reshape, interpolation=cv.INTER_CUBIC):
    h, w = principal_image.shape[:2]
    reshaped = cv.resize(image_to_reshape, (w, h), interpolation=interpolation)
    return reshaped

def rotate(image):
    h = image.shape[0]
    w = image.shape[1]
    channels = image.shape[2]
    image_result = np.zeros((w, h, channels), dtype=image.dtype)

    for i in range(h):
        for j in range(w):
            image_result[j, i] = image[i, j]

    return image_result


def sharpen_image_strong(image):
    """
    Applies a sharpen kernel to the recieved image
    """
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]], dtype=np.float32)

    sharpened = cv.filter2D(image, -1, kernel)
    return sharpened

def extract_red_channel(image):
    # Crear imagen de ceros con mismo tamaño y tipo
    red_only = np.zeros_like(image)
    # Asignar solo el canal rojo (índice 2 en BGR)
    red_only[:, :, 2] = image[:, :, 2]
    return red_only

def print_info(img, imgPath):
    print("Tipo de imagen: "+img.dtype.str)
    height = img.shape[0]
    width = img.shape[1]

    print(f"Image dimensions: {width} x {height} pixels")
    print(f"Total pixels: {width * height}")

    file_size_bytes = os.path.getsize(imgPath)
    print(f"File size: {file_size_bytes} bytes")