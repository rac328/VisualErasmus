# python
import os
import cv2 as cv
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

numPhotos = 2

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

def displayMosaico(numPhotos):
    """
    Crea un mosaico 2x2 con imagen1 e imagen2 redimensionadas al mismo tamaño.
    """
    path1 = os.path.join('../Photos', 'imagen1.jpg')
    path2 = os.path.join('../Photos', 'imagen2.jpg')
    img = cv.imread(path1, cv.IMREAD_COLOR)
    img1 = cv.imread(path2, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"displayMosaico: imagen no encontrada: {path1}")
    if img1 is None:
        raise FileNotFoundError(f"displayMosaico: imagen no encontrada: {path2}")

    # Usar reshape_image para redimensionar img1 al tamaño de img
    #img1_resized = reshape_image(img, img1)

    # EXERCISE 3
    img_sharpened = sharpen_image_strong(img)
    # EXERCISE 4
    img_rotated = reshape_image(img, rotate(img))
    # EXERCISE 5
    img_redChannel = extract_red_channel(img)
    # EXERCISE 6
    print_info(img, path1)
    h = img.shape[0]
    w = img.shape[1]
    channels = img.shape[2]

    # Crear arreglo con mismo dtype que la imagen
    mosaico = np.zeros((h * numPhotos, w * numPhotos, channels), dtype=img.dtype)

    # EXERCISE 2
    imgArray = [img_sharpened, img_rotated, img_redChannel, img]
    for i in range(numPhotos):
        for j in range(numPhotos):
            idx = i * numPhotos + j  # Convertir (i,j) a índice lineal: 0,1,2,3
            if idx < len(imgArray):
                mosaico[i * h:(i + 1) * h, j * w:(j + 1) * w] = imgArray[idx]

    return mosaico

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
    plt.title('FIRTS ASSIGMENT RAÚL CABRERA ROZALÉN')
    plt.axis('off')
    plt.show()




def displayIMG1():
    path = os.path.join('../Photos', 'imagen1.jpg')
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagen no encontrada: `{path}`")

    # Detectar si hay servidor gráfico disponible
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    backend = matplotlib.get_backend().lower()

    if backend.startswith('agg') or not has_display:
        # No hay GUI: no mostrar y no guardar preview
        print(f"Entorno sin servidor gráfico (`{matplotlib.get_backend()}`) — no se mostrará ni se guardará ninguna vista previa.")
        return

    # Entorno gráfico: convertir BGR->RGB y mostrar con matplotlib
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Imagen 1')
    plt.axis('off')
    plt.show()

