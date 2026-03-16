import cv2 as cv
import numpy as np

REPLACEMENT_COLOR = (0, 255, 0)   # green
INITIAL_LOWER_HSV = (0, 80, 60) # lower hsv of the rank
INITIAL_UPPER_HSV = (25, 255, 255) # upper hsv of the rank
# https://opencv.org/color-spaces-in-opencv/
# H de 0 a 179. 0 color rojo. Queremos incluir colores mas anaranjados/marron. (25)
# S = 0 grisaceo. S=255 rojo puro. Queremos valores altos porque sino se incluirían reflejos y zonas blancas de la imagen
# V = 0 negro puro. V= 255 el más claro. Nuestro color no es muy socuro, en algunas imagenes un poco mas, por eso empezamos desde 60.
# S entre 80 y 255 son colores bastante saturados
# V entre 60 y 255 colores no muy oscuros

# Este rango de colores es el que vamos a cambiar a verde.

def create_hsv_range():
    """inferior and superior limits HSV."""
    lower = np.array(INITIAL_LOWER_HSV, dtype=np.uint8)  # creation of array
    upper = np.array(INITIAL_UPPER_HSV, dtype=np.uint8)
    return lower, upper


def detect_color_regions(bgr_image, lower_hsv, upper_hsv): #bgr_immage: original img
    """Thresholding HSV: binary mask of chosen color."""
    hsv = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV) # original img to hsv
    mask = cv.inRange(hsv, lower_hsv, upper_hsv) # apply the umbral to the hsv image.
    # La mask es de colores rojos/anaranjados. Osea la mask son los colores rojos y los otros colores son ahora 0.
    # 255 son los colores rojos
    return mask


def apply_color_replacement(bgr_image, mask, replacement_bgr=REPLACEMENT_COLOR):
    """ mask==255 to replacement_bgr."""
    result = bgr_image.copy() # copy of the original image
    result[mask == 255] = replacement_bgr # Todos los 255 son cambiados por color verde.
    return result


