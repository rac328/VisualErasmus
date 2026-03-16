import glob
import os
import cv2 as cv
from SecondProject import ColorFilter

IMAGE_FOLDER = "ColorsFilter"

def main():
    print("Current Work Directory:", os.getcwd())
    print("Searching images in:", IMAGE_FOLDER)

    image_paths = sorted( glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))

    if not image_paths:
        print(f" Not found Images in: {IMAGE_FOLDER}")
        return


    lower_hsv, upper_hsv = ColorFilter.create_hsv_range() # create the color range

    for i, path in enumerate(image_paths):
        img = cv.imread(path)
        if img is None:
            print(f"Can't read: {path}")
            continue

        mask = ColorFilter.detect_color_regions(img, lower_hsv, upper_hsv) # detect color regions cambia rgb a hsv
        # aplica el umbral (thresholding) y crea una imagen binaria de 0 y 255. Donde 255 son los colores del umbral
        # seleccionados, y 0 el resto

        result = ColorFilter.apply_color_replacement(img, mask) # aplicamos la función en la que se hace cop

        out_name = f"output_{i:04d}.png"
        cv.imwrite(out_name, result)
        print("Guardado:", out_name)

if __name__ == "__main__":
    main()
