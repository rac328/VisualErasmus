import cv2
import os
from pynput import keyboard
from ximea import xiapi

output_dir = "Photos"
cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(10000)  #microseconds
cam.start_acquisition()

photo_count = 0
capture_flag = False

def on_press(key):
    global capture_flag
    try:
        if key == keyboard.Key.space:
            capture_flag = True
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        return False

# Configurar listener del teclado
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Presiona ESPACIO para capturar fotos. Presiona ESC para salir.")

try:
    while True:
        if capture_flag:
            # Capturar imagen
            img = cam.get_image()
            img_array = img.get_image_data_numpy()

            # Guardar imagen
            photo_count += 1
            filename = os.path.join(output_dir, f"photo_{photo_count:04d}.png")
            cv2.imwrite(filename, img_array)
            print(f"Foto guardada: {filename}")

            capture_flag = False
finally:
    cam.stop_acquisition()
    cam.close_device()
