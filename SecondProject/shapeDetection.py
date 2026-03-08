import cv2
import numpy as np
import sys

dist, mtx, newcameramtx = 0
DISTANCE_CM = 30.0
CIRCLE_PARAMS = dict(
    method=cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=50,  # Canny edge threshold
    param2=30,  # Accumulator threshold
    minRadius=10,
    maxRadius=200
)


def load_calibration(filepath):
    """Load camera matrix and distortion coefficients from a numpy .npz file."""
    try:
        data = np.load(filepath)
        camera_matrix = data['mtx']
        dist_coeffs = data['dist']
        print("Calibration loaded successfully.")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        sys.exit(1)




def detect_circles(gray):
    # los ** son para pasar los argumentos de una vez
    circles = cv2.HoughCircles(gray, **CIRCLE_PARAMS)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)  # (x, y, radius)
        return circles
    return []




def pixels_to_cm(size_pixels, focal_length, distance_cm):
    return (size_pixels * distance_cm) / focal_length
