import cv2
import numpy as np
import sys
import os

DISTANCE_CM = 30.0
CIRCLE_PARAMS = dict(
    method=cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=250,  # Canny edge threshold
    param2=50,  # Accumulator threshold
    minRadius=5,
    maxRadius=300
)


def load_calibration():
    """Load camera matrix and distortion coefficients from a numpy .npz file."""
    try:
        ruta_archivo = os.path.join('savedata', 'dist.npy')
        dist = np.load(ruta_archivo)
        ruta_archivo = os.path.join('savedata', 'mtx.npy')
        mtx = np.load(ruta_archivo)
        ruta_archivo = os.path.join('savedata', 'newcameramtxreal.npy')
        newcameramtx = np.load(ruta_archivo)

        return dist, mtx, newcameramtx
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

def draw_shapes(frame, circles, focal_length, distance_cm):

    for (x, y, r) in circles:
        #draws the circle:
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        #draws the center:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        cv2.putText(frame, f"Circle", (x-40, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        diameter_pix = 2 * r

        #Calculation distance from camera to circle knowing its diameter
        #distance_cm=distance_cms(3.5,focal_length,diameter_pix)


        diameter_cm = pixels_to_cm(diameter_pix, focal_length, distance_cm)
        cv2.putText(frame, f"d={diameter_cm:.1f}cm", (x-40, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)


def pixels_to_cm(size_pixels, focal_length, distance_cm):
    return (size_pixels * distance_cm) / focal_length

#def distance_cms(real_diameter_cm,focal_length,size_pixels):
    return (real_diameter_cm * focal_length) / size_pixels

def detect_squares(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # para detectar las bordes
    edges = cv2.Canny(blurred, 50, 150)
    
    # contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for cnt in contours:
        # aproximar el contorno al poligonl
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        #como es un cuadrado, tiene que haber cuatro vértices
        if len(approx) == 4:
            # Compute side lengths and angles
            pts = approx.reshape(-1, 2)
            edges_dist = []
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i+1)%4]
                edges_dist.append(np.linalg.norm(p2 - p1))
            
            # comprueba si todos los bordes miden lo mismo 
            max_side = max(edges_dist)
            min_side = min(edges_dist)
            if max_side / min_side < 1.2:  # simplemente es para la tolerancia 
                squares.append(approx)
    
  #dibuja los rectángulos reconocidos 
    result = img.copy()
    cv2.drawContours(result, squares, -1, (0, 255, 0), 3)
    cv2.imshow('Detected Squares', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return squares
