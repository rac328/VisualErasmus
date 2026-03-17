from SecondProject import shapeDetection
import cv2 as cv
import glob
import os

images = glob.glob('shapes/*.png')
print(f" {len(images)} images")

dist, mtx, newcameramtx = shapeDetection.load_calibration()

for i, file in enumerate(images):
    img = cv.imread(file)
    h, w = img.shape[:2]
    fx = newcameramtx[0, 0]
    fy = newcameramtx[1, 1]
    focal_length = (fx + fy) / 2

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = shapeDetection.detect_circles(gray)
    shapeDetection.draw_shapes(img, circles, focal_length,47)

    cv.imwrite(f'Output_{i + 1:02d}.png', img)
    print(f" {len(circles)} circles detected")

print("Completed")

