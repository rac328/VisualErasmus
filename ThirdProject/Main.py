import numpy as np
import cv2 as cv
import glob

images = glob.glob('ThirdProject/photos/*.jpg')

for fname in images:
    img = cv.imread(fname)

    corners, R = harris_detector(img)


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                    # OpenCV
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    img1 = img.copy()
    img1[corners == 255] = [0, 0, 255]

    img2 = img.copy()
    img2[dst > 0.01 * dst.max()] = [0, 255, 0]

    cv.imshow('Manual Harris', img1)
    cv.imshow('OpenCV Harris', img2)

    cv.waitKey(0)

cv.destroyAllWindows()