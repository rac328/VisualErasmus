import numpy as np
import cv2 as cv
import glob
import pickle
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*5,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('SecondProject/photos/*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    print(ret)
    print("foto" + fname)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(250)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


ruta_archivo = os.path.join('SecondProject/savedata', 'mtx.npy')
np.save(ruta_archivo, mtx)

path1 = os.path.join('SecondProject/photos', 'photo_0001.png')
img = cv.imread(path1, cv.IMREAD_COLOR)
h = img.shape[0]
w = img.shape[1]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

ruta_archivo = os.path.join('SecondProject/savedata', 'newcameramtx.npy')
np.save(ruta_archivo,dist)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

cv.destroyAllWindows()