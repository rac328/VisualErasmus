from SecondProject import shapeDetection
import cv2 as cv
import glob
import generalFunctions

images = glob.glob('shapes/*.png')

dist, mtx, newcameramtx = shapeDetection.load_calibration()

for file in images:
    img = cv.imread(file)
    #print(img.shape)
    #h, w = img.shape[:2]
    #print(h, w)
    #hf=round(h/2.5)
    #wf=round(w/2.5)
    # print(hf,wf)
    #img = cv.resize(img, (wf, hf))
    # Use average focal length from the new (optimized) camera matrix
    fx = newcameramtx[0, 0]
    fy = newcameramtx[0, 1]
    focal_length = (fx + fy) / 2
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    circles = shapeDetection.detect_circles(gray)

    shapeDetection.draw_shapes(img, circles, focal_length, 30)

    # generalFunctions.displayImage(img) # no nos funcionaba

    generalFunctions.displayImage(img)
    cv.waitKey(5000)
    cv.destroyAllWindows()

    #print(array)