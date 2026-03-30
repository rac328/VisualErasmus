import numpy as np
import cv2 as cv

#Convolution
def convolve(image, kernel):
    h, w = image.shape
    kh, kw = (3,3)

    pad_h = 1
    pad_w = 1

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant') #Adds white borders to do Kernel
    output = np.zeros_like(image)#Creates Output image

    for i in range(h):
        for j in range(w): #Goes through every pixel
            region = padded[i:i+kh, j:j+kw] #extracts 3*3 per pixel
            output[i, j] = np.sum(region * kernel) #xd

    return output

#Gaussian Kernel
def gaussian_kernel(size=3, sigma=1):
    #size (int): Side length of the square kernel.
    #sigma (float): Standard deviation of the Gaussian distribution.

    #creates an array of evenly spaced numbers over a specified interval.
    ax = np.linspace(-(size//2), size//2, size)
    #creates the matrix from the previous array
    xx, yy = np.meshgrid(ax, ax)

    #gaussian formula
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    #normalization of the kernel
    kernel /= np.sum(kernel)

    return kernel


#Harris Detector
def harris_detector(img, k=0.04, threshold=0.01):

    # 1. Grayscale to avoid choosing which rgb channel use
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 2. Sobel kernels
    sobel_x = np.array([   #Vertical border detection
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([    #Horizontal border detection
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # 3. Gradients
    Ix = convolve(gray, sobel_x) #Derivative of the image to find borders
    Iy = convolve(gray, sobel_y)

    # 4. Products
    Ixx = Ix * Ix #Calculates points to create Matrix ([Ixx,Ixy],[Ixy,Iyy])
    Iyy = Iy * Iy #For every pixel
    Ixy = Ix * Iy

    # 5. Gaussian smoothing
    g = gaussian_kernel(3, 1)

    Sxx = convolve(Ixx, g)
    Syy = convolve(Iyy, g)
    Sxy = convolve(Ixy, g)

    # 6. Harris response
    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2) #Detects if two borders (vertical and horizontal) are detected at the same time
    #Value big = corner
    # 7. Thresholding
    corners = np.zeros_like(R)
    corners[R > threshold * R.max()] = 255

    return np.uint8(corners), R