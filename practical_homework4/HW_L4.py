import cv2 as cv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
#Problem 1

img = cv.imread('pic1.jpg') 
#cv.imshow('pic1', img)
#cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray_hist = cv.calcHist([gray], [0], None, [256], [0,256]) 
gray_hist = [i[0] for i in gray_hist]

mpl.use('tkagg')
x = np.arange(256)
plt.plot(x,gray_hist)
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.show()


#Problem 2

colors = ('b', 'g', 'r')

for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256]) 
    mpl.use('tkagg') #backend for using matplotlib with any shell
    x = np.arange(256)
    plt.plot(x,hist, color=col)
    plt.title('Color Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')

plt.show()

"""

# Problem 3

img = cv.imread('pic1.jpg') 
cv.imshow('pic1', img) 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

threshold, thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY) 
cv.imshow('By Hand', thresh)
 
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 17, 0)
cv.imshow('Adaptive Mean', adaptive_thresh) 


adaptive_thresh_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, 0)
cv.imshow('Adaptive Gaussian', adaptive_thresh_gaussian) 
cv.waitKey(0)


# Problem 4

img = cv.imread('pic2.jpg') 
cv.imshow('pic2', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(9, 9),0)

canny = cv.Canny(gray, 150, 175)
canny_bl = cv.Canny(blur, 200, 90)

cv.imshow('raw canny', canny)
cv.imshow('blurred canny', canny_bl)

cv.waitKey(0)
