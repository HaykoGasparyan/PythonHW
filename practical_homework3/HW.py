import cv2
import numpy as np


# Problem 1

image = cv2.imread('pic1.jpg')
cv2.imshow('pic1', image)


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imshow('RGB', image_rgb)
cv2.imshow('HSV', image_hsv)
cv2.imshow('LAB', image_lab)
cv2.imshow('GRAY', image_gray)

cv2.waitKey(0)

# Problem 2

image = cv2.imread('pic1.jpg')
b,g,r = cv2.split(image)

blank = np.zeros(shape=image.shape[:2], dtype='uint8')

blue = cv2.merge([b, blank, blank])
green = cv2.merge([blank, g, blank])
red = cv2.merge([blank, blank, r])

cv2.imshow('b_gray', b)
cv2.imshow('g_gray', g)
cv2.imshow('r_gray', r)

cv2.imshow('b_colored', blue)
cv2.imshow('g_colored', green)
cv2.imshow('b_colored', red)


cv2.waitKey(0)

# Problem 3

image2 = cv2.imread('pic2.jpg')
cv2.imshow('pic2', image2)

image_av_blur = cv2.blur(image2, (3,3))
bilateral_filter_1 = cv2.bilateralFilter(image2, 15, 15, 15)
bilateral_filter_2 = cv2.bilateralFilter(image2, 200, 15, 15)
bilateral_filter_3 = cv2.bilateralFilter(image2, 15, 200, 15)
bilateral_filter_4 = cv2.bilateralFilter(image2, 15, 15, 200)

# Compared with average blurring, all bilateral blurrings have preserved edges. But I found out also
# that increasing sigma color is preserving edges with high gradient only.  

cv2.imshow('Average Blur', image_av_blur)
cv2.imshow('Bilateral Blur 1', bilateral_filter_1)
cv2.imshow('Bilateral Blur 2', bilateral_filter_2)
cv2.imshow('Bilateral Blur 3', bilateral_filter_3)
cv2.imshow('Bilateral Blur 4', bilateral_filter_4)

cv2.waitKey(0)

# Problem 4

image2 = cv2.imread('pic2.jpg')
cv2.imshow('pic2', image2)

blank = np.zeros(image2.shape[:2], dtype = 'uint8')
mask = cv2.circle(blank, (image2.shape[1]//2, image2.shape[0]//2), 70, (255,0,0), -1)

masked_image = cv2.bitwise_and(image2, image2, mask=mask)
cv2.imshow('result', masked_image) 
cv2.waitKey(0)


# Problem 5

blank = np.zeros(shape=(200,200))
rectangle = cv2.rectangle(blank.copy(), (15, 15), (185, 185), 0.5, -1)
circle = cv2.circle(blank.copy(), (100, 100), 100, 0.5, -1)

shape1 = cv2.bitwise_xor(rectangle, circle)
shape2 = cv2.bitwise_or(rectangle, circle)


blank2 = np.zeros(shape=(200,200,3))
rectangle2 = cv2.rectangle(blank2.copy(), (15, 15), (185, 185), (147/255, 20/255, 255/255), -1)
circle2 = cv2.circle(blank2.copy(), (100, 100), 100, (147/255, 20/255, 255/255), -1)
shape3 = cv2.bitwise_xor(rectangle2, circle2)

cv2.imshow('Shape 1', shape1)
cv2.imshow('Shape 2', shape2)
cv2.imshow('Shape 3', shape3)


cv2.waitKey(0)

